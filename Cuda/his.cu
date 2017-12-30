#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include <cuda.h>


#define COMMENT "Histogram_GPU"
#define RGB_COMPONENT_COLOR 255
#define BLOCK_SIZE 1024

// STRUCTS
typedef struct {
	unsigned char red, green, blue;
} PPMPixel;

typedef struct {
	int x, y;
	PPMPixel *data;
} PPMImage;


//PROTOTYPES
//__global__ void calc_hist(PPMImage *d_a, float *h, int n);

double rtclock()
{
    struct timezone Tzp;
    struct timeval Tp;
    int stat;
    stat = gettimeofday (&Tp, &Tzp);
    if (stat != 0) printf("Error return from gettimeofday: %d",stat);
    return(Tp.tv_sec + Tp.tv_usec*1.0e-6);
}


static PPMImage *readPPM(const char *filename) {
	char buff[16];
	PPMImage *img;
	FILE *fp;
	int c, rgb_comp_color;
	fp = fopen(filename, "rb");
	if (!fp) {
		fprintf(stderr, "Unable to open file '%s'\n", filename);
		exit(1);
	}

	if (!fgets(buff, sizeof(buff), fp)) {
		perror(filename);
		exit(1);
	}

	if (buff[0] != 'P' || buff[1] != '6') {
		fprintf(stderr, "Invalid image format (must be 'P6')\n");
		exit(1);
	}

	img = (PPMImage *) malloc(sizeof(PPMImage));
	if (!img) {
		fprintf(stderr, "Unable to allocate memory\n");
		exit(1);
	}

	c = getc(fp);
	while (c == '#') {
		while (getc(fp) != '\n')
			;
		c = getc(fp);
	}

	ungetc(c, fp);
	if (fscanf(fp, "%d %d", &img->x, &img->y) != 2) {
		fprintf(stderr, "Invalid image size (error loading '%s')\n", filename);
		exit(1);
	}

	if (fscanf(fp, "%d", &rgb_comp_color) != 1) {
		fprintf(stderr, "Invalid rgb component (error loading '%s')\n",
				filename);
		exit(1);
	}

	if (rgb_comp_color != RGB_COMPONENT_COLOR) {
		fprintf(stderr, "'%s' does not have 8-bits components\n", filename);
		exit(1);
	}

	while (fgetc(fp) != '\n')
		;

	img->data = (PPMPixel*) malloc(img->x * img->y * sizeof(PPMPixel));

	if (!img) {
		fprintf(stderr, "Unable to allocate memory\n");
		exit(1);
	}

	if (fread(img->data, 3 * img->x, img->y, fp) != img->y) {
		fprintf(stderr, "Error loading image '%s'\n", filename);
		exit(1);
	}

	fclose(fp);
	return img;
}

// Kernel - Função que será executada na GPU
__global__ void calc_hist(PPMPixel *d_a, float *h, int n) {
	
	int idx_h;
	// Padrão comum para cálculo do índice único de cada thread.
	int idx =  blockDim.x * blockIdx.x + threadIdx.x;	

	// Threads a mais não deveram participar dos incrementos que irão gerar o histograma.
	if (idx < n) {

		// Através dos valores de Red, Green e Blue do pixel representado pela thread,
		// é cálculado o índice para o histograma, o qual deve ser incrementado, contabilizando assim 
		// a quantidade de pixels com o mesmo valor RGB.
		idx_h = (d_a[idx].red * 16) + (d_a[idx].green * 4) + (d_a[idx].blue);
		atomicAdd((float * )&h[idx_h],1);
	}
	__syncthreads();

}

void Histogram(PPMImage *image, float *h) {

	cudaEvent_t start, stop;
	cudaEvent_t startAlloc, stopAlloc;
	cudaEvent_t startMemCp, stopMemCp;
	cudaEvent_t startMemRec, stopMemRec;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventCreate(&startAlloc);
	cudaEventCreate(&stopAlloc);

	cudaEventCreate(&startMemCp);
	cudaEventCreate(&stopMemCp);

	cudaEventCreate(&startMemRec);
	cudaEventCreate(&stopMemRec);

	int i;
	float n = image->y * image->x;

	for (i = 0; i < n; i++) {
	
		image->data[i].red = floor((image->data[i].red * 4) / 256);
		image->data[i].blue = floor((image->data[i].blue * 4) / 256);
		image->data[i].green = floor((image->data[i].green * 4) / 256);
	}

	PPMPixel *d_a;
	float * d_h;

	cudaEventRecord(startAlloc);
	//tempo_GPU_offload_enviar
	if (cudaMalloc( (void **) &d_a, sizeof(PPMPixel) * n) != cudaSuccess) {
		printf("error cudaMalloc buffer");
		exit(1);
	}
    
	if (cudaMalloc( (void **) &d_h, sizeof(float) * 64) != cudaSuccess) {
		printf("error cudaMalloc hist");
		exit(1);
	}
	cudaEventRecord(stopAlloc);
	cudaEventSynchronize(stopAlloc);

	cudaEventRecord(startMemCp);
 	// Copia as entradas no dispositivo.
    cudaMemcpy(d_a, image->data, sizeof(PPMPixel) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_h, h, sizeof(float) * 64, cudaMemcpyHostToDevice);
	
	cudaEventRecord(stopMemCp);
	cudaEventSynchronize(stopMemCp);

	cudaEventRecord(start);
	// Executa a funcao que calcula os dados para o histograma. Sendo que cada thread irá tratar de um pixel.
	// Foi utilizado o número máximo de threads por bloco (1024). O número de blocos escolhido foi o de 
	// N pixels / número de threads por bloco
    calc_hist <<< ceil((float) n/BLOCK_SIZE), BLOCK_SIZE >>> (d_a, d_h, n);
	
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	cudaEventRecord(startMemRec);
    // Copia o resultado armazenado no device para a Memoria da CPU (HOST) d_h => h.
    cudaMemcpy(h, d_h, sizeof(float) * 64, cudaMemcpyDeviceToHost);
	
	cudaEventRecord(stopMemRec);
	cudaEventSynchronize(stopMemRec);

	// Realiza a normalização dos dados do histograma.
	for(i=0; i < 64; i++) 
		h[i] = h[i]/n;

	cudaFree(d_a);
	cudaFree(d_h);

	// Imprime Métricas do Código 
	/*
	float millisecondsAlloc = 0;
	cudaEventElapsedTime(&millisecondsAlloc, startAlloc, stopAlloc);
	printf("Allocation Execution time: %f\n", millisecondsAlloc/1e3);

	float millisecondsMemCp = 0;
	cudaEventElapsedTime(&millisecondsMemCp, startMemCp, stopMemCp);
	printf("Memomry Send Execution time: %f\n", millisecondsMemCp/1e3);

  	float milliseconds = 0;
  	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("Kernel Execution time: %f\n", milliseconds/1e3);

	float millisecondsRec = 0;
  	cudaEventElapsedTime(&millisecondsRec, startMemRec, stopMemRec);
	printf("Copy Received Execution time: %f\n", milliseconds/1e3);

	float millisecondsTotal = millisecondsAlloc + millisecondsMemCp + milliseconds + millisecondsRec;
	printf("Total Execution time: %f\n", millisecondsTotal/1e3);
	*/
} 
 
int main(int argc, char *argv[]) {

	if( argc != 2 ) {
		printf("Too many or no one arguments supplied.\n");
	}

	double t_start, t_end;
	int i;
	char *filename = argv[1]; //Recebendo o arquivo!;
	
	//scanf("%s", filename);
	PPMImage *image = readPPM(filename);

	float *h = (float*)malloc(sizeof(float) * 64);

	//Inicializar h
	for(i=0; i < 64; i++) 
		h[i] = 0.0;

	t_start = rtclock();
	
	Histogram(image, h);
	
	t_end = rtclock();

	for (i = 0; i < 64; i++){
		printf("%0.3f ", h[i]);
	}
	printf("\n");
	//fprintf(stdout, "\n%0.6lfs\n", t_end - t_start);  
	free(h);
}


/***********
RA: ra189918 - Andrius Sperque
RESULTS (Table format):

entrada: ************************************ arq1.ppm *******************************************
tempo_serial: 0.322303s
tempo_GPU_criar_buffer: 0.001186
tempo_GPU_offload_enviar: 0.000634
tempo_kernel: 0.002781
tempo_GPU_offload_receber: 0.002781
GPU_total: 0.004621
speedup (tempo_serial / GPU_total): 0.322303s / 0.004621 = 69.747457

entrada: ************************************ arq2.ppm *******************************************
tempo_serial: 0.585311s
tempo_GPU_criar_buffer: 0.001291
tempo_GPU_offload_enviar: 0.001284
tempo_kernel: 0.008463
tempo_GPU_offload_receber: 0.008463
GPU_total: 0.011060
speedup (tempo_serial / GPU_total): 0.585311 / 0.011060 = 52.921428

entrada: ************************************ arq3.ppm *******************************************
tempo_serial: 1.679165s 
tempo_GPU_criar_buffer: 0.002209
tempo_GPU_offload_enviar: 0.005373
tempo_kernel:  0.034532
tempo_GPU_offload_receber: 0.034532
GPU_total: 0.042178
speedup (tempo_serial / GPU_total): 1.679165 / 0.042178 = 39.8113945
*/