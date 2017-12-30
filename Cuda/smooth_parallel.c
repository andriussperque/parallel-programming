#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include <cuda.h>

#define MASK_WIDTH 5

#define COMMENT "Smoothing_image_GPU"
#define RGB_COMPONENT_COLOR 255
#define BLOCK_SIZE 256

typedef struct {
    unsigned char red, green, blue;
} PPMPixel; 

typedef struct {
    int x, y;
    PPMPixel *data;
} PPMImage;

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

void writePPM(PPMImage *img) {

    fprintf(stdout, "P6\n");
    fprintf(stdout, "# %s\n", COMMENT);
    fprintf(stdout, "%d %d\n", img->x, img->y);
    fprintf(stdout, "%d\n", RGB_COMPONENT_COLOR);

    fwrite(img->data, 3 * img->x, img->y, stdout); 
    fclose(stdout);
}


__global__ void smoothing_GPU_Parallel(PPMPixel *d_image_output, PPMPixel *d_image, int imageY, int imageX, int n) {

    // Shared Memory (didn't worked)
    __shared__ PPMPixel s[(BLOCK_SIZE) * (BLOCK_SIZE)];
    int total_red, total_blue, total_green, y, x;

    int idxTotal = blockDim.x * blockIdx.x + threadIdx.x;
    
    //  Verificação se o índice estava correto.
    //if (idxTotal < 100)   
    //    printf("index: %d \n",  idxTotal);
 
    // Transfer data to shared memory 
    // s[idxTotal] = d_image[idxTotal];

    __syncthreads(); 

    total_red = total_blue = total_green = 0;

    for (y = ((int)idxTotal / (int)imageX) - ((MASK_WIDTH-1)/2); y <= (((int)idxTotal / (int)imageX ) + ((MASK_WIDTH-1)/2)); y++) {

        for (x = ((int)idxTotal % (int)imageX) - ((MASK_WIDTH-1)/2); x <= (((int)idxTotal % (int)imageX) + ((MASK_WIDTH-1)/2)); x++) {
           
            if (x >= 0 && y >= 0 && y < imageY && x < imageX) {
               
                total_red += d_image[(y * imageX) + x].red;
                total_blue += d_image[(y * imageX) + x].blue;
                total_green += d_image[(y * imageX) + x].green;
            }
        }
    }

    // Execute work
	//if ((idxTotal % imageX) >= 0 && (idxTotal / imageX)  >= 0 && (idxTotal / imageX)  < imageY && (idxTotal % imageX) < imageX) {

    //    s[idxTotal].red = total_red / (MASK_WIDTH*MASK_WIDTH);
    //    s[idxTotal].blue = total_blue / (MASK_WIDTH*MASK_WIDTH);
    //    s[idxTotal].green = total_green / (MASK_WIDTH*MASK_WIDTH);

    __syncthreads(); 

    d_image_output[idxTotal].red = total_red / (MASK_WIDTH*MASK_WIDTH);
    d_image_output[idxTotal].blue = total_blue / (MASK_WIDTH*MASK_WIDTH);
    d_image_output[idxTotal].green = total_green / (MASK_WIDTH*MASK_WIDTH);
        
      // if (idxTotal < 100)
        //  printf("thread_num: %d, r: %d, b: %d, g: %d \n", idxTotal, d_image_output[idxTotal].red, d_image_output[idxTotal].blue, d_image_output[idxTotal].green);

    
    /*  if (idxTotal == 2  && (blockIdx.x == 0) && (threadIdx.x < 10)) {
        printf("thread_num: %d, r: %d, b: %d, g: %d \n", idxTotal, d_image_output[idxTotal].red, d_image_output[idxTotal].blue, d_image_output[idxTotal].green);
    } */
}


int main(int argc, char *argv[]) {

    if( argc != 2 ) {
        printf("Too many or no one arguments supplied.\n");
    }

    double t_start, t_end;
    int i;
    char *filename = argv[1];

    PPMImage *image = readPPM(filename);
    PPMImage *image_output = readPPM(filename);

    float n = image->y * image->x;

    // CUDA CODE 
    {
        PPMPixel *d_image;
        PPMPixel *d_image_output;

        if (cudaMalloc( (void **) &d_image, sizeof(PPMPixel) * n) != cudaSuccess) {
            printf("error cudaMalloc buffer image");
            exit(1);
        }
        
        if (cudaMalloc( (void **) &d_image_output, sizeof(PPMPixel) * n) != cudaSuccess) {
            printf("error cudaMalloc buffer image result");
            exit(1); 
        }
 
        cudaMemcpy(d_image, image->data, sizeof(PPMPixel) * n, cudaMemcpyHostToDevice);
        
        //Realiza o dimensionamento  de blocos para 2 dimensoes
       // dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

        //Realiza o dimensionamento da Grid para 2 dimensoes
        //dim3 dimGrid(ceil((float)BLOCK_SIZE/image->x), ceil((float)BLOCK_SIZE/image->y));
       // dim3 dimGrid(3,3);

        smoothing_GPU_Parallel <<<ceil(n/(float)BLOCK_SIZE)+1, BLOCK_SIZE>>> (d_image_output, d_image, image->y, image->x, n);
    
        //smoothing_GPU_Parallel_2 <<<ceil((float)BLOCK_SIZE/n), BLOCK_SIZE>>> (d_image_output, d_image, image->y, image->x, n);

        cudaMemcpy(image_output->data, d_image_output, sizeof(PPMPixel) * n, cudaMemcpyDeviceToHost);

        cudaFree(d_image); 
	    cudaFree(d_image_output);

    }
    writePPM(image_output);

    free(image);
    free(image_output);
}
