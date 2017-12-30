#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>

#define BLOCK_SIZE 16
#define THREADS_PER_BLOCK 8


__global__ void add2d(int *A, int *B, int *C, int colunas, int linhas) {

  // Calcula linha e coluna.
  int coluna = BLOCK_SIZE * blockIdx.x + threadIdx.x;
  int linha = BLOCK_SIZE * blockIdx.y + threadIdx.y;

  if (linha < linhas && coluna < colunas){
     int index = colunas * linha + coluna;
     C[index] = A[index] + B[index];
  }

    // Usando BlockDIM no modo vetorizado
    //  unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
    //if(colunas <= blockIdx.x && linhas <= threadIdx.x)
    //    C[index] = A[index] + B[index];

}

int main()
{
    // C  pias de A B e C da CPU.
    int *A, *B, *C;
    int i, j;

    //Input
    int linhas, colunas;
    scanf("%d", &linhas);
    scanf("%d", &colunas);

    //Alocando mem  ria na CPU
    A = (int *)malloc(sizeof(int)*linhas*colunas);
    B = (int *)malloc(sizeof(int)*linhas*colunas);
    C = (int *)malloc(sizeof(int)*linhas*colunas);

    //Inicializar Matrizes A e B
    for(i = 0; i < linhas; i++){
        for(j = 0; j < colunas; j++){
            A[i*colunas+j] =  B[i*colunas+j] = i+j;
        }
    }

    //========= CUDA programming configuration ===========//

    // C  pias da GPU para A, B e C.
    int *d_a, *d_b, *d_c;
    int size = sizeof(int) * colunas * linhas;
    //cudaError_t cuda_error;

    // Aloca o de espa  o para as c  pias de A e B e armazenamento do resultado em C.
    cudaMalloc( (void **)&d_a, size);
    cudaMalloc( (void **)&d_b, size);
    cudaMalloc( (void **)&d_c, size);

    // Copia as entradas no dispositivo.
    cudaMemcpy(d_a, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, B, size, cudaMemcpyHostToDevice);

    //Realiza o dimensionamento  de blocos para 2 dimensoes
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

     //Realiza o dimensionamento da Grid para 2 dimensoes
    dim3 dimGrid(ceil((float) colunas/BLOCK_SIZE), ceil((float)linhas/BLOCK_SIZE));

    // Executa a funcao ADD 2d na GPU
    add2d <<< dimGrid, dimBlock >>> (d_a, d_b, d_c, colunas, linhas);

    // Copia o resultado armazenado em C para a Mem  ria da CPU (HOST).
    cudaMemcpy(C, d_c, size, cudaMemcpyDeviceToHost);

    //====================================================//

    long long int somador=0;
    // Manter esta computa    o na CPU - C  lcula o resultado.
    for(i = 0; i < linhas; i++){
        for(j = 0; j < colunas; j++){
            somador+=C[i*colunas+j];
        }
    }

    // Imprime o resultado
    printf("%lli\n", somador);

    // Limpa e libera mem  ria reservada no dispositivo (GPU)
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // Libera mem  ria utilizada para CPU (HOST)
    free(A);
    free(B);
    free(C);
}
