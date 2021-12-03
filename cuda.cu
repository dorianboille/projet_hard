#include <stdio.h>
#include <stdlib.h>
#include <cstdlib>




void MatrixInit(float *M, int n, int p){
    srand(time(NULL));
    for (int  i = 0; i <n*p; i++){
        M[i] = (float)rand()/RAND_MAX*2-1;
    };
}
void MatrixUnit(float *M, int n, int p){
    srand(time(NULL));
    for (int  i = 0; i <n*p; i++){
        M[i] = 1;
    };
}
void MatrixPrint(float *M, int n, int p){
    
    for (int  i = 0; i <n*p; i++){
        if (i%p==0){
            printf("\n");
        }
        printf("  %f  ", M[i]);
    };

}
void MatrixAdd(float *M1, float *M2, float *Mout, int n, int p){
    
    for (int  i = 0; i <n*p; i++){
        Mout[i] = M1[i] + M2[i];
    };

}
// fonction appellée par un thread ils "connaissent" donc leurs ids 
__global__ void cudaMatrixAdd(float *M1, float *M2, float *Mout, int n, int p){

        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid < n){
            Mout[tid] = M1[tid] + M2[tid];
        }

}
int main(){
    float *M, *M2;

    float *d_M, *d_M2, *d_M3;

    int n = 5,p=5;
    //alocation de mémoire accessible par le CPU 
    M  = (float*)malloc(sizeof(float) * n*p);
    M2  = (float*)malloc(sizeof(float) * n*p);

    //alocation de mémoire accessible par le GPU 
    cudaMalloc((void**)&d_M, sizeof(float)*n*p);
    cudaMalloc((void**)&d_M2, sizeof(float)*n*p);
    cudaMalloc((void**)&d_M3, sizeof(float)*n*p);

    MatrixUnit(M,n,p);
    MatrixUnit(M2,n,p);

    //MatrixAdd(M,M2,M3, n,p);
    //MatrixPrint(M3,n,p);

    // On transfert la mémoire du CPU vers le GPU 
    cudaMemcpy(d_M, M, sizeof(float) * n*p, cudaMemcpyHostToDevice);
    cudaMemcpy(d_M2, M2, sizeof(float) * n*p, cudaMemcpyHostToDevice);

    // on a un bloc par ligne et un thread par colonne
    int block_size = n;
    int grid_size = p;

    cudaMatrixAdd<<<grid_size,block_size>>>(d_M, d_M2, d_M3, n,p);
    MatrixPrint(d_M3,n,p);

    cudaDeviceSynchronize();

    return 0;
}
