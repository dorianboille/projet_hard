#include <stdio.h>
#include <stdlib.h>
#include <cstdlib>




void MatrixInit(float *M, int n, int p){
    srand(time(NULL));
    for (int  i = 0; i <n*p; i++){
        M[i] = (float)rand()/RAND_MAX*2-1;
    };
}

void MatrixPrint(float *M, int n, int p){
    
    for (int  i = 0; i <n*p; i++){
        if (i%p==0){
            printf("\n");
        }
        printf("  %1.1f  ", M[i]);
    };
    printf("\n\n");

}

// Cette fonction permet de créer une matrice de taille nxp remplie du même coefficient//
/*
    - En argument on fourni le pointeur vers la matrice à remplir
    - On donne aussi sa taille n lignes p colonnes 
    - On donne la valeure val correspondant à la valeure prise par tout les cofficients
*/
void MatrixUnit(float *M, int n, int p, float value){
    srand(time(NULL));
    for (int  i = 0; i <n*p; i++){
        M[i] = value;
    };
}


// Cette fonction permet d'obtenir un noyau identité //
/* 
    - En argumen t on donne le pointeur vers la matrice contenant les kernels.
    - On donne aussi la taille des kernels : dim 
      ainsi que leur nombre : prof 
    - On donne aussi la valeure prise par le , coefficient central des kernels.
*/

void Id_Kernel(float *M, int dim, int prof, int val){
    for (int  i = 0; i <dim*dim*prof; i++){
        M[i] = 0;
    }
    int index = 0;
    for (int  i = 0; i <dim*dim*prof; i++){
        
        if (i%(((dim*dim)-1)/2) == 0 && i!=0 && i%((dim*dim)-1)!=0)
        {   
            M[i+index]= val;
            index = index+1;
        }
    }
} 

// Cette fonction fait une convolution 2D

__global__ void cudaConv2D(float* M, float* kernel, float* Mout, int M_line, int M_col, int kernel_size, int nb_kernel){
    
    int offset = (kernel_size-1)/2;
    int out_line = M_line - offset;
    int out_col = M_col - offset;
    //Convolution d'une matrice par un kernel
    int lig = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float conv = 0.0;

    if (lig < out_line && col < out_col){
        int temp = M_line * M_col;

        for (int k_line = 0; k_line < kernel_size; k_line++) {
            for (int k_col = 0; k_col < kernel_size; k_col++) {
                for (int n_k = 0; n_k < nb_kernel; n_k++){
                    conv += M[(lig + k_line) * M_col + col + k_col + n_k * temp] * kernel[k_line * kernel_size + k_col + n_k * nb_kernel];
            
                }
            }
        }
        Mout[lig * out_col + col] = conv;
    }
}


__global__ void cudaMeanPool(float* M, float* Mout, int M_line, int M_col, int profondeur){
    
    int out_line = M_col/2;
    int out_col=M_line/2;
    //MeanPool d'une matrice par un kernel 2x2
    int line = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float s = 0.0;
    int tot_meanpool = 2 * 2;

    if (line % 2 == 0 && col % 2 == 0){
        int tot = M_line * M_col;

        for (int meanpool_line = 0; meanpool_line < 2; meanpool_line++) {
            for (int meanpool_col = 0; meanpool_col < 2; meanpool_col++) {
                for (int n_prof = 0; n_prof < profondeur; n_prof++){
                    s += M[(line + meanpool_line) * M_col + col + meanpool_col + n_prof * tot] / tot_meanpool;
            
                }
            }
        }
        if (line == 0){
            Mout[line * out_col + (col / 2)] = s;
    
        }
        else if (col == 0){
            Mout[(line / 2) * out_col + col] = s;
    
        }
        else{
            Mout[(line / 2) * out_col + (col / 2)] = s;
        }
    }
}

// fonction tanh, elle permet de faire la 3 ème couche de notre réseau
__device__ float* activation_tanh(float* M, int nbr_Thread){
    
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nbr_Thread; i+= blockDim.x * gridDim.x){
        M[i] = tanh(M[i]);
    }
    
    return M;
}

__global__ void cudaTanh(float* M, int nbr_Thread){
    activation_tanh(M, nbr_Thread);
}


//////////////////////////////////////////////////Main Partie 2 ////////////////////////////////////////////////////////////

int main() { 
    
    float *raw_data, *S1_data,*C1_kernel, *C1_data;
    float *d_raw_data, *d_S1_data, *d_C1_kernel, *d_C1_data;
    
    int n_raw = 32,kernel_size=5 ,C1_data_size = n_raw-(kernel_size-1) ,S1_data_size=C1_data_size/2,val=1, profondeur = 6;
    
    //allocation de mémoire accessible par le CPU 
    raw_data  = (float*)malloc(sizeof(float) * n_raw*n_raw);
    C1_data  = (float*)malloc(sizeof(float) * C1_data_size*C1_data_size*profondeur);
    C1_kernel  = (float*)malloc(sizeof(float) * kernel_size*kernel_size*profondeur);
    S1_data  = (float*)malloc(sizeof(float) * S1_data_size*S1_data_size*profondeur);

    
    MatrixUnit(raw_data,n_raw,n_raw,1);
    MatrixUnit(C1_data,C1_data_size*profondeur,C1_data_size,0);
    MatrixUnit(S1_data,S1_data_size*profondeur,S1_data_size,0);
    Id_Kernel(C1_kernel, kernel_size, profondeur, val);
    
    printf("Matrice sur laquelle on applique la convolution : \n\n");
    MatrixPrint(raw_data,n_raw,n_raw);

    printf("Matrice de kernel avec laquelle on applique la convolution : \n\n");
    MatrixPrint(C1_kernel, profondeur*kernel_size,kernel_size);
    
    printf("Matrice de sortie de la convolution : \n\n");
    MatrixPrint(C1_data,C1_data_size,C1_data_size);
    
    //allocation de mémoire accessible par le GPU 
    cudaMalloc((void**)&d_raw_data, sizeof(float)*n_raw*n_raw);
    cudaMalloc((void**)&d_C1_data, sizeof(float)*C1_data_size*C1_data_size*profondeur);
    cudaMalloc((void**)&d_C1_kernel, sizeof(float)*kernel_size*kernel_size*profondeur);
    cudaMalloc((void**)&d_S1_data, sizeof(float)*S1_data_size*S1_data_size*profondeur);

    
    // On transfert la mémoire du CPU vers le GPU 
    cudaMemcpy(d_raw_data, raw_data, sizeof(float)*n_raw*n_raw, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C1_data, C1_data, sizeof(float)*C1_data_size*profondeur*C1_data_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C1_kernel, C1_kernel, sizeof(float) *kernel_size*kernel_size*profondeur, cudaMemcpyHostToDevice);
    cudaMemcpy(d_S1_data, S1_data, sizeof(float)*S1_data_size*S1_data_size*profondeur, cudaMemcpyHostToDevice);


    
    
    dim3 block_count(1,1);
    dim3 thread_count(n_raw,n_raw);
    cudaDeviceSynchronize();

        
    cudaConv2D<<<block_count,thread_count>>>(d_raw_data,d_C1_kernel, d_C1_data, n_raw,n_raw, kernel_size,profondeur);
           cudaDeviceSynchronize();

    cudaMemcpy(C1_data, d_C1_data, sizeof(float) * C1_data_size*C1_data_size*profondeur, cudaMemcpyDeviceToHost);
    printf("Matrice issue de la convolution : \n\n");
    MatrixPrint(C1_data,C1_data_size,C1_data_size);

    MatrixInit(raw_data,n_raw,n_raw);
    printf("Matrice issue d'entrée random : \n\n");
    MatrixPrint(raw_data, n_raw, n_raw);
    cudaMemcpy(d_raw_data, raw_data, sizeof(float)*n_raw*n_raw, cudaMemcpyHostToDevice);
    
    cudaConv2D<<<block_count,thread_count>>>(d_raw_data,d_C1_kernel, d_C1_data, n_raw,n_raw, kernel_size,profondeur);
           cudaDeviceSynchronize();

    cudaMemcpy(C1_data, d_C1_data, sizeof(float) * C1_data_size*C1_data_size*profondeur, cudaMemcpyDeviceToHost);
    printf("Matrice issue de la convolution : \n\n");
    MatrixPrint(C1_data,C1_data_size,C1_data_size);
 
    cudaMeanPool<<<block_count,thread_count>>>(d_raw_data,d_S1_data,  n_raw,n_raw,profondeur);
               cudaDeviceSynchronize();

    cudaMemcpy(S1_data, d_S1_data, sizeof(float) * S1_data_size*profondeur*S1_data_size, cudaMemcpyDeviceToHost);
    printf("Matrice issue de la convolution : \n\n");
    MatrixPrint(S1_data,S1_data_size,S1_data_size);
    
    cudaTanh<<<block_count,thread_count>>>(d_S1_data, 14*14);
               cudaDeviceSynchronize();

    cudaMemcpy(S1_data, d_S1_data, sizeof(float) * S1_data_size*profondeur*S1_data_size, cudaMemcpyDeviceToHost);
    printf("Matrice de sortie après activation : \n\n");
    MatrixPrint(S1_data,S1_data_size,S1_data_size);
   
}