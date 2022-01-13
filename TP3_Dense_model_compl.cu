#include <stdio.h>
#include <stdlib.h>
#include <cstdlib>


//fonctions utilitaires, construction de matrice / Affichage //

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

void MatrixUnit(float *M, int n, int p, float value){
    srand(time(NULL));
    for (int  i = 0; i <n*p; i++){
        M[i] = value;
    };
}

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

///////////////////////////////////////// Couches du modèle ///////////////////////////////////// 

// Conv 2D//

// Cette fonction fait une convolution 2D (Cf part2)//

__global__ void cudaConv2D(float* M, float* kernel, float* Mout, int M_line, int M_col, int kernel_size, int nb_kernel){
    
    int offset = (kernel_size-1)/2;
    int out_line = M_line - offset;
    int out_col = M_col - offset;
    //Convolution d'une matrice par un kernel
    int lig = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;



    if (lig < out_line && col < out_col){
        
        int temp = out_line * out_col;
         for (int n_k = 0; n_k < nb_kernel; n_k++){
            float conv = 0.0;
 
            for (int k_col = 0; k_col < kernel_size; k_col++) {
               for (int k_line = 0; k_line < kernel_size; k_line++) {
                    conv += M[(lig + k_line) * M_col + col + k_col] * kernel[k_line * kernel_size + k_col + n_k * kernel_size*kernel_size];
            
                }
            }
        Mout[lig * out_col + col + n_k*temp] = conv;
             
        }
        
    }
}


// Activation tanh (Cf part2)//


__device__ float* activation_tanh(float* M, int nbr_Thread){
    
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nbr_Thread; i+= blockDim.x * gridDim.x){
        M[i] = tanh(M[i]);
    }
    
    return M;
}

__global__ void cudaTanh(float* M, int nbr_Thread){
    activation_tanh(M, nbr_Thread);
}

// Mean pool //

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


// Fonction permettant de faire une couche de FC //
/*
    - en argument on passe les dimension de la couche précédente 
        - lig_prec
        - col_rec
        - prof_prec
    - on passe aussi la matrice de poid w.
    - on passe le vecteur de donnée linéarisé x ainsi sur la matrice de sortie Mout.
    
*/

__global__ void cudaFullyConnected(float *x_in, float *w, float *Mout, int lig_prec, int col_prec,int prof_prec){

    int lig  = threadIdx.x;
    float temp = 0;
    
    for (int i=0; i<lig_prec*col_prec*prof_prec;i++){
        temp+=x_in[i]*w[i*lig_prec*col_prec*prof_prec +lig];
    }
    
    Mout[lig] = temp;

}

int main() { 
    
    // création des pointeurs GPU et CPU de chaque matrice
    float *raw_data;
    float *S1_data,*C1_kernel, *C1_data;
    float *d_raw_data, *d_S1_data, *d_C1_kernel, *d_C1_data;
    float *S2_data,*C2_kernel, *C2_data;
    float *d_S2_data, *d_C2_kernel, *d_C2_data;
    
    int val=1,kernel_size=5;
    int n_raw = 32; 
   
    // Paramètre de la première conv et du premier average pooling 
    int C1_data_size = n_raw-(kernel_size-1) ,S1_data_size=C1_data_size/2, profondeur1 = 6;
    
    // Paramètre de la première conv et du premier average pooling 
    int C2_data_size = S1_data_size-(kernel_size-1) ,S2_data_size=C2_data_size/2, profondeur2 = 16;

    //allocation de mémoire accessible par le CPU 
    raw_data  = (float*)malloc(sizeof(float) * n_raw*n_raw);
    
    C1_data  = (float*)malloc(sizeof(float) * C1_data_size*C1_data_size*profondeur1);
    C1_kernel  = (float*)malloc(sizeof(float) * kernel_size*kernel_size*profondeur1);
    S1_data  = (float*)malloc(sizeof(float) * S1_data_size*S1_data_size*profondeur1);
    
    C2_data  = (float*)malloc(sizeof(float) * C2_data_size*C2_data_size*profondeur2);
    C2_kernel  = (float*)malloc(sizeof(float) * kernel_size*kernel_size*profondeur2);
    S2_data  = (float*)malloc(sizeof(float) * S1_data_size*S1_data_size*profondeur2);
    

    // Initialisation de la donnée d'entrée 
    MatrixUnit(raw_data,n_raw,n_raw,1);
    // Initialisation des matrices de la 1ère couche de convolution et average pooling 
    MatrixUnit(C1_data,C1_data_size*profondeur1,C1_data_size,0);
    MatrixUnit(S1_data,S1_data_size*profondeur1,S1_data_size,0);
    Id_Kernel(C1_kernel, kernel_size, profondeur1, val);
    
    // Initialisation des matrices de la 2ème couche de convolution et average pooling 
    MatrixUnit(C2_data,C2_data_size*profondeur2,C2_data_size,0);
    MatrixUnit(S2_data,S2_data_size*profondeur2,S2_data_size,0);
    Id_Kernel(C2_kernel, kernel_size, profondeur2, val);
    
    printf("Matrice sur laquelle on applique la convolution : \n\n");
    MatrixPrint(raw_data,n_raw,n_raw);

    printf("Matrice de kernel avec laquelle on applique la convolution : \n\n");
    MatrixPrint(C1_kernel, profondeur1*kernel_size,kernel_size);
    
    printf("Matrice de sortie de la convolution : \n\n");
    MatrixPrint(C1_data,C1_data_size,C1_data_size);
    
    
    cudaMalloc((void**)&d_raw_data, sizeof(float)*n_raw*n_raw);    
    
    //allocation de mémoire accessible par le GPU pour la 1ère convolution et average sampling 

    cudaMalloc((void**)&d_C1_data, sizeof(float)*C1_data_size*C1_data_size*profondeur1);
    cudaMalloc((void**)&d_C1_kernel, sizeof(float)*kernel_size*kernel_size*profondeur1);
    cudaMalloc((void**)&d_S1_data, sizeof(float)*S1_data_size*S1_data_size*profondeur1);
    
    //allocation de mémoire accessible par le GPU pour la 2ème convolution et average sampling 

    cudaMalloc((void**)&d_C2_data, sizeof(float)*C2_data_size*C2_data_size*profondeur2);
    cudaMalloc((void**)&d_C2_kernel, sizeof(float)*kernel_size*kernel_size*profondeur2);
    cudaMalloc((void**)&d_S2_data, sizeof(float)*S2_data_size*S2_data_size*profondeur2);

    
    // On transfert la mémoire du CPU vers le GPU 
    cudaMemcpy(d_raw_data, raw_data, sizeof(float)*n_raw*n_raw, cudaMemcpyHostToDevice);
    
    cudaMemcpy(d_C1_data, C1_data, sizeof(float)*C1_data_size*profondeur1*C1_data_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C1_kernel, C1_kernel, sizeof(float) *kernel_size*kernel_size*profondeur1, cudaMemcpyHostToDevice);
    cudaMemcpy(d_S1_data, S1_data, sizeof(float)*S1_data_size*S1_data_size*profondeur1, cudaMemcpyHostToDevice);

    cudaMemcpy(d_C2_data, C2_data, sizeof(float)*C2_data_size*profondeur2*C2_data_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C2_kernel, C2_kernel, sizeof(float) *kernel_size*kernel_size*profondeur2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_S2_data, S2_data, sizeof(float)*S2_data_size*S2_data_size*profondeur2, cudaMemcpyHostToDevice);

    
    
    dim3 block_count(1,1);
    dim3 thread_count(n_raw,n_raw);
    cudaDeviceSynchronize();

        
    cudaConv2D<<<block_count,thread_count>>>(d_raw_data,d_C1_kernel, d_C1_data, n_raw,n_raw, kernel_size,profondeur1);
           cudaDeviceSynchronize();

    cudaMemcpy(C1_data, d_C1_data, sizeof(float) * C1_data_size*C1_data_size*profondeur1, cudaMemcpyDeviceToHost);
    printf("Matrice issue de la convolution : \n\n");
    MatrixPrint(C1_data,C1_data_size,C1_data_size);

    MatrixInit(raw_data,n_raw,n_raw);
    printf("Matrice issue d'entrée random : \n\n");
    MatrixPrint(raw_data, n_raw, n_raw);
    cudaMemcpy(d_raw_data, raw_data, sizeof(float)*n_raw*n_raw, cudaMemcpyHostToDevice);
    
    cudaConv2D<<<block_count,thread_count>>>(d_raw_data,d_C1_kernel, d_C1_data, n_raw,n_raw, kernel_size,profondeur1);
           cudaDeviceSynchronize();

    cudaMemcpy(C1_data, d_C1_data, sizeof(float) * C1_data_size*C1_data_size*profondeur1, cudaMemcpyDeviceToHost);
    printf("Matrice issue de la convolution : \n\n");
    MatrixPrint(C1_data,C1_data_size,C1_data_size);
    
    cudaTanh<<<block_count,thread_count>>>(d_C1_data, C1_data_size*C1_data_size);
               cudaDeviceSynchronize();

    cudaMemcpy(C1_data, d_C1_data, sizeof(float) * C1_data_size*profondeur1*C1_data_size, cudaMemcpyDeviceToHost);
    printf("Matrice de sortie après activation : \n\n");
    MatrixPrint(C1_data,C1_data_size,C1_data_size);
        
    
    dim3 thread_count2(C1_data_size,C1_data_size);
    cudaMeanPool<<<block_count,thread_count2>>>(d_C1_data,d_S1_data, C1_data_size,C1_data_size,profondeur1);
               cudaDeviceSynchronize();

    cudaMemcpy(S1_data, d_S1_data, sizeof(float) * S1_data_size*profondeur1*S1_data_size, cudaMemcpyDeviceToHost);
    printf("Matrice issue du average pool : \n\n");
    MatrixPrint(S1_data,S1_data_size,S1_data_size);
                   cudaDeviceSynchronize();

    
    
    
    dim3 thread_count3(C2_data_size,C2_data_size);
    cudaConv2D<<<block_count,thread_count3>>>(d_S1_data,d_C2_kernel, d_C2_data, n_raw,n_raw, kernel_size,profondeur2);
           cudaDeviceSynchronize();

    cudaMemcpy(C2_data, d_C2_data, sizeof(float) * C2_data_size*C2_data_size*profondeur2, cudaMemcpyDeviceToHost);
    printf("Matrice issue de la convolution : \n\n");
    MatrixPrint(C2_data,C2_data_size,C2_data_size);

    
    cudaTanh<<<block_count,thread_count3>>>(d_C2_data, C2_data_size*C2_data_size);
               cudaDeviceSynchronize();

    cudaMemcpy(C2_data, d_C2_data, sizeof(float) * C2_data_size*profondeur2*C2_data_size, cudaMemcpyDeviceToHost);
    printf("Matrice de sortie après activation : \n\n");
    MatrixPrint(C2_data,C2_data_size,C2_data_size);
        
    
    dim3 thread_count4(C2_data_size,C2_data_size);
    cudaMeanPool<<<block_count,thread_count4>>>(d_C2_data,d_S2_data, C2_data_size,C2_data_size,profondeur2);
               cudaDeviceSynchronize();

    cudaMemcpy(S2_data, d_S2_data, sizeof(float) * S2_data_size*profondeur2*S2_data_size, cudaMemcpyDeviceToHost);
    printf("Matrice issue du average pool : \n\n");
    MatrixPrint(S2_data,S2_data_size,S2_data_size);
                   cudaDeviceSynchronize();


}