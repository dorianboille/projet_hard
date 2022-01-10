#include <stdio.h>
#include <stdlib.h>
#include <cstdlib> 



// Fonction permettant la création d'une matrice M à n lignes et p colonnes //
/* 
   - Cette matrice sera remplie de float compris entre -1 et 1
   - Cette matrice est créée sur le CPU
*/

void MatrixInit(float *M, int n, int p){
    srand(time(NULL));
    for (int  i = 0; i <n*p; i++){
        M[i] = (float)rand()/RAND_MAX*2-1;
    };
}

// Fonction permettant d'afficher une matrice sur le CPU
/* 
   - prend en entrée l'adresse de la matrice (sous forme de colonne)
   - prend en argument la taille de la matrice: n lignes p colonnes
*/

void MatrixPrint(float *M, int n, int p){
    
    for (int  i = 0; i <n*p; i++){
        
        // on print ligne par ligne 
        if (i%p==0){
            printf("\n");
        }
        printf("  %1.1f  ", M[i]);
    };
    printf("\n \n");
}

// Fonction réalisant l'addition de deux matrices sur le GPU//

/*
    - Prend en argument l'adresse des deux matrices à additioner ainsi que celle de la matrice de sortie.
    - Prend en argument la taille de la matrice: n lignes p colonnes.
    - Les deux matrices doivent bien entendu avoir les mêmes dimension 
*/

void MatrixAdd(float *M1, float *M2, float *Mout, int n, int p){
    
    for (int  i = 0; i <n*p; i++){
        Mout[i] = M1[i] + M2[i];
    };

}

// Fonction permettant de faire une multiplication de matrice de taille nxn //

/*
    - Prend en argument l'adresse des deux matrices à multiplier ainsi que celle de la matrice de sortie.
    - Prend en argument la taille de la matrice: n lignes n colonnes.
    - Les deux matrices doivent avoir les mêmes dimension (n x n)
*/

void MatrixMult(float *M1, float *M2, float *Mout, int n){
    
    
    for (int lig = 0; lig < n; lig++){
        for (int col = 0; col < n; col++){
            float s = 0.0f;
            for (int i = 0; i < n; i++) {
                s += M1[lig * n + i] * M2[i * n + col];
            }
            Mout[lig * n + col] = s;
        };
    }
}

///// Fonctions équivalente écrite pour le GPU /////

__global__ void cudaMatrixAdd(float *M1, float *M2, float *Mout, int n, int p){

        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid < n*p){
            Mout[tid] = M1[tid] + M2[tid];
        }

}


__global__ void cudaMatrixMult(float *M1, float *M2, float *Mout, int n){
    
    int lig = blockIdx.x;
    int col = threadIdx.x;
    
    float s = 0.0f;
    
    if(lig < n && col < n){
        for (int i = 0; i < n; i++){
            s += M1[lig * n + i] * M2[i * n + col];
        }
    }
    Mout[lig * n + col] = s;
}

////////////////////////////////////////// Main ///////////////////////////////////////////////

int main(){
    
    // Déclaration des pointeurs des matrices utilisés par le CPU et par le GPU
    float *M1, *M2, *Mout;
    float *d_M1, *d_M2, *d_Mout;

    // On se propose de travailler avec des matrices 5x5
    int n = 5,p=5;
    
    //allocation de mémoire accessible par le CPU 
    
    M1  = (float*)malloc(sizeof(float) * n*p);
    M2  = (float*)malloc(sizeof(float) * n*p);
    Mout  = (float*)malloc(sizeof(float) * n*p);



    //allocation de mémoire accessible par le GPU 
    cudaMalloc((void**)&d_M1, sizeof(float)*n*p);
    cudaMalloc((void**)&d_M2, sizeof(float)*n*p);
    cudaMalloc((void**)&d_Mout, sizeof(float)*n*p);

    // Ici on initialise les matrices M1 et M2 avant de lancer nos opérations 
    MatrixInit(M1,n,p);
    MatrixInit(M2,n,p);
    
    // On copie les matrices en mémoire du CPU vers le GPU 
    cudaMemcpy(d_M1, M1, sizeof(float) * n*p, cudaMemcpyHostToDevice);
    cudaMemcpy(d_M2, M2, sizeof(float) * n*p, cudaMemcpyHostToDevice);
    
    printf("Matrice M1 de taille (%d,%d) :\n",n,p);
    MatrixPrint(M1,n,p);
        
    printf("Matrice M2 de taille (%d,%d) :\n",n,p);
    MatrixPrint(M2,n,p);

    printf("///////// Comparaison des deux additions ////////// \n \n");   
    
    printf("Addition du CPU \n");   
    MatrixAdd(M1,M2,Mout,n,p);
    MatrixPrint(Mout,n,p);


    // on a un bloc par ligne et un thread par colonne
    int block_size = n;
    int grid_size = p;

    cudaMatrixAdd<<<grid_size,block_size>>>(d_M1, d_M2, d_Mout, n,p);
    // On copie ici la Matrice de sortie du GPU vers celle du CPU
    cudaMemcpy(Mout, d_Mout, sizeof(float) * n*p, cudaMemcpyDeviceToHost);
    printf("Addition de Matrices sur le GPU : \n");
    MatrixPrint(Mout,n,p);
    
    printf("///////// Comparaison des deux multiplications  ////////// \n \n");   

        
    
    MatrixMult(M1,M2,Mout, n);
    printf("Multiplication du CPU /n");
    MatrixPrint(Mout,n,p);



    //printf("Multiplication du GPU");
    cudaMatrixMult<<<grid_size,block_size>>>(d_M1, d_M2, d_Mout, n);
    cudaMemcpy(Mout, d_Mout, sizeof(float) * n*p, cudaMemcpyDeviceToHost);
    printf("Multiplication de Matrices sur le GPU : \n");
    MatrixPrint(Mout,n,p);


 

}
