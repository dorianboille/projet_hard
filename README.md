# Projet HSP // Zappellini/Boille


## Objectifs 
Le but de notre projet est de créer un réseau de CNN de type LeNet-5.
Ce réseau est constitué de deux alternances de CNN et d'average pooling, suivies de deux couches dense.

![image](https://user-images.githubusercontent.com/78031851/149531185-d8487c36-5b6d-4a4c-ae32-901396d8ab28.png)

Au cours de notre projet nous nous proposons de :

* Exploiter un logiciel de gestion de versions décentralisé tel que git.
* Appréhender la programmation en CUDA
* Faire une étude comparative des performances entre nos fonction GPU /CPU
* Créer à partir d'un langage bas niveau des couches de neurones de type convolutive ou fully connected
* Exploiter les poids d'un modèle entrainé grace à Keras de facons à paramètrer nos couches convolutives codées en C.


## Partie 1 : Prise en main de Cuda : Multiplication de matrices

### Création d'une matrice sur CPU :

Dans un premier temps nous allons dévelloper des fonctions simples exécutés par le GPU permettant de créer et observer des matrices.

Chaque matrice est un tableau indexé par un pointeur et la fonction matrixinit() permet de créer un tableau de taille NxP remplis de coefficients aléatoires compris entre -1 et 1. 


![image](https://user-images.githubusercontent.com/78031851/149535166-467ffdbf-2247-4876-9782-2773766c179e.png)

### Affichage d'une matrice sur CPU

Par la suite nous avons aussi créer une fonction MatrixPrint permettant d'afficher dans la console une matrice de taille n lignes p colonnes.

![image](https://user-images.githubusercontent.com/78031851/149536836-3b7c6b93-2fe6-45d9-a251-dc5d2001afb6.png)

Cette fonction est primordiale pour juger de la qualité de notre travail car elle permet de visualiser les résultats de nos diverses opération ainsi que de juger de la justesse des résultats obtenus.

### Addition de deux matrices sur CPU : 

Pour additionner deux matrices sur CPU, il suffit simplement d'ajouter les un après les autres les coefficients de même indice des deux matrice d'entrée puis de venir stocker ces résultats dans la matrice Mout. Il n'y a ici aucune parallélisation des calculs, les additions et les affectations sont faites par le CPU les une après les autres. On commence à comprendre que ces calculs pourraient etre beaucoups plus rapides si le calcul d'un coéfficient et une affectation étaient faites par un thread différents. On gagnerai ainsi un temps proportionnel au nombre de coéfficient. 

On obtient cependant un résultat tout à fait correct :

![image](https://user-images.githubusercontent.com/78031851/149578358-44bd2ca3-c089-4022-82d5-4a105a9d3e68.png)


### Addition de deux matrices sur GPU :

On se propose maintenant de réaliser une fonction cudaMatrixAdd() permettant d'éxploiter les capacités de parrallélisation de nos GPU. Pour cela ilm est important de réflechir aux dimensions que nous allons donner à notre Grid ainsi qu'a nos blocs.
Nous avons choisi de raisonner de la facons suivante :

 * Chaque block corréspond à une une ligne des matrices
 * Chaque thread au sein de ces blocks correspondent à une colonnes des matrices. 

![image](https://user-images.githubusercontent.com/78031851/149542884-bd182552-d9a0-470c-9d6a-3e63b76825a3.png)

En définissant la dimension de la faccons suivante :

![image](https://user-images.githubusercontent.com/78031851/149544760-df7a1254-e894-4c22-9d4b-d06a7cbb98db.png)

On en déduit la fonction suivante :

![image](https://user-images.githubusercontent.com/78031851/149544188-ac6c1f92-755e-4f16-bf3a-06289a0db73b.png)

On observe que cette fonction permet à chaque thread de chaque block de calculer un coefficient différent de l'aditin de atrice et de le stocker au bon endroit dans la matrice de sortie Mout.

En cuda les matrices sont indicées grâce au variables :
* gridDim.x/y représentant la dimension totale selon x/y de la grille que nous avons défini.
* blockDim.x/y représentant la dimension totale selon x/y de chaque block que nous avons défini au sein de la grid.
* blockIdx.x/y représentant l'Id du block (selon x ou y) auquel appartient le thread qui est concerné.
* ThreadIdx.xy représentant l'Id  (selon x ou y) du thread qui est concerné

Le fait que l'on utilise le specifier __global__ traduit le fait que la fonction est:
* Executée par le GPU. 
* Appelée par le CPU.

On obtient ce résultat :

![image](https://user-images.githubusercontent.com/78031851/149578565-6ffa28d3-27b6-472b-a33e-fab60c6c0546.png)

On note que la matrice issue de l'addition est la meme que celle calculée sur le CPU, cependant sont temps de calcul est fortement réduit.

![Comparaison](https://user-images.githubusercontent.com/78031851/149579608-fce0da4d-ecdc-4d28-931a-35b0b3e0150b.png)


### Multiplication de deux matrices NxN sur CPU :

Nous voulons maintenant effectuer la multiplication de deux matrices. Le calcul de chaque coefficient de la matrice de sortie est le suivant :

![image](https://user-images.githubusercontent.com/78031851/149547974-44eed4b4-1f49-4eef-b1ae-c85fad73a7e6.png)

On effectue ce calcul sur le CPU avec cette fonction 

![image](https://user-images.githubusercontent.com/78031851/149545248-f998cf28-df15-4350-a2bb-64acabd8def2.png)

On obtient ainsi la matice suivante :

![image](https://user-images.githubusercontent.com/78031851/149579478-3aa10ddb-1673-4fa6-88d9-7ce7ffabab34.png

Les valeurs sont bonnes il y a juste des arrondis.


### Multiplication de deux matrices NxN sur GPU :

On se propose maintenant de réaliser la même fonction mais en CUDA et executable par le GPU.

Chaque ligne est représenté par un Block et chaque colonne est représenté par un ID de thread au sein de ces blocks.

![image](https://user-images.githubusercontent.com/78031851/149549335-86ce292c-3f5f-4187-a156-12add19facb3.png)

On obtient ainsi la matice suivante :

![image](https://user-images.githubusercontent.com/78031851/149579810-3b2641e7-e1e7-4ae5-81fd-8b8b2901a214.png)
 
 On observe que les matrices issues du CPU et du GPU sont identiques. 
 
 ![image](https://user-images.githubusercontent.com/78031851/149580036-af804a1e-df9b-4b9c-8402-07204662bad7.png)


##  Partie 2 - Premières couches du réseau de neurone LeNet-5 : Convolution 2D et subsampling

### Layer 1 - Génération des données de test

Nous avons aussi codé une fonction permettant de créer une matrice de taille NxP remplie de coéfficients tous égaux à une valeur fixé val. Cela nous permettera par la suite de faire des vérification rapide que nos autres fonctions marchent bien, notament quand il s'agira de notre fonction de convolution.

De plus on crée une fonction permettant de faire une matrice de kernel carré de dimension dim identitaire dont la valeur centrale est fixée par la variable val.

![image](https://user-images.githubusercontent.com/78031851/149550547-31532be1-b81f-4c26-844f-ea05b03708a5.png)

Ces kernels sont très pratiques pour vérifier que nos convolutions marchent bien en un coup d'oeil.

On obtient donc les 6 noyaux de taille 5x5 suivants :

![image](https://user-images.githubusercontent.com/78031851/149580516-45d5250b-3023-4468-932e-c891fba61c8f.png)


### Layer 2 - Convolution 2D :

Nous voulons mettre en place une convolution en 2 dimension de notre image 32x32x1 d'entrée issue de MNIST. Nous voulons réaliser cette convolution sur GPU de facons à diminuer au minimum notre temps de calcul.

![image](https://user-images.githubusercontent.com/78031851/149553590-3e7358a1-ac2b-4692-8427-98cbb4a21385.png)

Dans notre cas on souhaite faire la convolution de cette image par 6 kernels de taille 5x5, nous obtiendrons donc une sortie de taille 28x28x6.
Layer 2- Convolution avec 6 noyaux de convolution de taille 5x5. La taille résultantes est donc de 6x28x28.

Voici la fonction de convolution que nous avons utilisé pour la suite :

``` C++
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
```
### Layer 3 - Sous-échantillonnage

Nous nous fixons maintenant l'objectif de faire un mean pooling 2x2 de la sortie de notre couche de convolution.

![image](https://user-images.githubusercontent.com/78031851/149553896-231e3b11-25fb-445d-b52c-5c47484650be.png)

Cette étape permet de faire un subsampling de la feature map tout en introduisant une invariance à la translation. Ce type d'opération est très souvent utilisée dans après une couche de convolution.

### Tests

Afin de vérifier notre fonction de convolution, nous avons choisi de faire la convolution d'une matrice de 32*32*1 remplie de 1 par des noyaux identité dont la valeur centrale est 10.

Voici la matrice de sortie de la convolution :

![image](https://user-images.githubusercontent.com/78031851/149580797-ea9c81ad-85e6-490f-bcdf-8633e86afe11.png)


Les résultats sont bien conformes à ce que nous attendions à avoir, à savoir une matrice 28*28*6 remplie de 10.


### Fonctions d'activation

Afin d'achever cette partie nous allons coder une fonction d'activation afin de l'appliquée en sortie de nos deux couches à chacun des coéficients de la matrice. Le choix s'est porté sur une fonction tanh :

![image](https://user-images.githubusercontent.com/78031851/149557450-af774bb7-962d-47cd-b16e-5582e19684a8.png)

Cette fonction renvoie une valeur entre -1 et 1 et celle-ci satureà 1 en à partir de 2 et à -1 à partir de -2;
C'est pour cette raison que nous ;llons tester cette fonction à l'aide d'une matrice remplie de valeure entre -1 et 1 que nous allons convoluer avec un kenel unitaire suivi d'un average pooling.

On obtient la matrice suivante de taille 14x14 :

![image](https://user-images.githubusercontent.com/78031851/149581235-cdd1ec9f-4286-40e1-bf75-521b79dd9209.png)


Celle-ci correspond bien à ce qu'on s'attend à avoir.

## TP3  Un peu de Python
