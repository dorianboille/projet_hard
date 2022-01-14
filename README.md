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

/////////////////
////////////////

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

### Multiplication de deux matrices NxN sur CPU :


![image](https://user-images.githubusercontent.com/78031851/149545248-f998cf28-df15-4350-a2bb-64acabd8def2.png)

### Multiplication de deux matrices NxN sur GPU



## TP2 Conv, Mean pooling et Acivation 

Nous avons aussi codé une fonction permettant de créer une matrice de taille NxP remplie de coéfficients tous égaux à une valeure fixé val. Cela nous permettera par la suite de faire des vérification rapide que nos autres fonctions marchent bien, notament quand il s'agira de notre fonction de convolution.

## TP3  Réseau complet Dense Layer
