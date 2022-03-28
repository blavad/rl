# TP 2 : Résolution des labyrinthes

Dans ce problème, on s'intéresse à la résolution de toutes les instances de labyrinthe de taille fixée (exemples 5x5 et 7x7).
Pour cela, nous utiliserons l'apprentissage par renforcement profond. 

## Partie 1 : Théorie

L'intérêt de la représentation des fonctions d'action-valeur (ou Q-valeur)  par une architecture de réseaux de neurones est de faire face aux éléments suivants:
- nombre d'états trop grands (potentiellement infini) 
- généraliser le comportement à des instances inconnues

1. Lire le fichier DQN (agent/dqn_agent.py)
   - *Note: ce fichier utilise la bibliothèque logiciel [Pytorch](https://pytorch.org)*
2. S'agit-il d'un algorithme basé valeur ou politique?
3. Est-ce un algorithme tabulaire? 
 


## Partie 2 : Multi Layer Perceptron (pour DQN)

Dans cette partie, nous utiserons une architecture de réseaux de neurones simpliciste mais pas la plus adéquate pour notre problème, à savoir l'architecture [MLP](https://en.wikipedia.org/wiki/Multilayer_perceptron). 

1. Lancer le programme principal : `python3 main.py mlp`, interpréter les traces d'exécution. 
2. Modifier l'hyper-paramètre `n_episodes = 5000` et relancer le programme. Que constatez-vous? 
3. Modifier l'hyper-paramètre `n_episodes = 50000` et relancer le programme. Que pouvez-vous conclure sur le choix adequat de l'hyper-paramètre `n_episodes`?
4. Modifier l'architecture du réseau de neurones (cf. `networks.py`) en changeant le nombre de neurones à 16 de la couche interne. Que constatez-vous?
5. Modifier l'architecture du réseau de neurones (cf. `networks.py`) en changeant le nombre de neurones à 128 de la couche interne. Que pouvez-vous conclure sur le choix adequat de l'hyper-paramètre `nombre de neurones` de la couche interne? 

## Partie 3 : Convolutional Neural Network

Dans cette partie, on propose de remplacer l'architecture de réseau de neurones précédente par une qui soit plus adaptée au problème.
Pour cela nous utilisons les réseaux de neurones convolutifs ([CNN](https://fr.wikipedia.org/wiki/Réseau_neuronal_convolutif)) qui sont adaptés aux traitement de données ayant une représentation "en grille". 

1. Lancer DQN avec l'architecture de réseau de neurones CNN  `python3 main.py cnn`  avec `n_episodes = 2000`. Que constatez-vous?
2.  Changer la taille du labyrinthe en 7x7 et relancer l'algorithme. Que pouvez-vous conclure sur l'architecture CNN?
3.  Mettez en place un processus d'amélioration des hyperparamètres (cf. `main.py`). 
   - *Note: On pourra tenter de combiner une ou plusieurs modifications suivantes*
      - Augmenter le nombre d'épisodes d'apprentissage, cf. `main.py` ligne 45
      - Explorer plus longtemps, cf. `main.py` lignes 48
      - Augmenter la taille du buffer, cf. `main.py` ligne 53
      - Augmenter la taille de batch, cf. `main.py` ligne 52
      - Changer la largeur du 1er filtre de convolution (passer de 3 à 4), cf. `networks.py` ligne 46
      - Changer le nombre de neurones de l'avant dernière couche, cf. `networks.py` ligne 51-52
      - Ajouter une couche de convolution, cf. `networks.py` lignes 46-49
