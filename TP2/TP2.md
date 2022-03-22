# TP 2 : Résolution des labyrinthes

Dans ce problème, on s'intéresse à la résolution de toutes les instances de labyrinthe de taille fixé (ici 7x7).
Pour cela, nous introduisons l'apprentissage par renforcement profond.

## Partie 1 : Théorie

1. Explication de l'intérêt de la représentation de Q par un réseau de neurones 
- nombre d'états trop grands (potentiellement infini) 
- nécessitée de généraliser à des instances inconnues

2. Lire et compléter le fichier DQN (agent/dqn_agent.py)
   - *Rq: ce fichier utilise la bibliothèque logiciel [Pytorch](https://pytorch.org)*

## Partie 2 : Multi Layer Perceptron (pour DQN)

1. Lancer le programme principal : `python3 main.py mlp`
2. Questions :
   1. Obtient-on les résultats optimaux ?
   2. Selon vous, quels peuvent être les raisons de cette sous-optimalité ?
3. Modifier des hyperparamètres d'apprentissage pour obtenir la quasi-optimalité *(i.e. success_ratio > 0.95)*
4. Changer la taille du labyrinthe à 7x7 et relancer votre méthode. Que se passe-t-il ? 

## Partie 3 : Convolutional Neural Network

Dans cette partie, on propose d'améliorer l'architecture de réseau de neurones qui n'est pas la plus adaptée au problème.
Pour cela on introduit les réseaux de neurones convolutifs (CNN) qui sont adaptés aux traitement de données ayant une représentation "en grille". 

1. Lancer l'apprentissage DQN avec une architecture de réseau de neurones CNN : `python3 main.py cnn`
2. Relancer les expérimentations sur 5x5
   1. Obtient-on les résultats optimaux ?
   2. Quelle différence par rapport à la partie précédente ?
3. Changer la taille du labyrinthe en 7x7 et relancer l'algorithme.
4. Que se passe-t-il ?
5. Mettez en place un processus d'amélioration des hyperparamètres
-> *Note: On pourra tenter de combiner une ou plusieurs modifications suivantes*
   1. Augmenter le nombre d'épisodes d'apprentissage
   2. Explorer plus longtemps
   3. Augmenter la taille du buffer
   4. Augmenter la taille de batch
   5. Changer la largeur du 1er filtre de convolution (passer de 3 à 4)
   6. Changer le nombre de sorties de l'avant dernière couche
   7. Ajouter une couche de convolution 
