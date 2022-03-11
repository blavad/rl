# TP 2 : Résolution des labyrinthes

Dans ce problème, on s'intéresse à la résolution de toutes les instances de labyrinthe de taille fixé (ici 7x7).
Pour cela, nous introduisons l'apprentissage par renforcement profond.

## Partie 1 : Théorie

1. Explication de l'intérêt de la représentation de Q par un réseau de neurones 
- nombre d'états trop grands (potentiellement infini) 
- nécessitée de généraliser à des instances inconnues

2. Lire et compléter le fichier DQN (agent/dqn_agent.py)
   - *Rq: ce fichier utilise la bibliothèque logiciel [Pytorch](https://pytorch.org)*

## Partie 3 : Multi Layer Perceptron (pour DQN)

1. Lancer l'apprentissage DQN avec une architecture de réseau de neurones MLP (voir `main.py` et `networks.py`)
2. Questions :
   1. Obtient-on les résultats optimaux ?
   2. Selon vous, quels peuvent être les raisons de cette sous-optimalité ?
3. Modifier des hyperparamètres d'apprentissage pour obtenir la quasi-optimalité *(i.e. success_ratio > 0.95)*
4. Changer la taille du labyrinthe à 9x9 et relancer votre méthode. Que se passe-t-il ? 

## Partie 4 : Convolutional Neural Network

Dans cette partie, on propose d'améliorer l'architecture de réseau de neurones qui n'est pas la plus adaptée au problème.
Pour cela on introduit les réseaux de neurones convolutifs (CNN) qui sont adaptés aux traitement de données ayant une représentation en grille. 

1. Changer dans le `main`, l'architecture du réseau de neurones pour utiliser un CNN. 
2. Relancer les expérimentations sur 5x5 puis 9x9
   1. Obtient-on les résultats optimaux ?
   2. Quelle différence par rapport à la partie précédente ?  