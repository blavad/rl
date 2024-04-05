# TP n°4 : Résolution d'un labyrinthe quelconque

Dans ce problème, on s'intéresse à la résolution de toutes les instances de labyrinthe de taille fixé (ici 7x7).
Pour cela, nous introduisons l'apprentissage par renforcement profond.

Les compétences travaillées durant cette activité sont les suivantes :

- Comprendre la notion de réseau de neurones
- Comprendre l'intérêt de l'apprentissage par renforcement profond
- Comprendre l'algorithme **DQN**
- Manipuler des hyperparamètres d'apprentissage
- Comprendre l'impact des hyperparamètres sur l'apprentissage

## Partie 0 : Pour commencer

10min

### Récupération des travaux pratiques

1. Ouvrir le dossier des TPs dans un terminal

2. Activer l'environnement virtuel python créé lors du TP n°2

   ```bash
    source .venv/bin/activate # activation de l'environnement
   ```

3. Installer des dépendances du TP 4.
   ```bash
   # Dans le dossier TP4-DeepRL
   pip3 install -r requirements.txt
   ```

## Partie 1 : Théorie

40min

1. Expliquer pourquoi nous ne pouvons pas résoudre toutes les instances l'algorithme de labyrinthe de taille fixé grâce à l'algorithme Q-learning classique.
2. Dans un cadre général, quel est l'intérêt de représenter Q par un réseau de neurones ?
<!-- - nombre d'états trop grands (potentiellement infini)
   - nécessitée de généraliser à des instances inconnues -->
3. Lire et compléter le fichier DQN (`agent/dqn_agent.py`)

   - _Rq: ce fichier utilise la bibliothèque logiciel [Pytorch](https://pytorch.org)_

4. Qu'est-ce que le buffer d'expérience replay ? A quoi sert-il ?

## Partie 2 : Multi Layer Perceptron (pour DQN)

40min

1. Lancer l'apprentissage DQN avec une architecture de réseau de neurones MLP (voir `main.py` et `networks.py`)
2. Questions :
   1. Obtient-on les résultats optimaux ?
   2. Selon vous, quels peuvent être les raisons de cette sous-optimalité ?
3. Modifier des hyperparamètres d'apprentissage pour obtenir la quasi-optimalité _(i.e. success_ratio > 0.95)_
4. Changer la taille du labyrinthe à 7x7 et relancer votre méthode. Que se passe-t-il ?

## Partie 3 : Convolutional Neural Network

40min

Dans cette partie, on propose d'améliorer l'architecture de réseau de neurones qui n'est pas la plus adaptée au problème.
Pour cela on introduit les réseaux de neurones convolutifs (CNN) qui sont adaptés aux traitement de données ayant une représentation en grille.

1. Lancer l'apprentissage DQN avec une architecture de réseau de neurones CNN (voir `main.py` et `networks.py`)
2. Relancer les expérimentations sur 5x5
   1. Obtient-on les résultats optimaux ?
   2. Quelle différence par rapport à la partie précédente ?
3. Changer la taille du labyrinthe en 7x7 et relancer l'algorithme.
4. Que se passe-t-il ?
5. Mettez en place un processus d'amélioration des hyperparamètres

   > **Aide:** On pourra tenter de combiner une ou plusieurs modifications suivantes
   >
   > - Augmenter le nombre d'épisodes d'apprentissage
   > - Explorer plus longtemps
   > - Augmenter la taille du buffer
   > - Augmenter la taille de batch
   > - Changer la largeur du 1er filtre de convolution (passer de 3 à 4)
   > - Changer le nombre de sorties de l'avant dernière couche
   > - Ajouter une couche de convolution

## Partie 4 : Optimisation

Les algorithmes d'apprentissage par renforcement profond sont pertinents pour la résolution de problèmes de prise de décision dont l'espace d'état est infini. Toutefois, pour des problèmes complexes, il faudra en général attendre plusieurs heures, plusieurs jours ou même plusieurs semaines pour converger vers un solution optimale. Cela est dû au fait que la mise à jour du réseau de neurones est progessive et nécessite un nombre d'épisodes conséquent.

**L'efficacité de la mise à jour du réseau de neurones et de la mise à jour de l'environnement (fonction `step`) est donc primordiale pour éviter toute perte de temps et de ressources inutiles**.

### Problème ouvert

L'algorithme DQN présenté ici a été implémenté dans un cadre pédagogique sans focus particulier sur les contraintes de temps.

1. A vue d'oeil, voyez-vous des parties de code qui vous semble mal optimisées ?
2. Utilisez un profiler ou recréer votre propre profiler pour analyser de temps passé dans chacune des fonctions de l'algorithme
3. Maintenant, quelles parties prennent le plus de temps ?
4. Peuvent-elles être optimisées ?

Si des propositions sont pertinentes, elles pourront être mises en place dans le TP pour l'année prochaine en mentionnant votre nom.
