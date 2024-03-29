# TP n°1 : Résolution d'un labyrinthe fixé

Dans ce problème, on s'intéresse à la résolution d'un labyrinthe fixé de taille quelconque. Les compétences travaillées durant cet activité sont les suivantes.

- Écrire en Python l'algorithme **Value Iteration** de résolution d'un processus décisionnel de Markov, avec modèle connu.

## Partie 1 : Théorie

30min

Soit le labyrinthe ci-dessous :

  <img src="https://raw.githubusercontent.com/blavad/IAT/master/TP2-ModelBased/tests/MazeEx2NR.png" width="200" title="hover text">

1. Sur papier, calculer la fonction de valeur d'états optimale :

   - `Discount Factor = 1.0`
   - `R(s, a) = -1` pour tout s, pour tout a
   - **Transition déterministes**

   > **_Remarque :_** l'objectif est atteint quand l'agent atteint le point rouge et dans ce cas, l'agent perçoit une récompense nulle.

2. Sur papier, calculer la fonction de valeur d'états optimale :

   - `Discount Factor = 1.0`
   - `R(s, a) = -1` pour tout s, pour tout a
   - **Transition stochastiques** et on distinguera deux cas :
     1. Si l'action est exécutable, c'est-à-dire qu'elle ne mène pas à un mur, alors la probabilité de succès est de 80%, la probabilité qu'elle échoue est de 20%. Lorsqu'une action échoue, l'agent reste dans la cellule courante.
     2. Si l'action est non exécutable, alors l'action échoue systématiquement et l'agent demeure dans la cellule courante.

## Partie 2 : Value Iteration

30min

### Intro

Quand on connait les modèles de la dynamique, `p(s' | s,a)` et `r(s,a)`, on peut utiliser un algorithme de planification pour déterminer la politique optimale.

### Value Iteration

Il s'agit d'une méthode de résolution des processus décisionnels de Markov avec connaissance parfaite du modèle de l'environnement. L'algorithme procède de façon itérative, mettant à jour la fonction de valeur jusqu'à ce que l'écart entre deux mises à jour soit inférieur à un seuil fixé, e.g., 0.01.

> Chaque mise à jour consiste à l'application des équations d'optimalité de Bellman.

- Lire et compléter le fichier Value Iteration (`TP2-ModelBased/agent/viagent.py`)

  > L'état `s` correspond au couple de coordonnées `(y,x)`

- Lancer le programme principal avec comme paramètre `vi` (pour **v**alue **i**teration)
  - Lancer `python3 main.py vi`
- Commenter la ligne `env = Maze(7, 7, min_shortest_length=15)`
  - Utiliser l'environnement `env = Maze.from_file("tests/maze_ex2.txt")` qui correspond au labyrinthe de la partie 1
- Comparer les résultats obtenus aux résultats théoriques en affichant la fonction de valeur obtenue après résolution

## Partie 2.5 : Visualisation

- Visualiser l'évolution de la fonction de valeur avec la commande `python3 logAnalysisV.py` dans le dossier `partie_2/visualisation/`
<!--

## Partie 3 : Implémenter l'algorithme "Q-Learning" (40min)

**Intro** : Parfois, les modèles de la dynamique sont inconnus ou gigantesques. S'il est possible d'interagir avec le système directement et récupérer des informations au fil de l'eau, il est alors possible d'implémenter des algorithmes d'apprentissage par renforcement pour déterminer la politique optimale.

1. Lire et compléter le fichier Q-learning (agent/qagent.py)
2. Lancer le programme principal avec comme paramètre `qlearning`

- Lancer`python3 main.py qlearning`

4. Augmenter la taille du labyrinthe à (14, 14) et recommencer l'apprentissage

- Que remarque-t-on ?
- Quelle(s) solution peut-on apporter ?
  - _Indice:_ modifier les paramètres d'apprentissage `n_episodes` et `max_steps` par exemple

5. Modifier le paramètre `eps_profile` pour ne faire que de l'exploration ?

- Analyser les résultats
- Quel est l'intérêt de faire décroître ce paramètre ?

## Partie 3.5 : Visualisation

- Visualiser la courbe d'évolution de la Q-valeur avec la commande `python3 main.py logAnalysisQ`
- Visualiser l'évolution de la fonction de valeur avec la commande `python3 logAnalysisV.py` dans le dossier `partie_3/visualisation/` -->
