# TP 1 : Résolution d'un labyrinthe fixé

Dans ce problème, on s'intéresse à la résolution d'un labyrinthe fixé de taille quelconque. Les compétences travaillées durant cet activité sont les suivantes.

- Retrouver, differentier formellement les equations d'optimalité de Bellman et les équations de Bellman.
- Écrire en Python l'algorithme **Value Iteration** de résolution  d'un processus décisionnel de Markov, avec modèles connus. 
- Écrire en Python l'algorithme **Q-Learning** de résolution  d'un processus décisionnel de Markov, avec modèles inconnus. 
- Savoir conduire des simulations d'un algorithme d'apprentissage par renforcement via les hyper-paramètres d'apprentissage.
- Savoir interpréter les courbes d'apprentissage en phase d'apprentissage de l'algorithme.   



## Partie 1 : Théorie (30min)

1. Equations de Bellman pour la résolution d'un processus décisionnel de Markov avec connaissance parfaite des modèles.

  - Soit un processus decisionnel de Markov donné par le tuple (S, A, p, r). Considérons une politique quelconque notée **pi**. Écrire l'expression d'évaluation de cette politique, dans le cadre du critère des recompenses décomptées de paramètre de décompte **gamma**.
  - En déduire les **équations de Bellman**, i.e., système d'équations dont la résolution permet de déterminer la fonction de valeur en tout état de la politique fixée. 
  - Démontrer de manière similaire les **équations d'optimalité de Bellman**, i.e., système d'équations dont la résolution permet de déterminer la politique optimale. 


2. Calculer sur papier la fonction de valeur optimale associée au labyrinthe ci-dessous : 

![Image](https://raw.githubusercontent.com/blavad/IAT/master/TP1/tests/MazeEx2NR.png)

  - paramètre de décompte : gamma = 1.0
  - R(s, a) = -1 pour tout s, pour tout a
  - **Transition déterministes** 


3. Calculer sur papier la fonction de valeur optimale associée au même labyrinthe
  - parameters : gamma = 1.0
  - R(s, a) = -1 pour tout s, pour tout a
  - **Transition stochastiques** 
  On distinguera deux cas de figure. Si l'action est exécutable, c'est-à-dire qu'elle ne mène pas à un obstacle ou mur, alors la probabilité de succès est de 80%, la probabilité qu'elle échoue est de 20%. Lorsqu'une action échoue, l'agent reste dans la cellule courante. Si l'action est non exécutable, alors l'action échoue systématiquement et l'agent demeure dans la cellule courante. 


## Partie 2 : Implémenter l'algorithme "Value Iteration" (30min)

**Intro** : Quand on connait les modèles de la dynamique, p(s' | s,a) et r(s,a), on peut utiliser un algorithme de planification 
pour déterminer la politique optimale.

**Value Iteration**: Il s'agit d'une méthode de résolution des processus décisionnels de Markov avec connaissance parfaite des modèles de la dynamique et des récompenses. L'algorithme procède de façon iterative, mettant à jour la fonction de valeur, d'une iteration à l'autre jusqu'à ce que l'écart entre deux mises à jour est inférieur à un seuil à fixer, e.g., 0.01. Chaque mise à jour consiste à l'application des équations d'optimalité de Bellman énoncées plus tôt. 

- Lire et compléter le fichier Value Iteration (`TP1/agent/viagent.py`)
- Lancer le programme principal avec comme paramètre `vi` 
  - Lancer `python3 main.py vi
- Commenter la ligne `env = Maze(7, 7, min_shortest_length=15)`
  -  Utiliser l'environnement `env = Maze.from_file("tests/maze_ex2.txt")` qui correspond au labyrinthe de la partie 1
- Comparer les résultats obtenus aux résultats théoriques en affichant la fonction de valeur obtenue après résolution

## Partie 2.5 : Visualisation

- Visualiser l'évolution de la fonction de valeur avec la commande python3 logAnalysisV.py dans le dossier partie_2/visualisation/

## Partie 3 : Implémenter l'algorithme "Q-Learning" (40min)

**Intro** : Parfois, les modèles de la dynamique sont inconnus ou gigantesques. S'il est possible d'interagir avec le système directement et récupérer des informations au fil de l'eau, il est alors possible d'implémenter des algorithmes d'apprentissage par renforcement pour déterminer la politique optimale.

1. Lire et compléter le fichier Q-learning (agent/qagent.py)
2. Lancer le fichier main.py pour vérifier les résultats `python3 main.py qlearning`
3. Augmenter la taille du labyrinthe à (14, 14) et recommencer l'apprentissage 
  - Que remarque-t-on ?
  - Quelle(s) solution peut-on apporter (jouer avec les paramètres d'apprentissage - i.e. `n_episodes` et `max_steps`) ?
4. Modifier le paramètre `eps_profile` pour ne faire que de l'exploration ?
  - Analyser les résultats
  - Quel est l'intérêt de faire décroître ce paramètre ?

## Partie 3.5 : Visualisation

- Visualiser la courbe d'évolution de la Q-valeur avec la commande python3 main.py logAnalysisQ
- Visualiser l'évolution de la fonction de valeur avec la commande python3 logAnalysisV.py dans le dossier partie_3/visualisation/

