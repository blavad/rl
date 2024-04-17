# TP n°3 : Résolution d'un labyrinthe fixé par apprentissage

Dans ce problème, on s'intéresse à la résolution d'un labyrinthe fixé de taille quelconque. Les compétences travaillées durant cette activité sont les suivantes :

- Résoudre un problème de décision de Markov dont le modèle de la dynamique est inconnu
- Implémenter l'algorithme **Q-learning**
- Implémenter l'algorithme **Sarsa**

## Partie 0 : Pour commencer

10min

1. Ouvrir le dossier des TPs dans un terminal

2. Activer l'environnement virtuel python créé lors du TP n°2

   ```bash
    source .venv/bin/activate # activation de l'environnement
   ```

### Vérifier l'installation

```bash
# Dans le dossier TP3-ModelFree
python3 main.py random
```

## Partie 1 : Algorithme Q-Learning

45min

### Intro

Parfois, les modèles de la dynamique ($\mathcal{T}$ et $\mathcal{R}$) sont inconnus ou de grande taille. S'il est possible d'interagir avec le système directement et de récupérer des informations au fil de l'eau, il est alors possible d'implémenter des algorithmes d'apprentissage par renforcement pour déterminer la politique optimale.

### Implémentation

1.  Lire et compléter le fichier `agent/qagent.py`
2.  Lancer le fichier main.py pour vérifier les résultats

    > python main.py qlearning

3.  Augmenter la taille du labyrinthe à **(14, 14)** et recommencer l'apprentissage

4.  Que remarque-t-on ?
5.  Quelle(s) solution(s) peut-on apporter ?

    > Indice : Jouez avec les paramètres d'apprentissage - i.e. `n_episodes` et `max_steps`

6.  Modifier le paramètre `eps_profile` pour ne faire que de l'exploration.

7.  Analyser les résultats
8.  Quel est l'intérêt de faire décroître ce paramètre ?

### Visualisation
20min

1. Modifier le code existant afin de visualiser l'apprentissage

    > Indice : On pourra en fin d'apprentissage afficher $\max_{a}q(s_{start}, a)$ en fonction des $épisodes$.
   

## Partie 2 : SARSA

1h00

Il s'agit d'une autre méthode de d'estimation d'une fonction de valeur Q. Cette méthode met à jour directement la politique d'action mais est plus coûteuse en temps.

### A faire

Sur un modèle similaire au fichier `qagent.py`, créer un fichier `sarsa.py` qui implémente l'algorithme de SARSA.

Pour cela :

1. Créer le fichier `sarsa.py` dans le dossier `agents`
1. Créer une classe `SarsaAgent` qui hérite de `AgentInterface`
1. Implémenter un constructeur avec la signature suivante :

   > `def __init__(self, maze: Maze, gamma: float, alpha: float)`

1. Implémenter la méthode `updateQ`

1. Implémenter la méthode `solve` qui résout le problème de décision

   > `def solve(self, error: float)`

1. Surcharger la méthode `select_action`

1. Modifier le fichier `main.py` pour tester votre algorithme. On pourra notamment insérer les lignes suivantes à l'endroit voulu :

   ```python
   elif agent == "sarsa":
        agent = SarsaAgent(env, gamma, alpha)
        agent.learn(env, n_episodes, max_steps)
        test_maze(env, agent, max_steps, speed=0.1, display=True)
   ```

## Et ensuite ?

Déjà terminé ? Vous pouvez commencer [le TP n°4 sur les algorithmes d'apprentissage par renforcement profond](../TP4-DeepRL/README.md).
