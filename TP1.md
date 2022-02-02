# TP 1 : Résolution d'un labyrinthe fixé

Dans ce problème, on s'intéresse à la résolution d'un labyrinthe fixé de taille quelconque.

## Partie 1 : Théorie (30min)

1. Rappeler les équations de Bellman pour V

2. Calculer fonction de valeur V(s) associer au labyrinthe suivant (dessiner un labyrinthe et le mettre dans un fichier .txt pour être utilisé dans les tests)
  - parameters : gamma = 1.0
  - R_t(s, a) = -1 pour tout s, pour tout a, et pour tout t
  - **Transition déterministes** 

3. Calculer fonction de valeur V(s) associer au même labyrinthe (cette fois le modèle de transition est stochastique)
  - parameters : gamma = 1.0
  - R_t(s, a) = -1 pour tout s, pour tout a, et pour tout t
  - **Transition stochastiques** (modèle à définir) 

## Partie 2 : Value Iteration (30min)

**Intro** : Quand on connait le modèle de la dynamique (T et R), on peut utiliser un algorithme de planification 
pour déterminer la politique d'action optimale.

- Lire et compléter le fichier Value Iteration (agent/viagent.py)
- Lancer la programme principal avec comme parametre VI et le chemin vers le laryrinthe en txt
  - `python main.py vi ./assets/maze_exo1.txt`
  - `python main.py vi ./assets/maze_exo1.txt --stochastic`
- Comparer les résultats obtenus au résultats théoriques
- **Bonus** : Algo d'évaluation de la valeur ou TD-learning ou Monte-Carlo

## Partie 3 : Q-Learning (40min)

**Intro** : Parfois, le modèle de la dynamique (T et R) est inconnu. On peut simplement échantilloner interagir avec le problème et récupérer 
des informations au fil de l'eau. On parle d'algorithmes d'apprentissage (par renforcement).

1. Lire et compléter le fichier Q-learning (agent/qagent.py)
2. Lancer le fichier main.py pour vérifier les résultats `python main.py qlearning`
3. Augmenter la taille du labyrinthe à (14, 14) et recommencer l'apprentissage 
  - Que remarque-t-on ?
  - Quelle(s) solution peut-on apporter (jouer avec les paramètres d'apprentissage - i.e. `n_episodes` et `max_steps`) ?
4. Modifier le paramètre `eps_profile` pour ne faire que de l'exploration ?
  - Analyser les résultats
  - Quel est l'intérêt de faire décroître ce paramètre ?
  
**Bonus**: Ici on pourrait comparer avec SARSA et montrer que l'algo SARSA ne converge vers l'optimal que quand *epsilon* décroit vers 0 car c'est un algo d'évaluation de politique au même titre que TD-learning.

## Partie 3 : Reinforce (Complémentaire)
(à faire)