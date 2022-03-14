# TP 1 : Résolution d'un labyrinthe fixé

Dans ce problème, on s'intéresse à la résolution d'un labyrinthe fixé de taille quelconque.

## Partie 1 : Théorie (30min)

1. Rappeler les équations d'optimalité de Bellman pour V^*

2. Calculer fonction de valeur V^* associer au labyrinthe suivant (dessiner un labyrinthe et le mettre dans un fichier .txt pour être utilisé dans les tests)
  - parameters : gamma = 1.0
  - R(s, a) = -1 pour tout s, pour tout a
  - **Transition déterministes** 

3. Calculer fonction de valeur V(s) associer au même labyrinthe (cette fois le modèle de transition est stochastique)
  - parameters : gamma = 1.0
  - R(s, a) = -1 pour tout s, pour tout a
  - **Transition stochastiques** (modèle à définir) 

## Partie 2 : Implémenter l'algorithme "Value Iteration" (30min)

**Intro** : Quand on connait les modèles de la dynamique (T et R), on peut utiliser un algorithme de planification 
pour déterminer la politique optimale.

- Lire et compléter le fichier Value Iteration (agent/viagent.py)
- Lancer la programme principal avec comme paramètre VI et le chemin vers le laryrinthe en .txt
  - `python3 main.py vi ./assets/maze_exo1.txt`
  - `python3 main.py vi ./assets/maze_exo1.txt --stochastic`
- Comparer les résultats obtenus au résultats théoriques
- **Bonus** : Implémenter l'algorithme d'évaluation de la valeur ou TD-learning ou Monte-Carlo

## Partie 2.5 : Visualisation

- Implémenter le stockage de l'évolution de fonction de valeur au cours de la résolution dans un fichier logAnalysisVi.csv.
- Ecrire un script d'analyse du fichier logAnalysis.csv pour visualiser l'évolution de la fonction de valeur
- Visualiser l'évolution de la fonction de valeur

- Tips : vous pouvez utiliser le module "animation" de matplotlib

## Partie 3 : Implémenter l'algorithme "Q-Learning" (40min)

**Intro** : Parfois, les modèles de la dynamique (T et R) sont inconnus ou gigantesques. S'il est possible d'interagir avec le système directement et récupérer 
des informations au fil de l'eau, il est alors possible d'implementer des algorithmes d'apprentissage par renforcement pour déterminer la politique optimale.

1. Lire et compléter le fichier Q-learning (agent/qagent.py)
2. Lancer le fichier main.py pour vérifier les résultats `python main.py qlearning`
3. Augmenter la taille du labyrinthe à (14, 14) et recommencer l'apprentissage 
  - Que remarque-t-on ?
  - Quelle(s) solution peut-on apporter (jouer avec les paramètres d'apprentissage - i.e. `n_episodes` et `max_steps`) ?
4. Modifier le paramètre `eps_profile` pour ne faire que de l'exploration ?
  - Analyser les résultats
  - Quel est l'intérêt de faire décroître ce paramètre ?

## Partie 3.5 : Visualisation

- Implémenter le stockage de l'évolution de fonction de valeur au cours de la résolution dans un fichier logAnalysisVi.csv.
- Visualiser l'évolution de la fonction de valeur
- Que constatez-vous?

- Implémenter le stockage de l'évolution de la Q-valeur à l'état initial au cours de la résolution dans un fichier logAnalysis.csv.
- Ecrire un script d'analyse du fichier logAnalysis.csv pour visualiser l'évolution de la Q-valeur
- Visualiser la courbe d'évolution de la Q-valeur

**Bonus**: Comparer avec SARSA et montrer que l'algo SARSA ne converge vers l'optimal que quand *epsilon* décroit vers 0 car c'est un algo d'évaluation de politique au même titre que TD-learning.

## Partie 4 : Reinforce (Complémentaire)
(à faire)

## Partie 5 : HSVI (Bonus)
- Modéliser le problème comme un MDP (Markov Decision Process), voir http://researchers.lille.inria.fr/~lazaric/Webpage/MVA-RL_Course14_files/notes-lecture-02.pdf 
- Compléter hsvi.py
- Résoudre le problème en utilisant l'algorithme hsvi
- Que constatez-vous des performances d'hsvi par rapport à VI & Q-learning?
