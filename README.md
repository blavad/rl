# Introduction à l'apprentissage par renforcement, 2024

Ce cours a pour objectif d'initier les étudiants à l'apprentissage par renforcement.

L'apprentissage par renforcement est une branche de l'intelligence artificielle qui consiste à apprendre les actions à réaliser, à partir d'expériences, de façon à optimiser une récompense quantitative au cours du temps.

## Pour commencer

10 min

### Environnement de travail

Pour bien apprendre pendant ces travaux pratiques il est conseillé d'avoir les éléments suivants sur sa machine:

- python (version >= 3.10)
- un accès à terminal
  - pour les utilisateurs `Windows`, l'utilisation de `wsl` est fortement recommandée
- un éditeur de texte. Il existe des environnements de développements très performants que vous serez amené à utiliser en entreprise. Pourquoi ne pas les utiliser dès à présent ?
  - `Visual Studio Code`, `PyCharm`, `SublimeText`, ...

### Récupération des travaux pratiques

1. Récupérer le code source

   ```bash
   git clone https://github.com/blavad/rl.git
   cd rl
   ```

2. Créer un environnement virtuel python

   ```bash
    pip3 install virtualenv # installation virtualenv
    python3 -m venv .venv # création d'un environnement
    source .venv/bin/activate # activation de l'environnement
   ```

3. Installer des dépendances des TPs

   ```bash
   pip3 install -r requirements.txt
   ```

### Vérifier l'installation

```bash
cd TP1-ModelBased
python3 main.py random
```

## Et ensuite ?

Si votre environnement de travail est près. Vous pouvez commencer [le TP n°1 sur les processus de décision de Markov](./TP1-MDP/README.md).
