# Intelligence Artificielle pour les Télécoms (IAT), 2022

Le cours d'IAT a pour objectif d'initier les étudiants de la spécialité Télécoms aux méthodes récentes d'intelligence artificielle. Dans ce sous-module, nous nous interesserons tout particulièrement à l'apprentissage par renforcement et ses variantes profondes. 

## 0. Pour commencer
10 min
### Mettre en place son environnement de travail

1. Récupérer le code source 
```bash
git clone https://github.com/blavad/IAT.git
cd IAT
```

2. Installer des dépendances
```bash
pip3 install setuptools==65.4.0
pip3 install git+https://github.com/openai/gym.git@9180d12e1b66e7e2a1a622614f787a6ec147ac40
pip3 install -r requirements.txt  install importlib-metadata==4.13.0
```

### Vérifier l'installation
```bash
cd TP1
python3 main.py random
```

## TP 1. Résoudre un labyrinthe fixé
2h

Lire les consignes dans `TP1.md`.

## TP 2. Résoudre tous les labyrinthes
2h

1. Installer les dépendances du TP 2. 
```bash
pip3 install -r TP2/requirements_2.txt
```

2. Lire les consignes dans `TP2.md`.
