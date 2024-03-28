# Introduction à l'apprentissage par renforcement, 2024

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
pip3 install virtualenv
python3 -m virtualenv venv
source venv/bin/activate
pip3 install "setuptools<=65.4.0"
pip3 install git+https://github.com/openai/gym.git@9180d12e1b66e7e2a1a622614f787a6ec147ac40
pip3 install -r requirements.txt  install "importlib-metadata<=4.13.0"
```

## for having venv without python3 -m

echo 'export PATH=$PATH:~/.local/bin' >> ~/.bashrc
source ~/.bashrc

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
python3 -m virtualenv venv
source venv/bin/activate
pip3 install "setuptools<=65.4.0"
pip3 install git+https://github.com/openai/gym.git@9180d12e1b66e7e2a1a622614f787a6ec147ac40
pip3 install -r requirements_2.txt  install "importlib-metadata<=4.13.0"
```

2. Lire les consignes dans `TP2.md`.

## note : s'il y a des problèmes d'installations liés à gym, aller modifier render_human(..) de world/maze.py pour :

```bash
    def render_human(self, mode='human'):
        '''
        from gymgrid2.envs.classic_control import rendering
        if self.viewer is None:
            self.viewer = rendering.SimpleImageViewer()
        return self.viewer.imshow(self.render(mode='rbg_array'))
        '''
        #for i in range(self.nx):
        #    for j in range(self.ny):

        print("\n")
        for i in range(self.nx):
            line = "|";
            for j in range(self.ny):
                if (i==self.terminal_state[0] and j==self.terminal_state[1]):
                    sys.stdout.write(fg("blue")+"G")
                    #line+="G"
                elif (self.loc[0]==i and self.loc[1]==j):
                    #line+="x"
                    sys.stdout.write(fg("rosy_brown")+"x")
                elif (i==self.init_state[0] and j==self.init_state[1]):
                    #line+="s"
                    sys.stdout.write(fg("grey_42")+"s")
                elif (self.maze[i][j]==1):
                    #line+="w"
                    sys.stdout.write(fg("white")+"w")
                else:
                    #line+=" "
                    sys.stdout.write(" ")
            #print(fg('blue')+line+"|")
            print("")
```
