from setuptools import setup, find_packages

setup(name='iat',
      version='1.0.0',
      description='TP 1 : IA pour les Telecoms',
      url='https://github.com/blavad/iat-rl',
      author='David Albert',
      author_email='david.albert@insa-lyon.fr',
      packages=[],
      install_requires=['gym[classic_control]==0.18.0', 'gym[box2d]==0.18.0', 'gym[atari]==0.18.0', 'gym[board_game]==0.18.0', 'numpy', 'matplotlib']
)