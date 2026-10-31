# Python pour l'électronique — spécial IA

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/VladdKid/python-electronique-ia/HEAD?labpath=notebooks%2F01_prise_en_main_jupyter.ipynb)
[![Ouvrir dans Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/VladdKid/python-electronique-ia/blob/main/notebooks/01_prise_en_main_jupyter.ipynb)

Cours de Python en 12 notebooks Jupyter conçu pour des étudiants en
électronique. Chaque notion est illustrée par des tensions, des capteurs, des
signaux et des composants, puis prolongée vers l'intelligence artificielle :
traitement du signal, apprentissage automatique et réseaux de neurones.

**Chaque ligne de code est commentée en français** et chaque bloc est précédé
d'une explication. Les exercices sont fournis directement dans les notebooks.

## Ce que vous saurez faire à la fin

- écrire un programme Python lisible (types, boucles, fonctions, structures) ;
- acquérir et analyser des mesures avec NumPy, Matplotlib et Pandas ;
- filtrer un signal bruité et l'analyser dans le domaine fréquentiel (FFT) ;
- entraîner un modèle de régression et de classification avec scikit-learn ;
- comprendre et coder un réseau de neurones, puis utiliser PyTorch ;
- mener un projet complet : tri de composants par CNN et détection
  d'anomalies sur des mesures capteurs.

Aucun matériel n'est nécessaire : toutes les mesures et images sont simulées.
Pour une utilisation sans installation, voir la section
[Utiliser le cours en ligne](#utiliser-le-cours-en-ligne).

## Installation

Prérequis : Python 3.11 ou plus récent (testé avec Python 3.12).

```bash
# 1. Se placer à la racine du dépôt (le dossier qui contient ce README)
cd chemin/vers/le-depot

# 2. Créer un environnement virtuel
python -m venv .venv

# 3. L'activer
# Windows (PowerShell) :
.venv\Scripts\Activate.ps1
# Windows (cmd) :
.venv\Scripts\activate.bat
# macOS / Linux :
source .venv/bin/activate

# 4. Installer les dépendances
pip install -r requirements.txt

# 5. Générer les jeux de données (déjà présents, mais reproductibles)
python scripts/generer_donnees.py
```

> PyTorch est volumineux. Sur une machine sans GPU, la version CPU suffit :
> `pip install torch --index-url https://download.pytorch.org/whl/cpu`
> (voir la documentation officielle pour les autres plateformes).

## Lancer le cours

**Important : lancez Jupyter depuis la racine du dépôt**, car les notebooks
chargent les données avec des chemins relatifs (`data/...`).

```bash
jupyter lab
# ou
jupyter notebook
```

Ouvrez ensuite le dossier `notebooks/` et commencez par `01_...`.

## Utiliser le cours en ligne

Deux options fonctionnent sans rien installer. Les jeux de données sont inclus
dans le dépôt.

### Binder (recommandé, aucune installation)

Cliquez sur le badge Binder en haut du README, ou ouvrez
<https://mybinder.org/v2/gh/VladdKid/python-electronique-ia/HEAD>.

Le **premier** lancement construit l'image Docker (5 à 10 minutes, PyTorch CPU
inclus) ; les suivants sont plus rapides. Binder démarre à la racine du dépôt :
les chemins `data/...` fonctionnent directement et les 12 notebooks sont
disponibles dans `notebooks/`. La configuration se trouve dans
`binder/requirements.txt`.

### Google Colab

[![Ouvrir dans Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/VladdKid/python-electronique-ia/blob/main/notebooks/01_prise_en_main_jupyter.ipynb)

Ouvrez le notebook voulu depuis GitHub avec Colab, par exemple
`https://colab.research.google.com/github/VladdKid/python-electronique-ia/blob/main/notebooks/06_numpy_signaux.ipynb`.

Exécutez **d'abord la cellule « Utilisation en ligne »** placée au début de
chaque notebook : dans Colab, elle clone le dépôt public et place le notebook à
sa racine pour que les données soient accessibles. Sur votre ordinateur, cette
cellule ne fait rien.

Colab fournit déjà NumPy, pandas, scikit-learn et PyTorch ; les versions
peuvent différer légèrement de `requirements.txt`, donc les scores affichés
peuvent varier de quelques dixièmes. Le dossier `models/` et les figures
produites disparaissent à la fin de la session Colab.

> **JupyterLite** (Jupyter entièrement dans le navigateur, sans serveur) n'est
> pas compatible avec ce cours : PyTorch n'y est pas disponible, donc les
> notebooks 11 et 12 ne peuvent pas s'exécuter.

## Modifier le cours (enseignants)

Les notebooks sont générés depuis les fichiers `sources/*.py`, écrits au
format « percent » (le format de Jupytext) : une cellule de code commence par
`# %%`, une cellule Markdown est écrite en commentaires `# `. Pour modifier
le contenu, éditez le fichier `.py` puis régénérez :

```bash
# Tous les notebooks, avec exécution et sorties enregistrées
python tools/build_notebooks.py --execute

# Un seul notebook
python tools/build_notebooks.py 06_numpy_signaux.py --execute
```

Sans `--execute`, le notebook est reconstruit mais les sorties déjà
enregistrées sont perdues. Vous pouvez aussi modifier directement un `.ipynb`
dans Jupyter, mais régénérez ensuite depuis `sources/` pour éviter que les
deux versions ne divergent.

## Parcours

| # | Notebook | Contenu | Durée indicative |
|---|----------|---------|------------------|
| 01 | `prise_en_main_jupyter` | cellules, `print`, f-strings, loi d'Ohm | 30 min |
| 02 | `variables_types_operateurs` | types, unités, opérateurs, impédances complexes | 45 min |
| 03 | `conditions_boucles` | seuils, boucles, `for`/`while`, charge RC | 45 min |
| 04 | `fonctions_modules` | fonctions, modules `math`/`cmath`, CAN | 45 min |
| 05 | `structures_donnees` | listes, tuples, dictionnaires, ensembles, BOM | 45 min |
| 06 | `numpy_signaux` | tableaux, vectorisation, RMS, SNR, bruit | 50 min |
| 07 | `matplotlib_signaux` | tracés, sous-graphiques, calibration, export | 40 min |
| 08 | `pandas_mesures` | CSV, filtres, statistiques, séries temporelles | 50 min |
| 09 | `traitement_signal` | moyenne glissante, filtre passe-bas, FFT | 60 min |
| 10 | `machine_learning_sklearn` | régression, classification, validation croisée | 60 min |
| 11 | `reseaux_neurones` | neurone, descente de gradient, XOR, PyTorch | 75 min |
| 12 | `projet_final_ia` | CNN de tri de composants, IsolationForest | 75 min |

Total estimé : environ 11 heures, à répartir en 12 séances ou 6 séances
doubles. Les notebooks 01 à 07 peuvent servir de remise à niveau rapide en
Python scientifique ; les notebooks 09 à 12 forment le cœur « IA ».

## Conventions pédagogiques

- Un paragraphe d'introduction précède chaque cellule de code.
- Chaque ligne exécutable porte un commentaire `#` qui explique son rôle.
- Une section **À vous de jouer** propose des exercices ; les cellules
  contiennent des `TODO` à compléter et s'exécutent sans erreur en l'état.
- Une section **À retenir** clôt chaque notebook.

## Données et artefacts

Les jeux de données sont **simulés** et reproductibles (graines fixes) :

| Fichier | Contenu |
|---------|---------|
| `data/calibration_thermistance.csv` | 150 points résistance / température d'une CTN 10 kΩ |
| `data/maintenance_capteurs.csv` | 1600 mesures vibratoires et thermiques d'un moteur, étiquetées sain / défaillant |
| `data/mesures_capteur.csv` | 1800 mesures d'ambiance toutes les 10 s, avec quelques pics de courant |

`scripts/generer_donnees.py` régénère ces fichiers à l'identique. Le jeu de
maintenance contient volontairement **3 % d'étiquettes erronées**, comme dans
un vrai projet : aucun modèle ne peut atteindre 100 % de précision.

Les figures exportées vont dans `figures/`, les modèles entraînés dans
`models/`. Ces dossiers sont recréés par les notebooks.

## Dépannage

- **`ModuleNotFoundError`** : l'environnement virtuel n'est pas activé, ou
  `pip install -r requirements.txt` n'a pas été exécuté dans cet environnement.
- **`FileNotFoundError: data/...`** : Jupyter a été lancé depuis `notebooks/`.
  Relancez-le depuis la racine du dépôt.
- **Le noyau ne répond plus** : menu *Kernel → Restart* puis réexécutez les
  cellules de haut en bas.
- **Les graphiques ne s'affichent pas** : exécutez la cellule
  `import matplotlib.pyplot as plt` avant toute cellule de tracé.
- **Windows et accents** : les notebooks sont en UTF-8 ; ouvrez-les avec
  Jupyter plutôt qu'avec un éditeur configuré en CP1252.

## Organisation du dépôt

```
.
├── notebooks/                 # les 12 notebooks, prêts à l'emploi
├── sources/                   # sources éditables (format percent)
├── data/                      # jeux de données simulés
├── scripts/
│   └── generer_donnees.py     # régénère les CSV
├── tools/
│   └── build_notebooks.py     # sources/ -> notebooks/
├── figures/                   # sorties graphiques éventuelles
├── models/                    # modèles sauvegardés par les notebooks
├── requirements.txt
└── README.md
```

Bon cours !
