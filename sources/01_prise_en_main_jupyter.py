# %% [markdown]
# # 01 — Prise en main : Jupyter et premiers pas en Python
#
# Bienvenue ! Ce cours est écrit pour des étudiants en électronique : chaque
# notion Python est illustrée avec des tensions, des courants, des composants
# et des mesures. Les notebooks suivants s'appuient sur celui-ci, avancez donc
# dans l'ordre.
#
# **Objectifs du notebook**
#
# - comprendre ce qu'est une cellule et comment exécuter du code ;
# - afficher un résultat avec `print` et les f-strings ;
# - écrire des commentaires ;
# - faire un premier calcul utile : la loi d'Ohm.
#
# **Comment exécuter une cellule ?** Cliquez dedans puis appuyez sur
# `Maj + Entrée` (ou `Ctrl + Entrée`). Le résultat apparaît juste en dessous.
# Exécutez toujours les cellules **de haut en bas**.

# %% [markdown]
# ## Utilisation en ligne (Google Colab)
#
# Cette cellule ne fait rien sur votre ordinateur. Dans Google Colab, elle
# clone le dépôt public (données comprises) et place le notebook à sa racine,
# afin que les chemins `data/...` fonctionnent. Exécutez-la en premier.

# %%
import os  # manipulation des chemins et du dossier courant
import subprocess  # lance la commande git clone
import sys  # permet de détecter l'environnement Colab

DEPOT = "https://github.com/VladdKid/python-electronique-ia.git"  # adresse publique du cours
DOSSIER = "python-electronique-ia"  # dossier local créé par le clone

if "google.colab" in sys.modules and not os.path.isdir("data"):  # uniquement dans Colab, si les données manquent
    if not os.path.isdir(DOSSIER):  # évite de cloner deux fois le dépôt
        subprocess.run(["git", "clone", "--depth", "1", DEPOT, DOSSIER], check=True)  # télécharge le cours
    os.chdir(DOSSIER)  # se place à la racine du dépôt cloné
    print(f"Dépôt prêt : {os.getcwd()}")  # confirme le dossier de travail

# %% [markdown]
# ## 1. Deux types de cellules
#
# Un notebook est une suite de cellules de deux sortes :
#
# - **Markdown** : du texte (ce que vous lisez en ce moment) ;
# - **Code** : du Python exécutable.
#
# Le *noyau* (kernel) est le programme Python qui exécute les cellules. Il garde
# en mémoire les variables d'une cellule à l'autre, tant que vous ne le
# redémarrez pas.

# %% [markdown]
# ## 2. Premier programme
#
# `print` affiche une valeur. Une cellule de code affiche aussi automatiquement
# le résultat de sa **dernière expression**, sans `print`.

# %%
print("Bonjour à tous les électroniciens !")  # affiche un texte entre guillemets

# %%
2 + 3  # dernière expression : le notebook affiche automatiquement 5

# %% [markdown]
# Les deux cellules ci-dessus font la même chose que ceci, mais avec `print`
# explicite. `print` est préférable quand on veut un message lisible, surtout
# dans une boucle.

# %%
print(2 + 3)  # calcule 2 + 3 puis affiche le résultat (5)

# %% [markdown]
# ## 3. Les commentaires
#
# Un commentaire commence par `#`. Python l'ignore : il sert uniquement à
# expliquer le code aux humains. Dans tout ce cours, **chaque ligne de code est
# suivie d'un commentaire** qui explique ce qu'elle fait.

# %%
# Ceci est une ligne entièrement commentée : elle ne fait rien du tout
tension = 5.0  # range la valeur 5.0 dans une variable nommée tension
print(tension)  # affiche le contenu de la variable tension

# %% [markdown]
# ## 4. Loi d'Ohm : U = R × I
#
# Pour une résistance, la tension à ses bornes vaut `U = R × I`.
# Prenons une résistance de 220 Ω traversée par 20 mA (soit 0,02 A).
# **Attention aux unités** : on travaille toujours en volts, ohms et ampères.

# %%
resistance = 220.0  # résistance en ohms (symbole Ω)
courant = 0.02  # courant en ampères (20 mA = 0,02 A)
tension = resistance * courant  # calcule U = R × I
print(tension)  # affiche 4.4 (volts)

# %% [markdown]
# Un nombre brut `4.4` ne dit pas grand-chose. On utilise une **f-string**
# pour insérer une valeur dans un texte : on place un `f` devant les
# guillemets, puis `{expression}` là où la valeur doit apparaître.

# %%
print(f"La tension vaut {tension} V")  # insère la valeur de tension dans le texte
print(f"La tension vaut {tension:.2f} V")  # :.2f limite à 2 chiffres après la virgule
print(f"{resistance:.0f} Ω traversés par {courant * 1000:.0f} mA donnent {tension:.2f} V")  # combine calculs et unités

# %% [markdown]
# ## 5. L'ordre d'exécution compte
#
# Le noyau mémorise les variables **dans l'ordre où vous exécutez les
# cellules**, pas dans l'ordre du fichier. Si vous sautez la cellule suivante,
# la variable `tension_alim` n'existe pas et Python lève une `NameError`.
#
# Exécutez la cellule ci-dessous, puis celle d'après.

# %%
tension_alim = 12.0  # tension d'alimentation en volts, mémorisée par le noyau
print(f"Alimentation : {tension_alim} V")  # confirme la valeur stockée

# %%
courant_led = 0.015  # courant souhaité dans la LED : 15 mA = 0,015 A
resistance_serie = (tension_alim - 2.0) / courant_led  # R = (U_alim - U_led) / I, avec U_led ≈ 2 V
print(f"Résistance de série : {resistance_serie:.0f} Ω")  # affiche ≈ 667 Ω

# %% [markdown]
# ## 6. Vérifier le type d'une valeur
#
# Python distingue les nombres entiers (`int`), les nombres à virgule
# (`float`), les textes (`str`) et les booléens (`bool`). `type()` donne le type.

# %%
print(type(220))  # 220 est un entier -> int
print(type(220.0))  # 220.0 a une virgule -> float
print(type("220"))  # "220" entre guillemets est un texte -> str
print(type(True))  # True est un booléen -> bool

# %% [markdown]
# ## 7. À vous de jouer
#
# Complétez les cellules suivantes. Les commentaires indiquent ce qu'il faut
# écrire. Exécutez ensuite la cellule pour vérifier votre résultat.

# %% [markdown]
# **Exercice 1** — Une résistance de 330 Ω est branchée sous 5 V.
# Calculez le courant en ampères, puis affichez-le en milliampères.

# %%
# TODO : créez une variable resistance qui vaut 330.0

# TODO : créez une variable tension qui vaut 5.0

# TODO : calculez courant = tension / resistance

# TODO : affichez le courant en mA avec une f-string et 1 chiffre après la virgule

# %% [markdown]
# **Exercice 2** — La puissance dissipée vaut `P = U × I`.
# Reprenez les valeurs de l'exercice 1 et affichez la puissance en mW.

# %%
# TODO : calculez puissance = tension * courant

# TODO : affichez la puissance convertie en mW, arrondie à l'entier

# %% [markdown]
# **Exercice 3** — Affichez la phrase exacte :
# `Une LED de 2.0 V à 15 mA dissipe 30.0 mW`.
# Utilisez une f-string et le calcul `P = U × I`.

# %%
# TODO : définissez u_led = 2.0 et i_led = 0.015

# TODO : calculez p_led = u_led * i_led

# TODO : affichez la phrase demandée, puissance en mW avec 1 décimale

# %% [markdown]
# ## 8. À retenir
#
# - `Maj + Entrée` exécute la cellule courante.
# - `#` commence un commentaire.
# - `print(...)` affiche ; sans `print`, seule la dernière expression est affichée.
# - Une f-string s'écrit `f"texte {variable:.2f} unité"`.
# - Les unités doivent être cohérentes : volts, ohms, ampères.
#
# **Notebook suivant** : `02 — Variables, types et opérateurs`, où nous
# manipulerons notamment des nombres complexes pour les impédances.
