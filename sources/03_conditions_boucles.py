# %% [markdown]
# # 03 — Conditions et boucles
#
# Un système embarqué passe son temps à **tester** des grandeurs (une tension
# est-elle sous le seuil ?) et à **répéter** des opérations (balayer des voies,
# simuler une charge). C'est exactement ce que font `if` et les boucles `for`
# et `while`. Ce notebook montre aussi comment l'indentation remplace les
# accolades : en Python, la mise en forme **est** la syntaxe.
#
# **Objectifs du notebook**
#
# - écrire une décision avec `if` / `elif` / `else` et des comparaisons enchaînées ;
# - utiliser l'expression ternaire pour choisir une valeur ;
# - parcourir une liste avec `for`, `range` et `enumerate` ;
# - accumuler des mesures dans une variable ;
# - interrompre une boucle avec `break` et sauter un tour avec `continue` ;
# - simuler la charge d'un condensateur RC avec `while`.
#
# **Prérequis** : notebook `02` (types, comparaisons, opérateurs logiques).

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
# ## 1. `if` / `elif` / `else` : surveiller une batterie
#
# On teste les conditions **dans l'ordre** : dès qu'une est vraie, son bloc
# s'exécute et les suivants sont ignorés. L'ordre des seuils est donc capital :
# si l'on testait `< 12.0` avant `< 11.0`, l'état « critique » ne serait jamais
# atteint. C'est le même raisonnement qu'un comparateur à fenêtre.

# %%
tension_batterie = 11.4  # tension mesurée aux bornes de la batterie, en volts
if tension_batterie < 11.0:  # premier test : décharge profonde
    etat = "critique"  # ce bloc ne s'exécute que si la condition est vraie
elif tension_batterie < 12.0:  # sinon, on teste le seuil de batterie faible
    etat = "faible"  # 11,4 V tombe ici : entre 11,0 et 12,0 V
else:  # aucun des deux tests précédents n'est vrai
    etat = "ok"  # batterie au-dessus de 12,0 V
print(f"État : {etat}")  # affiche État : faible

# %% [markdown]
# Modifiez la valeur de `tension_batterie` (par exemple 10.8 puis 12.6) et
# réexécutez la cellule : le message change. Une seule branche est exécutée à
# chaque passage, jamais deux.

# %% [markdown]
# ## 2. Comparaisons enchaînées et expression ternaire
#
# Python autorise `11.0 <= u < 12.0`, équivalent à `11.0 <= u and u < 12.0`,
# ce qui évite d'écrire deux fois la variable. L'expression ternaire
# `a if condition else b` choisit une valeur en une ligne : pratique pour
# affecter un seuil ou un texte d'état.

# %%
tension = 11.4  # tension de la batterie, en volts
if 11.0 <= tension < 12.0:  # comparaison enchaînée : 11,0 <= tension ET tension < 12,0
    print("Fenêtre de batterie faible")  # s'affiche ici, la tension est dans la fenêtre

niveau = "critique" if tension < 11.0 else "normal"  # ternaire : deux valeurs possibles
print(niveau)  # affiche normal

# %% [markdown]
# ## 3. L'indentation est la syntaxe
#
# En Python, un bloc est délimité par les **deux-points** et par
# l'**indentation** de 4 espaces. Pas d'accolades ni de `end`. Une ligne non
# indentée après un `if` sort du bloc et s'exécute toujours. Mélanger tabulations
# et espaces provoque une `IndentationError` : configurez votre éditeur sur
# 4 espaces.

# %%
seuil = 2.0  # seuil de comparaison, en volts
mesure = 2.4  # tension effectivement mesurée, en volts
if mesure > seuil:  # les deux-points ouvrent un bloc
    ecart = mesure - seuil  # 4 espaces : cette ligne appartient au bloc
    print(f"Dépassement de {ecart:.2f} V")  # exécutée seulement si mesure > seuil
print("Fin du test")  # non indentée : hors du bloc, toujours exécutée

# %% [markdown]
# ## 4. Boucle `for` sur une liste
#
# `for valeur in liste:` parcourt les éléments un par un, sans indice à gérer.
# C'est la façon naturelle de traiter un lot de composants : on applique le
# même calcul à chaque résistance pour connaître le courant sous 5 V.

# %%
resistances = [100, 220, 470, 1000, 2200]  # valeurs de résistances en ohms
for r in resistances:  # r prend successivement chaque valeur de la liste
    courant = 5.0 / r  # loi d'Ohm pour une alimentation de 5 V
    print(f"R = {r:4d} Ω -> I = {courant * 1000:6.2f} mA")  # affiche le courant en mA

# %% [markdown]
# ## 5. `range` et `enumerate`
#
# `range(debut, fin, pas)` génère des entiers : `fin` est **exclue**. On
# l'utilise pour numéroter des voies ou balayer des points de mesure.
# `enumerate` fournit en plus l'indice, utile pour afficher un numéro de canal
# à côté de son nom.

# %%
for canal in range(4):  # range(4) produit 0, 1, 2, 3
    print(f"Voie analogique {canal}")  # affiche le numéro de la voie

for tension_point in range(0, 13, 3):  # de 0 à 12 inclus, par pas de 3 volts
    print(f"Point de mesure : {tension_point} V")  # affiche 0, 3, 6, 9 et 12 V

# %% [markdown]
# `enumerate` devient indispensable dès qu'il faut afficher un numéro à côté
# d'une valeur : numéro de canal, indice de point de mesure, rang d'un relevé.
# Il évite de maintenir un compteur à la main dans la boucle.

# %%
canaux = ["AIN0", "AIN1", "AIN2"]  # entrées analogiques du convertisseur
for index, nom in enumerate(canaux):  # enumerate renvoie l'indice ET la valeur
    print(f"Canal {index} : {nom}")  # affiche 0 : AIN0, 1 : AIN1, 2 : AIN2

# %% [markdown]
# ## 6. Le motif accumulateur
#
# Pour sommer des courants de branches, on initialise un accumulateur **avant**
# la boucle, puis on l'incrémente à chaque tour avec `+=`. La moyenne s'obtient
# en divisant par `len(liste)`. Ce motif revient sans cesse en traitement de
# mesures, et il est explicitement utilisé dans les algorithmes de filtrage.

# %%
courants_ma = [12.5, 8.0, 4.2, 15.3]  # courants des quatre branches, en mA
courant_total = 0.0  # accumulateur initialisé avant la boucle
for i in courants_ma:  # parcourt chaque mesure de courant
    courant_total += i  # ajoute la mesure à l'accumulateur
print(f"Somme des courants : {courant_total:.1f} mA")  # affiche 40.0 mA
moyenne = courant_total / len(courants_ma)  # len() renvoie le nombre de mesures
print(f"Moyenne : {moyenne:.2f} mA")  # affiche 10.00 mA

# %% [markdown]
# ## 7. `break` et `continue`
#
# `break` quitte la boucle immédiatement : on l'utilise quand une mesure
# sature et invalide toute la série. `continue` saute seulement le tour courant :
# on écarte une valeur aberrante sans arrêter l'acquisition. Les deux évitent
# d'imbriquer des `if` profonds.

# %%
codes_adc = [512, 700, 1023, 980]  # codes successifs lus sur le CAN
for code in codes_adc:  # parcourt la série de mesures
    if code == 1023:  # 1023 correspond à la saturation du convertisseur
        print("Saturation détectée, série abandonnée")  # message d'alerte
        break  # quitte immédiatement la boucle for
    print(f"Code accepté : {code}")  # exécuté tant qu'il n'y a pas saturation

# %% [markdown]
# Le second cas utilise `continue` : la mesure aberrante est écartée, mais
# l'acquisition se poursuit avec les valeurs suivantes. Choisir entre `break`
# et `continue`, c'est décider si l'erreur invalide la série entière ou
# seulement un point.

# %%
mesures = [0.0, 3.3, -1.0, 5.0]  # tensions lues, avec une valeur aberrante
for u in mesures:  # parcourt les mesures une par une
    if u < 0:  # une tension négative est impossible sur cette entrée
        continue  # ignore cette mesure et passe à la suivante
    print(f"Mesure valide : {u:.1f} V")  # affiche 0.0, 3.3 puis 5.0 V

# %% [markdown]
# ## 8. Boucle `while` : charge d'un condensateur RC
#
# Dans un circuit RC série, la tension aux bornes du condensateur suit
# `U(t) = E × (1 − e^(−t/RC))`. On simule le temps par pas de `dt` et on
# s'arrête quand `U` atteint 63 % de `E` (soit une constante de temps `tau`).
# La condition `n < 100` est un **garde-fou** : sans lui, une erreur de seuil
# provoquerait une boucle infinie qui bloque le noyau.
#
# **Attention** : `while True:` sans `break` ni modification de la condition
# bloque le notebook. Si cela arrive, utilisez le bouton *Interrupt* du noyau.

# %%
import math  # fournit math.exp pour l'exponentielle

e_alim = 5.0  # tension d'alimentation du circuit RC, en volts
r_charge = 10e3  # résistance de 10 kΩ, en ohms
c_charge = 100e-6  # condensateur de 100 µF, en farads
tau = r_charge * c_charge  # constante de temps tau = R × C, ici 1 s
cible = 0.63 * e_alim  # seuil de 63 % de E, soit environ 3,15 V

t = 0.0  # temps courant de la simulation, en secondes
dt = 0.1  # pas de simulation de 100 ms
u_c = 0.0  # tension aux bornes du condensateur au départ
n = 0  # compteur d'itérations
while u_c < cible and n < 100:  # boucle tant que le seuil n'est pas atteint (garde-fou à 100)
    t += dt  # avance le temps d'un pas
    u_c = e_alim * (1 - math.exp(-t / tau))  # loi de charge U(t) = E × (1 − e^(−t/tau))
    n += 1  # incrémente le compteur d'itérations
print(f"U = {u_c:.3f} V à t = {t:.1f} s après {n} itérations")  # environ 1,0 s et 10 itérations
print(f"Constante de temps tau = {tau:.1f} s")  # affiche 1.0 s : 63 % atteint à t = tau

# %% [markdown]
# ## À vous de jouer
#
# Complétez les cellules suivantes. Chaque `TODO` est un commentaire : les
# cellules s'exécutent donc sans erreur même non complétées.

# %% [markdown]
# **Exercice 1** — Une alimentation mesure 12.6 V. Écrivez un `if` / `elif` /
# `else` qui affiche `critique` sous 11.0 V, `faible` sous 12.0 V, `ok` sinon.

# %%
# TODO : créez tension = 12.6

# TODO : écrivez if / elif / else et affichez l'état

# %% [markdown]
# **Exercice 2** — Parcourez la liste `[4.7, 10, 22, 47]` (en kΩ) avec une
# boucle `for` et affichez chaque valeur convertie en ohms.

# %%
# TODO : créez la liste valeurs_ko = [4.7, 10, 22, 47]

# TODO : bouclez avec for et affichez valeur * 1000 en ohms

# %% [markdown]
# **Exercice 3** — Dans `[100, 220, 470, 1000, 2200]`, trouvez la première
# résistance supérieure ou égale à 1000 Ω avec une boucle `for` et un `break`,
# puis affichez-la.

# %%
# TODO : bouclez sur la liste et testez r >= 1000

# TODO : affichez la valeur trouvée puis sortez avec break

# %% [markdown]
# ## À retenir
#
# - `if` / `elif` / `else` teste les conditions dans l'ordre : le premier vrai gagne.
# - L'indentation de 4 espaces délimite les blocs, les deux-points les ouvrent.
# - `for x in liste` parcourt les valeurs ; `range(debut, fin, pas)` génère des entiers.
# - `enumerate` donne l'indice et la valeur ; l'accumulateur `+=` somme des mesures.
# - `break` sort de la boucle, `continue` saute un tour.
# - `while` répète tant que la condition est vraie : prévoyez toujours un garde-fou.
#
# **Notebook suivant** : `04 — Fonctions et modules`, pour regrouper ces calculs
# dans des fonctions réutilisables et testables.
