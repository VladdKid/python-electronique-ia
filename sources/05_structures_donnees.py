# %% [markdown]
# # 05 — Structures de données
#
# Une acquisition ne produit pas une valeur isolée mais un **lot** de mesures ;
# un montage ne contient pas un composant mais une **nomenclature** ; un capteur
# ne se pilote pas par une variable mais par une **carte de registres**. Les
# listes, tuples, dictionnaires et ensembles sont les quatre structures qui
# couvrent ces besoins. Choisir la bonne évite beaucoup de bugs.
#
# **Objectifs du notebook**
#
# - indexer, trancher et modifier une liste de mesures ;
# - écrire une compréhension de liste pour convertir mA en A ;
# - utiliser un tuple immuable pour un brochage et le déballer ;
# - décrire un composant et une carte de registres avec un dictionnaire ;
# - dédupliquer des codes de défaut avec un ensemble (`set`) ;
# - parcourir une nomenclature (liste de dictionnaires) et calculer un coût ;
# - distinguer mutable et immuable, et éviter le piège de l'alias.
#
# **Prérequis** : notebooks `03` et `04` (boucles, fonctions).

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
# ## 1. Listes : indice et tranche
#
# Une liste se crée entre crochets et conserve l'**ordre** d'insertion. L'indice
# commence à 0 ; un indice négatif compte depuis la fin, bien pratique pour la
# dernière mesure. Une tranche `debut:fin` extrait une sous-liste, `fin` étant
# exclue — exactement comme `range`.

# %%
mesures = [3.28, 3.31, 3.29, 3.30, 3.27]  # cinq tensions mesurées, en volts
print(mesures[0])  # indice 0 : première mesure (3.28 V)
print(mesures[-1])  # indice négatif : dernière mesure (3.27 V)
print(mesures[1:3])  # tranche des indices 1 et 2 : la borne 3 est exclue
print(mesures[:2])  # du début jusqu'à l'indice 2 exclu
print(mesures[-2:])  # les deux dernières mesures
print(len(mesures))  # nombre d'éléments de la liste : 5

# %% [markdown]
# ## 2. Modifier, trier, résumer
#
# `append` ajoute en fin de liste : c'est la façon naturelle d'accumuler des
# acquisitions. `min`, `max`, `sum` et `len` résument un lot de mesures sans
# écrire de boucle. Attention : `sorted()` renvoie une **nouvelle** liste, alors
# que `liste.sort()` modifie la liste en place.

# %%
releves = [12.1, 11.9, 12.4]  # tensions relevées au fil du temps, en volts
releves.append(12.2)  # ajoute une mesure en fin de liste
print(releves)  # affiche [12.1, 11.9, 12.4, 12.2]
print(min(releves), max(releves))  # minimum et maximum du lot
print(f"Moyenne : {sum(releves) / len(releves):.2f} V")  # somme divisée par le nombre de mesures
print(sorted(releves))  # renvoie une nouvelle liste triée par ordre croissant
print(releves)  # la liste d'origine garde l'ordre d'acquisition

# %% [markdown]
# ## 3. Compréhension de liste : convertir mA en A
#
# `[expression for x in liste]` construit une liste en une ligne. On peut y
# ajouter un filtre `if`. C'est la forme idiomatique pour changer d'unité ou
# ne garder que les mesures exploitables ; elle remplace une boucle avec
# `append`, plus longue et plus bruyante.

# %%
courants_ma = [12.5, 8.0, 4.2, 15.3]  # courants mesurés, en milliampères
courants_a = [i / 1000 for i in courants_ma]  # compréhension : convertit chaque valeur en ampères
print(courants_a)  # affiche [0.0125, 0.008, 0.0042..., 0.0153...] : les arrondis binaires apparaissent
forts = [i for i in courants_ma if i > 10]  # compréhension filtrante : ne garde que i > 10 mA
print(forts)  # affiche [12.5, 15.3]

# %% [markdown]
# ## 4. Tuples : un brochage ne change pas
#
# Un tuple s'écrit entre parenthèses et ne peut plus être modifié : il décrit
# une configuration figée, comme le rôle d'une broche de connecteur. Le
# **déballage** `num, sigle, role = broche` affecte chaque élément à une
# variable nommée, ce qui rend les boucles sur un brochage très lisibles.

# %%
broche = (14, "GPIO14", "TX série")  # tuple (numéro, nom, fonction) : non modifiable
numero, nom, fonction = broche  # déballage du tuple en trois variables
print(f"Broche {numero} : {nom} ({fonction})")  # affiche Broche 14 : GPIO14 (TX série)

brochage = [(1, "VCC", "alimentation"), (2, "GND", "masse"), (3, "SDA", "I2C données")]  # liste de tuples
for num, sigle, role in brochage:  # déballe chaque tuple au passage
    print(f"{num:2d} {sigle:4s} {role}")  # affiche une ligne par broche

# %% [markdown]
# Cette tentative de modification montre la garantie offerte par le tuple :
# Python refuse l'affectation et lève une `TypeError`, ce qui protège la
# description du brochage contre une erreur de frappe.

# %%
try:  # on tente de modifier un tuple
    broche[0] = 15  # interdit : un tuple est immuable
except TypeError:  # l'affectation lève une TypeError
    print("Un tuple ne peut pas être modifié")  # message pédagogique

# %% [markdown]
# ## 5. Dictionnaires : fiche composant et carte de registres
#
# Un dictionnaire associe des **clés** à des valeurs. On y range une fiche
# composant (`reference`, `valeur`, `tolerance`) ou une carte de registres dont
# les clés sont des adresses hexadécimales. `get(cle, defaut)` évite la
# `KeyError` quand une information manque, et affecter une clé existante met à
# jour sa valeur.

# %%
composant = {"reference": "R12", "valeur": 4700.0, "tolerance": 5}  # fiche d'une résistance
print(composant["reference"])  # accès par clé : affiche R12
print(composant.get("boitier", "0603"))  # get : valeur par défaut si la clé est absente
composant["valeur"] = 4680.0  # met à jour la valeur mesurée au lieu de la nominale
composant["puissance"] = 0.25  # ajoute une nouvelle clé au dictionnaire
print(composant)  # affiche le dictionnaire complet

# %% [markdown]
# Le dictionnaire sert aussi de carte de registres : la clé est l'adresse, la
# valeur le nom symbolique. `items()` parcourt les couples et permet d'afficher
# l'adresse en hexadécimal, comme dans une documentation de capteur.

# %%
registres = {0x00: "STATUS", 0x01: "CONFIG", 0x02: "DATA"}  # carte de registres d'un capteur
print(registres.keys())  # liste des adresses
print(registres.values())  # liste des noms de registres
for adresse, nom_reg in registres.items():  # items() fournit la clé ET la valeur
    print(f"0x{adresse:02X} : {nom_reg}")  # affiche 0x00 : STATUS, etc. en hexadécimal

# %% [markdown]
# ## 6. Ensembles : les codes de défaut
#
# Un `set` ne contient aucun doublon et ne garantit pas l'ordre. C'est la
# structure idéale pour des codes de défaut : ajouter deux fois le même défaut
# ne le duplique pas. Les opérateurs `&` (intersection) et `|` (union)
# répondent à « quels défauts sont communs aux deux sessions ? ».

# %%
defauts = {"SURTENSION", "SONDE_HORS_LIMITE", "SURTENSION"}  # set : le doublon est éliminé
print(defauts)  # affiche deux codes seulement, l'ordre n'est pas garanti
defauts.add("TEMPERATURE_HAUTE")  # ajoute un nouveau code de défaut
print(len(defauts))  # affiche 3 éléments distincts
nouveaux = {"TEMPERATURE_HAUTE", "BUS_ERREUR"}  # défauts relevés lors d'une seconde session
print(defauts & nouveaux)  # intersection : défauts présents dans les deux sessions
print(defauts | nouveaux)  # union : tous les défauts, sans doublon
print("SURTENSION" in defauts)  # test d'appartenance : affiche True, très rapide sur un set

# %% [markdown]
# ## 7. Structure imbriquée : la nomenclature (BOM)
#
# Une nomenclature est une **liste de dictionnaires** : chaque ligne décrit un
# composant avec les mêmes clés. On peut alors calculer un coût total,
# sélectionner les composants serrés en tolérance ou générer un tableau, sans
# jamais coder le nombre de composants en dur. C'est la structure de base de
# tout script de gestion de production.

# %%
nomenclature = [  # liste de dictionnaires : une nomenclature de trois lignes
    {"reference": "R1", "valeur": 1000.0, "tolerance": 5, "prix": 0.02},  # résistance 1 kΩ à 5 %
    {"reference": "R2", "valeur": 4700.0, "tolerance": 1, "prix": 0.03},  # résistance 4,7 kΩ à 1 %
    {"reference": "C1", "valeur": 100e-9, "tolerance": 10, "prix": 0.08},  # condensateur 100 nF à 10 %
]
total = 0.0  # accumulateur du coût total
for ligne in nomenclature:  # parcourt chaque composant de la nomenclature
    total += ligne["prix"]  # ajoute le prix unitaire au total
print(f"Coût total : {total:.2f} €")  # affiche 0.13 €

serres = [ligne["reference"] for ligne in nomenclature if ligne["tolerance"] <= 1]  # composants à 1 % ou mieux
print(serres)  # affiche ['R2']

# %% [markdown]
# ## 8. Mutable, immuable et le piège de l'alias
#
# Une liste est **mutable** (modifiable), un tuple ne l'est pas. L'affectation
# `b = a` ne copie rien : elle crée un second nom pour le **même** objet. Toute
# modification via `b` apparaît donc dans `a` — un classique quand on croit
# sauvegarder une série de mesures. Pour copier, on utilise `a.copy()` ou
# `list(a)`.

# %%
a = [1.0, 2.0, 3.0]  # liste d'origine
b = a  # ALIAS : b désigne la même liste, ce n'est pas une copie
b.append(4.0)  # modifie la liste à travers le nom b
print(a)  # affiche [1.0, 2.0, 3.0, 4.0] : a a changé aussi
c = a.copy()  # copy() crée une liste réellement indépendante
c.append(5.0)  # modifie seulement la copie c
print(a)  # a reste inchangée : [1.0, 2.0, 3.0, 4.0]
print(c)  # c contient la valeur ajoutée en plus

# %% [markdown]
# Même raisonnement pour un tuple : deux noms désignent le même objet, mais
# comme l'objet est immuable, aucun des deux ne peut le modifier. L'alias n'est
# donc dangereux que pour les objets mutables : listes, dictionnaires, ensembles.

# %%
t1 = (1.0, 2.0)  # un tuple est immuable
t2 = t1  # l'alias est ici sans danger : rien ne peut être modifié
print(t2)  # affiche (1.0, 2.0)

# %% [markdown]
# ## À vous de jouer
#
# Complétez les cellules suivantes ; les `TODO` sont des commentaires, donc les
# cellules s'exécutent déjà sans erreur.

# %% [markdown]
# **Exercice 1** — À partir de `[3.30, 3.35, 3.28, 3.33, 3.31]`, affichez la
# moyenne, le minimum et le maximum, puis la liste convertie en millivolts par
# compréhension.

# %%
# TODO : créez mesures = [3.30, 3.35, 3.28, 3.33, 3.31]

# TODO : affichez sum(mesures) / len(mesures), min(mesures) et max(mesures)

# TODO : créez mesures_mv par compréhension ([u * 1000 for u in mesures])

# %% [markdown]
# **Exercice 2** — Créez le dictionnaire `{"reference": "C3", "valeur": 100e-9,
# "tolerance": 10}`, ajoutez la clé `"boitier": "0603"`, puis affichez chaque
# couple clé / valeur avec `.items()`.

# %%
# TODO : créez le dictionnaire condensateur

# TODO : ajoutez la clé boitier

# TODO : bouclez sur condensateur.items() et affichez clé et valeur

# %% [markdown]
# **Exercice 3** — Deux sessions ont relevé les défauts `{"SURTENSION",
# "BUS_ERREUR"}` et `{"BUS_ERREUR", "SONDE_OUVERTE"}`. Affichez l'union,
# l'intersection, et le nombre de défauts distincts au total.

# %%
# TODO : créez les deux ensembles session_1 et session_2

# TODO : affichez session_1 | session_2 puis session_1 & session_2

# TODO : affichez len(session_1 | session_2)

# %% [markdown]
# ## À retenir
#
# - Liste : ordonnée, mutable, indice 0, tranches `debut:fin` (fin exclue).
# - `append`, `sorted`, `sum`, `min`, `max`, `len` résument un lot de mesures.
# - `[f(x) for x in liste if condition]` convertit et filtre en une ligne.
# - Tuple : immuable, idéal pour un brochage ; déballage `a, b = paire`.
# - Dictionnaire : clés → valeurs, `.keys()`, `.values()`, `.items()`, `get`.
# - `set` : pas de doublon, `&` intersection, `|` union, test `in` très rapide.
# - Liste de dictionnaires = nomenclature ; mutable contre immuable, et `copy()`.
#
# **Notebook suivant** : `06 — NumPy : calculer sur des signaux`, où ces listes
# deviendront des tableaux et ces boucles des opérations vectorisées.
