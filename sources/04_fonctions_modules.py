# %% [markdown]
# # 04 — Fonctions et modules
#
# Un même calcul revient des dizaines de fois dans un projet : convertir un code
# de CAN, calculer un diviseur de tension, vérifier une tolérance. Le copier-coller
# est la première source d'erreurs. Une **fonction** donne un nom à ce calcul,
# le rend réutilisable et testable. Un **module** regroupe des fonctions déjà
# écrites (`math`) que l'on importe au lieu de les réécrire.
#
# **Objectifs du notebook**
#
# - définir une fonction avec `def`, des paramètres et `return` ;
# - documenter avec une docstring et utiliser des valeurs par défaut ;
# - renvoyer plusieurs valeurs sous forme de tuple ;
# - comprendre la portée des variables (locale / globale) ;
# - valider les entrées avec `raise ValueError` et `try` / `except` ;
# - importer un module (`math`) et écrire une courte fonction `lambda`.
#
# **Prérequis** : notebooks `02` et `03` (opérateurs, conditions, boucles).

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
# ## 1. Définir et appeler une fonction
#
# `def nom(parametres):` définit la fonction ; le corps est indenté et `return`
# renvoie le résultat. Rien ne s'exécute à la définition : le calcul n'a lieu
# qu'à l'**appel**. Le diviseur de tension est le premier réflexe de tout
# électronicien, autant l'écrire une seule fois.

# %%
def tension_diviseur(u_entree, r1, r2):  # définit une fonction à trois paramètres
    """Renvoie la tension de sortie d'un diviseur résistif."""  # docstring : une phrase
    return u_entree * r2 / (r1 + r2)  # formule U_s = U_e × R2 / (R1 + R2)

u_sortie = tension_diviseur(12.0, 10e3, 4.7e3)  # appelle la fonction : U_e, R1, R2
print(f"Tension de sortie : {u_sortie:.3f} V")  # affiche 3.837 V

# %% [markdown]
# ## 2. Valeurs par défaut et arguments nommés
#
# Un paramètre peut avoir une valeur par défaut (`bits=10`), utilisée quand
# l'appel ne le précise pas. Les arguments nommés (`vref=3.3`) rendent l'appel
# lisible et évitent les erreurs d'ordre. C'est indispensable pour un
# convertisseur dont la résolution et la référence changent d'un montage à
# l'autre.

# %%
def adc_brut_vers_volts(code, bits=10, vref=5.0):  # bits et vref ont une valeur par défaut
    """Convertit un code brut de CAN en tension, en volts."""  # docstring
    q = vref / 2 ** bits  # quantum : plus petite variation mesurable, en volts
    return code * q  # tension = code × quantum

print(f"{adc_brut_vers_volts(512):.3f} V")  # valeurs par défaut (10 bits, 5 V) : 2.500 V
print(f"{adc_brut_vers_volts(512, 12):.4f} V")  # 12 bits, référence 5 V : 0.6250 V
print(f"{adc_brut_vers_volts(code=700, vref=3.3):.3f} V")  # arguments nommés : 2.256 V

# %% [markdown]
# ## 3. Retourner plusieurs valeurs
#
# `return mini, maxi` renvoie un **tuple**. On peut le déballer directement dans
# deux variables, ce qui convient parfaitement aux bornes d'une tolérance. Une
# fonction ne peut pas « modifier » la variable de l'appelant : elle renvoie une
# nouvelle valeur, à vous de la stocker.

# %%
def plage_tolerance(valeur, pourcent):  # deux paramètres positionnels
    """Renvoie les bornes minimale et maximale d'une tolérance."""  # docstring
    mini = valeur * (1 - pourcent / 100)  # borne basse : nominale × (1 − tol)
    maxi = valeur * (1 + pourcent / 100)  # borne haute : nominale × (1 + tol)
    return mini, maxi  # sans crochets, Python construit un tuple

v_min, v_max = plage_tolerance(4700, 5)  # déballage du tuple dans deux variables
print(f"{v_min:.0f} Ω à {v_max:.0f} Ω")  # affiche 4465 Ω à 4935 Ω
bornes = plage_tolerance(4700, 5)  # on peut aussi conserver le tuple entier
print(bornes)  # affiche (4465.0, 4935.0)

# %% [markdown]
# ## 4. Portée des variables
#
# Une variable créée dans une fonction est **locale** : elle disparaît à la fin
# de l'appel. Une fonction peut en revanche **lire** une variable globale, mais
# il vaut mieux passer les valeurs en paramètres pour rendre le code prévisible.
# La démonstration ci-dessous provoque volontairement une `NameError`.

# %%
gain_global = 10.0  # variable définie au niveau du notebook : portée globale

def appliquer_gain(tension):  # la fonction peut lire une variable globale
    """Applique le gain global et un facteur local à une tension."""  # docstring
    gain_local = 2.0  # variable locale : elle n'existe que pendant l'appel
    return tension * gain_global * gain_local  # utilise une globale et une locale

print(appliquer_gain(1.0))  # affiche 20.0
try:  # on tente d'accéder à la variable locale depuis l'extérieur
    print(gain_local)  # provoque une NameError : elle n'existe pas ici
except NameError:  # on intercepte l'erreur pour l'expliquer
    print("gain_local n'existe pas hors de la fonction")  # message pédagogique

# %% [markdown]
# ## 5. Réutiliser une fonction : série, parallèle, trois montages
#
# `*valeurs` collecte un nombre quelconque d'arguments dans un tuple : pratique
# pour une mise en série de N résistances. La mise en parallèle utilise la
# formule du produit sur la somme. Une même fonction appelée sur trois jeux de
# composants évite trois formules recopiées, donc trois occasions de se tromper.

# %%
def resistance_serie(*valeurs):  # *valeurs accepte un nombre variable d'arguments
    """Somme les résistances passées en argument."""  # docstring
    return sum(valeurs)  # sum() additionne le tuple reçu

print(resistance_serie(100, 220))  # affiche 320, deux résistances en série
print(resistance_serie(100, 220, 470, 1000))  # affiche 1790, quatre résistances

def resistance_parallele(a, b):  # deux résistances en parallèle
    """Renvoie la résistance équivalente de a en parallèle avec b."""  # docstring
    return a * b / (a + b)  # formule du produit divisé par la somme

print(f"{resistance_parallele(1000, 1000):.1f} Ω")  # affiche 500.0 Ω
print(f"{resistance_parallele(2.2e3, 4.7e3):.1f} Ω")  # affiche 1498.6 Ω

# %% [markdown]
# Une fonction écrite une fois se vérifie une fois. On l'appelle ici sur trois
# montages : échanger R1 et R2 complète la tension d'entrée, et deux résistances
# égales donnent exactement la moitié de `U_e`.

# %%
for r1, r2 in [(10e3, 4.7e3), (4.7e3, 10e3), (1e3, 1e3)]:  # trois jeux de résistances
    print(f"R1 = {r1:.0f} Ω, R2 = {r2:.0f} Ω -> U_s = {tension_diviseur(12.0, r1, r2):.2f} V")  # même fonction, trois résultats

# %% [markdown]
# ## 6. Conversions d'unités et validation
#
# Deux fonctions inverses se vérifient l'une l'autre. Pour éviter les valeurs
# absurdes (résistance nulle, division par zéro), on **valide** les entrées et
# on lève une `ValueError` avec un message clair. L'appelant décide ensuite de
# traiter l'erreur avec `try` / `except`, plutôt que de laisser le programme
# s'arrêter brutalement.

# %%
def celsius_vers_fahrenheit(celsius):  # conversion d'unité
    """Convertit une température Celsius en Fahrenheit."""  # docstring
    return celsius * 9 / 5 + 32  # formule F = C × 9/5 + 32

def fahrenheit_vers_celsius(fahrenheit):  # conversion inverse
    """Convertit une température Fahrenheit en Celsius."""  # docstring
    return (fahrenheit - 32) * 5 / 9  # formule inverse C = (F − 32) × 5/9

print(f"{celsius_vers_fahrenheit(25.0):.1f} °F")  # affiche 77.0 °F
print(f"{fahrenheit_vers_celsius(77.0):.1f} °C")  # affiche 25.0 °C

# %% [markdown]
# La même validation s'applique au diviseur : une résistance nulle donnerait une
# division par zéro ou une sortie absurde. Mieux vaut refuser l'appel avec un
# message explicite, que l'appelant peut afficher dans un journal de bord.

# %%
def tension_diviseur_verifie(u_entree, r1, r2):  # variante avec contrôle des entrées
    """Comme tension_diviseur, mais refuse les résistances nulles ou négatives."""  # docstring
    if r1 <= 0 or r2 <= 0:  # une résistance doit être strictement positive
        raise ValueError("les résistances doivent être > 0 ohm")  # lève une exception explicite
    return u_entree * r2 / (r1 + r2)  # calcul identique à la version simple

print(f"{tension_diviseur_verifie(12.0, 10e3, 4.7e3):.3f} V")  # cas valide : 3.837 V
try:  # on tente un appel invalide
    tension_diviseur_verifie(12.0, 0, 4700)  # r1 nul : la fonction refuse de calculer
except ValueError as erreur:  # on capture l'exception et son message
    print(f"Erreur : {erreur}")  # affiche le message levé par la fonction

# %% [markdown]
# ## 7. Modules `math` et fonctions `lambda`
#
# `import math` charge un module de la bibliothèque standard et ses fonctions :
# `math.sqrt`, `math.isclose` (comparaison tolérante des flottants) ou `pi`.
# `from math import pi` importe seulement un nom ; `from math import *` est à
# éviter car il pollue l'espace de noms et masque vos propres fonctions.
#
# Une `lambda` est une fonction anonyme limitée à **une expression**. Elle est
# concise pour une conversion triviale, mais illisible dès qu'il y a plusieurs
# étapes : dans ce cas, préférez `def` avec une docstring.

# %%
import math  # importe le module de mathématiques de la bibliothèque standard

print(f"{math.sqrt(2):.4f}")  # racine carrée, arrondie à 4 décimales : 1.4142
print(math.isclose(0.1 + 0.2, 0.3))  # comparaison tolérante des flottants : True
print(math.isclose(1.0, 1.001, rel_tol=1e-2))  # tolérance relative de 1 % : True

from math import pi  # importe uniquement le nom pi
print(f"pi = {pi:.5f}")  # affiche 3.14159

# %% [markdown]
# Comparons les deux écritures d'une même conversion : la `lambda` tient en une
# ligne, mais ne peut porter ni docstring, ni message d'erreur, ni plusieurs
# étapes. Pour du code partagé, `def` reste le choix lisible.

# %%
mv_vers_v = lambda millivolts: millivolts / 1000  # lambda : une seule expression
print(mv_vers_v(3300))  # affiche 3.3

# %% [markdown]
# ## À vous de jouer
#
# Complétez les cellules suivantes. Les `TODO` sont des commentaires, donc les
# cellules s'exécutent sans erreur avant même d'être complétées.

# %% [markdown]
# **Exercice 1** — Écrivez une fonction `puissance(u, i)` qui renvoie `u * i`,
# puis appelez-la pour 5 V et 0,02 A et affichez le résultat en milliwatts.

# %%
# TODO : définissez def puissance(u, i) avec une docstring et un return

# TODO : calculez p = puissance(5.0, 0.02)

# TODO : affichez p * 1000 en mW avec 1 décimale

# %% [markdown]
# **Exercice 2** — Écrivez `moyenne_valeurs(valeurs)` qui renvoie la moyenne
# d'une liste, puis testez-la sur `[1.0, 2.0, 3.0, 4.0]`.

# %%
# TODO : définissez la fonction avec return sum(valeurs) / len(valeurs)

# TODO : affichez moyenne_valeurs([1.0, 2.0, 3.0, 4.0])

# %% [markdown]
# **Exercice 3** — Écrivez `tension_adc(code, vref=5.0)` qui lève une
# `ValueError` si `code` est négatif, puis appelez-la dans un `try` / `except`.

# %%
# TODO : définissez la fonction avec un if code < 0 : raise ValueError(...)

# TODO : appelez-la avec -1 dans un try / except ValueError

# %% [markdown]
# ## À retenir
#
# - `def nom(a, b=defaut):` définit une fonction ; `return` renvoie le résultat.
# - Une docstring d'une ligne documente la fonction : écrivez-la toujours.
# - `return a, b` renvoie un tuple, déballable en `x, y = f()`.
# - Une variable créée dans une fonction est locale et disparaît après l'appel.
# - `*valeurs` accepte un nombre variable d'arguments ; `sum()` additionne un tuple.
# - `raise ValueError(...)` valide une entrée ; `try` / `except` la traite.
# - `math.sqrt`, `math.isclose` et `math.pi` évitent de réinventer la roue.
#
# **Notebook suivant** : `05 — Structures de données`, pour stocker un lot de
# mesures, une nomenclature de composants et une carte de registres.
