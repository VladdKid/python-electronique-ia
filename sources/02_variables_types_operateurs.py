# %% [markdown]
# # 02 — Variables, types et opérateurs
#
# Une mesure n'a de sens que si l'on connaît son **type** (entier, flottant,
# texte, booléen) et son **unité**. En électronique, une erreur d'unité entre
# millivolts et volts suffit à fausser tout un calcul. Ce notebook apprend à
# stocker, convertir et combiner des grandeurs, puis à calculer des impédances
# complexes comme le ferait un analyseur d'impédance.
#
# **Objectifs du notebook**
#
# - identifier les types `int`, `float`, `str`, `bool` et `complex` ;
# - convertir des unités (mV → V, kΩ → Ω, mA → A) et des types ;
# - utiliser `+ - * / // % **` et les comparaisons ;
# - écrire une impédance `Z = R + jX` et calculer son module et sa phase ;
# - formater un résultat avec une f-string (`:.2f`, `:.3f`, `:.2e`).
#
# **Prérequis** : avoir exécuté le notebook `01` (variables, `print`, f-strings).

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
# ## 1. Les quatre types de base (et le type complexe)
#
# Chaque valeur possède un type. `int` sert aux comptages (nombre de voies,
# code de conversion), `float` aux grandeurs physiques continues (tension,
# courant), `str` aux références de composants, `bool` aux états logiques.
# `type()` affiche le type d'une valeur : c'est le premier réflexe de débogage.

# %%
nombre_voies = 8  # entier (int) : nombre de voies du multiplexeur analogique
tension_ref = 3.3  # flottant (float) : tension de référence du convertisseur, en volts
reference_can = "ADS1115"  # chaîne (str) : référence du circuit intégré
can_alimente = True  # booléen (bool) : le convertisseur est sous tension

print(type(nombre_voies))  # affiche <class 'int'>
print(type(tension_ref))  # affiche <class 'float'>
print(type(reference_can))  # affiche <class 'str'>
print(type(can_alimente))  # affiche <class 'bool'>
print(type(1j))  # affiche <class 'complex'> : le type des nombres complexes

# %% [markdown]
# ## 2. Convertir les unités et les types
#
# La division `/` renvoie toujours un flottant, même entre deux entiers. Pour
# passer des millivolts aux volts ou des milliampères aux ampères, on divise
# par 1000. `float()` convertit un texte en nombre, `str()` fait l'inverse,
# `int()` tronque la partie décimale (attention : ce n'est pas un arrondi).

# %%
mesure_mv = 1250  # mesure lue sur le multimètre : 1250 mV
tension_v = mesure_mv / 1000  # convertit les millivolts en volts (1,25 V)
print(f"{mesure_mv} mV = {tension_v} V")  # affiche 1250 mV = 1.25 V

resistance_ko = "4.7"  # valeur récupérée dans un fichier texte : c'est une chaîne
resistance_ohm = float(resistance_ko) * 1000  # float() convertit le texte, puis kΩ -> Ω
print(f"{resistance_ko} kΩ = {resistance_ohm:.0f} Ω")  # affiche 4.7 kΩ = 4700 Ω

courant_ma = 15  # courant en milliampères, sous forme entière
courant_a = courant_ma / 1000  # mA -> A : la division produit un flottant
print(courant_a)  # affiche 0.015

code_adc = 700  # code entier renvoyé par le convertisseur
code_texte = str(code_adc)  # str() transforme le nombre en texte
print("code = " + code_texte)  # concatène deux chaînes avec l'opérateur +

# %% [markdown]
# `/` et `float()` conservent les décimales ; `int()` les coupe. Sur un
# convertisseur, cette troncature correspond exactement à la quantification :
# tout ce qui dépasse le dernier pas entier est perdu.

# %%
print(int(3.99))  # int() supprime la partie décimale : affiche 3, ce n'est pas un arrondi

# %% [markdown]
# ## 3. Opérateurs arithmétiques : puissance et tolérance
#
# Les opérateurs `+ - * /` se comportent comme en mathématiques. `**` élève à
# une puissance, ce qui permet d'écrire `U² / R` sans variable intermédiaire.
# Une tolérance de fabricant se traduit en deux bornes : valeur nominale ×
# (1 − tol) et valeur nominale × (1 + tol).

# %%
u_alim = 12.0  # tension d'alimentation en volts
i_led = 0.020  # courant dans la LED en ampères (20 mA)
p_led = u_alim * i_led  # puissance P = U × I, en watts
print(f"P = {p_led:.3f} W")  # affiche P = 0.240 W

r_chute = 2.0 / 0.020  # résistance de chute pour perdre 2 V à 20 mA : R = U / I
print(f"R = {r_chute:.1f} Ω")  # affiche R = 100.0 Ω

valeur_nominale = 1000.0  # résistance nominale en ohms
tolerance = 5 / 100  # tolérance de 5 % exprimée en fraction
valeur_min = valeur_nominale * (1 - tolerance)  # borne basse : 950 Ω
valeur_max = valeur_nominale * (1 + tolerance)  # borne haute : 1050 Ω
print(f"{valeur_min:.0f} Ω à {valeur_max:.0f} Ω")  # affiche 950 Ω à 1050 Ω

p_max = u_alim ** 2 / r_chute  # opérateur ** : U² / R = 1,44 W dissipé dans la résistance
print(f"P max = {p_max:.3f} W")  # affiche 1.440 W : il faut une résistance d'au moins 2 W

# %% [markdown]
# ## 4. Division entière `//` et modulo `%`
#
# Un convertisseur analogique-numérique découpe sa plage de mesure en pas
# égaux. `//` donne le nombre de pas entiers contenus dans une tension (donc
# le code du CAN) et `%` donne le reste, c'est-à-dire l'erreur de quantification.
# Ces deux opérateurs sont la base du calcul de résolution d'un CAN.

# %%
vref = 5.0  # tension de référence du CAN, en volts
nb_bits = 10  # résolution du convertisseur : 10 bits
nb_pas = 2 ** nb_bits  # 2^10 = 1024 pas possibles
pas = vref / nb_pas  # quantum : 5 / 1024 ≈ 4,88 mV par pas
print(f"Quantum : {pas * 1000:.3f} mV")  # affiche 4.883 mV

tension_mesuree = 3.72  # tension présente à l'entrée du CAN
code = int(tension_mesuree // pas)  # // : nombre entier de pas contenus dans la tension
reste = tension_mesuree % pas  # % : écart entre la tension réelle et ce code
print(f"Code = {code}, reste = {reste * 1000:.2f} mV")  # affiche 761 et environ 4.18 mV

# %% [markdown]
# ## 5. Nombres complexes et impédances
#
# En régime sinusoïdal, une impédance s'écrit `Z = R + jX`. Python utilise la
# lettre `j` (et non `i`, réservé à d'autres usages) : on écrit `1j` pour
# l'unité imaginaire. `abs(Z)` donne le module en ohms, `cmath.phase(Z)` donne
# l'argument en radians, que l'on convertit souvent en degrés.
#
# `cmath` est le module de la bibliothèque standard pour les complexes ; on
# l'importe une seule fois par noyau.

# %%
import cmath  # module standard pour les fonctions complexes (phase, module)

resistance = 100.0  # partie réelle : résistance série, en ohms
reactance = -50.0  # partie imaginaire : réactance du condensateur, en ohms
impedance = resistance + 1j * reactance  # Python note j (et non i) pour l'unité imaginaire
print(impedance)  # affiche (100-50j), c'est-à-dire R + jX

module_z = abs(impedance)  # module |Z| = racine(R² + X²), en ohms
phase_z = cmath.phase(impedance)  # argument de Z en radians (négatif : circuit capacitif)
print(f"|Z| = {module_z:.2f} Ω, phase = {phase_z:.3f} rad")  # affiche 111.80 Ω et -0.464 rad

phase_deg = phase_z * 180 / cmath.pi  # conversion radians -> degrés
print(f"Phase = {phase_deg:.1f} °")  # affiche -26.6 ° : le courant est en avance sur la tension

# %% [markdown]
# ## 6. Comparaisons et opérateurs logiques
#
# Une surveillance de batterie compare la tension à des seuils. Les opérateurs
# `== != < > <= >=` renvoient un booléen, que l'on combine avec `and`, `or` et
# `not`. Attention : `==` teste l'égalité, `=` affecte une valeur. Et comparer
# deux flottants avec `==` est risqué à cause des arrondis binaires.

# %%
tension_batterie = 11.8  # tension mesurée aux bornes de la batterie, en volts
seuil_critique = 11.0  # seuil de décharge profonde
seuil_bas = 12.0  # seuil de batterie faible
batterie_faible = tension_batterie < seuil_bas  # True : 11,8 est inférieur à 12,0
batterie_ok = tension_batterie >= seuil_bas  # False : comparaison inverse
print(batterie_faible, batterie_ok)  # affiche True False

courant_charge = 0.85  # courant débité par la batterie, en ampères
temperature = 28.0  # température du boîtier, en degrés Celsius
alarme = batterie_faible and courant_charge > 0.5  # and : les deux conditions doivent être vraies
securite = (temperature > 60.0) or (courant_charge > 2.0)  # or : une seule condition suffit
autorisation = not securite  # not inverse un booléen
print(alarme, securite, autorisation)  # affiche True False True

# %% [markdown]
# Ces quatre tests résument les pièges classiques : `!=` teste la différence, un
# texte n'est jamais égal à un nombre, et deux flottants « égaux » à l'écrit ne
# le sont pas en binaire. Pour comparer deux tensions, utilisez une tolérance
# (`math.isclose`, notebook 04) plutôt que `==`.

# %%
print(reference_can != "ADS1115")  # != teste la différence : affiche False
print(1.0 == 1)  # True : Python compare les valeurs numériques, pas les écritures
print("12" == 12)  # False : un texte n'est jamais égal à un nombre
print(0.1 + 0.2 == 0.3)  # False : arrondi binaire des flottants, jamais fiable

# %% [markdown]
# ## 7. Affectation augmentée
#
# `x += 1` est l'équivalent court de `x = x + 1`. On l'utilise surtout pour
# accumuler des mesures, par exemple additionner les courants de plusieurs
# branches d'un nœud sans écrire de longues expressions. Les variantes `-=`,
# `*=` et `/=` fonctionnent de la même façon.

# %%
courant_total = 0.0  # accumulateur initialisé à zéro ampère
courant_total += 0.012  # ajoute 12 mA : équivaut à courant_total = courant_total + 0.012
courant_total += 0.008  # ajoute 8 mA
courant_total += 0.005  # ajoute 5 mA
print(f"Courant total : {courant_total * 1000:.1f} mA")  # affiche 25.0 mA

# %% [markdown]
# ## 8. Formater les nombres avec les f-strings
#
# `:.2f` fixe le nombre de décimales, `:.3f` en fixe trois, `:.2e` passe en
# notation scientifique — indispensable pour les courants de fuite ou les
# capacités. Le format ne modifie pas la variable : il ne change que l'affichage.

# %%
tension_mesuree = 1.234567  # tension mesurée, en volts
print(f"{tension_mesuree:.2f} V")  # 2 décimales : affiche 1.23 V
print(f"{tension_mesuree:.3f} V")  # 3 décimales : affiche 1.235 V
print(f"{tension_mesuree:.2e} V")  # notation scientifique : affiche 1.23e+00 V
courant_fuite = 0.00042  # courant de fuite, en ampères
print(f"{courant_fuite:.2e} A = {courant_fuite * 1e6:.0f} µA")  # affiche 4.20e-04 A = 420 µA

# %% [markdown]
# ## À vous de jouer
#
# Complétez les cellules suivantes : remplacez chaque `TODO` par le code
# demandé. Les cellules s'exécutent déjà sans erreur, car les `TODO` sont de
# simples commentaires.

# %% [markdown]
# **Exercice 1** — Une résistance de 4,7 kΩ est alimentée sous 12 V. Convertissez
# sa valeur en ohms, calculez le courant en ampères, puis affichez-le en mA
# avec une décimale.

# %%
# TODO : créez resistance_ohm = 4.7 * 1000

# TODO : créez courant = 12.0 / resistance_ohm

# TODO : affichez f"Courant : {courant * 1000:.1f} mA"

# %% [markdown]
# **Exercice 2** — Une bobine a une résistance de 220 Ω et une réactance de
# +150 Ω. Construisez `Z = 220 + 1j * 150`, affichez son module à 2 décimales
# puis sa phase en degrés à 1 décimale (pensez à `cmath.phase`).

# %%
# TODO : importez cmath

# TODO : créez impedance = 220 + 1j * 150

# TODO : affichez abs(impedance) avec 2 décimales, en ohms

# TODO : affichez cmath.phase(impedance) * 180 / cmath.pi avec 1 décimale

# %% [markdown]
# **Exercice 3** — Un condensateur est donné pour 100 nF avec une tolérance de
# 10 %. Calculez et affichez les bornes min et max en nanofarads, en notation
# scientifique (`:.2e`).

# %%
# TODO : créez valeur_nominale = 100e-9 (farads) et tolerance = 0.10

# TODO : calculez valeur_min et valeur_max

# TODO : affichez les deux bornes avec :.2e

# %% [markdown]
# ## À retenir
#
# - `type()` identifie la nature d'une valeur ; `int`, `float`, `str`, `bool`, `complex`.
# - `/` renvoie toujours un flottant ; `//` donne le quotient entier, `%` le reste.
# - `**` élève à une puissance ; une tolérance se traduit en deux bornes.
# - En électronique, on écrit `Z = R + 1j * X` : Python utilise `j`, pas `i`.
# - `abs(Z)` donne le module, `cmath.phase(Z)` la phase en radians.
# - `:.2f`, `:.3f` et `:.2e` contrôlent l'affichage sans modifier la valeur.
#
# **Notebook suivant** : `03 — Conditions et boucles`, où ces comparaisons
# deviendront des décisions et ces accumulateurs des boucles.
