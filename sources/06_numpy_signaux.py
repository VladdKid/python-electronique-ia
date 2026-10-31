# %% [markdown]
# # 06 — NumPy : calculer sur des signaux
#
# En électronique, on ne manipule jamais **une** mesure isolée : on traite des
# séries de milliers d'échantillons (oscilloscope, carte d'acquisition, capteur
# I2C). NumPy est la bibliothèque qui rend ces calculs rapides et lisibles.
#
# **Objectifs du notebook**
#
# - créer et inspecter des tableaux NumPy ;
# - faire des calculs vectorisés (une opération sur tout le tableau) ;
# - extraire des échantillons par indice ou par condition ;
# - calculer des grandeurs utiles : moyenne, écart-type, valeur efficace (RMS),
#   valeur crête-à-crête ;
# - générer un signal sinusoïdal bruité, comme le ferait un capteur réel.
#
# **Prérequis** : notebooks 01 à 05 (variables, boucles, fonctions, listes).

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
# ## 1. Pourquoi NumPy plutôt que des listes ?
#
# Une liste Python stocke des objets hétérogènes et une boucle `for` est lente.
# Un tableau NumPy (`ndarray`) stocke des nombres du même type dans un bloc de
# mémoire compact : les opérations sont alors écrites sans boucle et exécutées
# en code compilé. On appelle cela la **vectorisation**.

# %%
import numpy as np  # importe NumPy sous le nom court np (convention universelle)

print(np.__version__)  # affiche la version installée, utile pour le dépannage

# %% [markdown]
# ## 2. Créer un tableau et l'inspecter
#
# `np.array` transforme une liste en tableau. Les attributs `dtype`, `shape`,
# `size` et `ndim` décrivent le contenu : type des nombres, dimensions, nombre
# total d'éléments et nombre d'axes.

# %%
lectures_mA = np.array([12.1, 11.8, 12.4, 60.2, 12.0])  # 5 mesures de courant en milliampères
print(lectures_mA)  # affiche le tableau complet
print(lectures_mA.dtype)  # float64 : nombres à virgule sur 64 bits
print(lectures_mA.shape)  # (5,) : une seule dimension de 5 éléments
print(lectures_mA.size)  # 5 : nombre total d'éléments
print(lectures_mA.ndim)  # 1 : le tableau a un seul axe

# %%
entiers = np.array([1, 2, 3])  # sans virgule, NumPy choisit le type entier
print(entiers.dtype)  # int64 sur cette machine
print(entiers + 0.5)  # une opération avec un float convertit le résultat en float

# %% [markdown]
# Deux fonctions créent des tableaux réguliers :
#
# - `np.arange(début, fin, pas)` : comme `range`, mais la borne de fin est exclue ;
# - `np.linspace(début, fin, nombre)` : répartit exactement `nombre` points, fin incluse.
#
# C'est exactement ce qu'il faut pour construire un axe de temps
# d'échantillonnage.

# %%
fs = 1000.0  # fréquence d'échantillonnage : 1000 échantillons par seconde
duree = 0.02  # on observe 20 ms de signal
t_arange = np.arange(0.0, duree, 1.0 / fs)  # pas de 1 ms, borne 20 ms exclue
t_linspace = np.linspace(0.0, duree, 21)  # 21 points de 0 à 20 ms inclus
print(len(t_arange), len(t_linspace))  # 20 échantillons contre 21 selon la méthode
print(t_arange[:3])  # les 3 premiers instants, en secondes

# %%
zeros = np.zeros(4)  # quatre zéros : pratique pour réserver un buffer
plein = np.full(4, 3.3)  # quatre copies de 3.3 V : une alimentation constante
print(zeros)  # affiche [0. 0. 0. 0.]
print(plein)  # affiche [3.3 3.3 3.3 3.3]

# %% [markdown]
# ## 3. Calcul vectorisé : la loi d'Ohm sur un lot de résistances
#
# Au lieu d'écrire une boucle, on applique l'opération directement sur les
# tableaux. NumPy parcourt les éléments en interne, beaucoup plus vite.

# %%
resistances = np.array([100.0, 220.0, 330.0, 1000.0, 4700.0])  # cinq valeurs normalisées en ohms
courant = np.full(resistances.shape, 0.01)  # 10 mA dans chaque résistance (même forme)
tensions = resistances * courant  # U = R × I, appliqué aux 5 composants d'un coup
print(tensions)  # affiche [1.  2.2 3.3 10. 47. ]

# %%
puissances_mW = tensions * courant * 1000.0  # P = U × I, convertie en milliwatts
print(np.round(puissances_mW, 2))  # arrondit l'affichage à 2 décimales

# %% [markdown]
# Les opérations classiques (`+ - * / **`) s'appliquent élément par élément.
# Une division par zéro produit `inf` et un avertissement, sans arrêter le
# programme : vérifiez toujours vos dénominateurs.

# %%
tensions2 = tensions ** 2  # élève au carré chaque tension (utile pour la puissance)
print(np.sqrt(tensions2))  # racine carrée : on retrouve les tensions d'origine

# %% [markdown]
# ## 4. Extraire des données : indices, tranches et masques
#
# Un **masque** est un tableau de booléens de la même taille que les données.
# Il permet de sélectionner tous les échantillons qui respectent une condition,
# par exemple un courant anormal.

# %%
print(lectures_mA[0])  # premier échantillon (l'indice commence à 0)
print(lectures_mA[-1])  # dernier échantillon : l'indice -1 part de la fin
print(lectures_mA[1:4])  # tranche : indices 1, 2 et 3 ; la borne 4 est exclue
print(lectures_mA[::2])  # un échantillon sur deux (sous-échantillonnage)

# %%
seuil_mA = 20.0  # seuil d'alarme choisi pour l'exemple
masque = lectures_mA > seuil_mA  # tableau de booléens : True là où le seuil est dépassé
print(masque)  # [False False False  True False]
print(lectures_mA[masque])  # ne garde que les mesures en alarme
print(np.count_nonzero(masque))  # compte les True : nombre de dépassements

# %%
etiquettes = np.where(masque, "ALARME", "OK")  # choisit un texte selon le masque
print(etiquettes)  # affiche une étiquette pour chaque mesure

# %% [markdown]
# ## 5. Statistiques de base
#
# Moyenne, écart-type, minimum, maximum et RMS résument un signal. Le **RMS**
# (valeur efficace) d'une sinusoïde d'amplitude A vaut A/√2 : c'est la valeur
# que mesurerait un multimètre en mode alternatif.

# %%
print(np.mean(lectures_mA))  # moyenne : influencée par le pic à 60 mA
print(np.median(lectures_mA))  # médiane : robuste aux valeurs aberrantes
print(np.std(lectures_mA))  # écart-type : dispersion autour de la moyenne
print(np.min(lectures_mA), np.max(lectures_mA))  # valeurs extrêmes
print(np.argmax(lectures_mA))  # indice du maximum (ici 3)

# %%
signal = np.array([0.0, 1.0, 0.0, -1.0])  # un signal carré simplifié
rms = np.sqrt(np.mean(signal**2))  # RMS = racine de la moyenne des carrés
crete_a_crete = np.max(signal) - np.min(signal)  # amplitude totale parcourue
print(f"RMS = {rms:.3f}, crête-à-crête = {crete_a_crete:.3f}")  # affiche les deux indicateurs

# %% [markdown]
# ## 6. Générer un signal sinusoïdal bruité
#
# Un convertisseur analogique-numérique échantillonne une tension à intervalle
# régulier. On modélise ici un signal de 50 Hz (réseau électrique) auquel
# s'ajoute un bruit gaussien, comme sur une vraie acquisition.

# %%
f_signal = 50.0  # fréquence du signal utile : 50 Hz
amplitude = 1.0  # amplitude en volts
fs_signal = 1000.0  # 1000 échantillons par seconde
t = np.arange(0.0, 0.05, 1.0 / fs_signal)  # 50 points sur 50 ms, soit 2,5 périodes
signal_propre = amplitude * np.sin(2.0 * np.pi * f_signal * t)  # sinusoïde : A·sin(2πft)
print(signal_propre[:5])  # premiers échantillons

# %%
rng = np.random.default_rng(42)  # générateur aléatoire reproductible (graine 42)
bruit = rng.normal(loc=0.0, scale=0.2, size=t.size)  # bruit gaussien : moyenne 0, écart-type 0.2
signal_bruite = signal_propre + bruit  # le capteur ajoute du bruit au signal
print(signal_bruite[:5])  # les valeurs sont maintenant moins régulières

# %%
puissance_utile = np.mean(signal_propre**2)  # puissance moyenne du signal propre
puissance_bruit = np.mean(bruit**2)  # puissance moyenne du bruit
snr_db = 10.0 * np.log10(puissance_utile / puissance_bruit)  # rapport signal/bruit en décibels
print(f"SNR ≈ {snr_db:.1f} dB")  # environ 14 dB : bruit bien visible

# %%
rms_propre = np.sqrt(np.mean(signal_propre**2))  # valeur efficace théorique ≈ 0.707 V
rms_bruite = np.sqrt(np.mean(signal_bruite**2))  # valeur efficace mesurée, gonflée par le bruit
print(f"RMS propre = {rms_propre:.3f} V, RMS bruité = {rms_bruite:.3f} V")  # compare les deux

# %% [markdown]
# On retrouve une propriété classique : le RMS d'une sinusoïde d'amplitude 1
# vaut 1/√2 ≈ 0,707. Le bruit augmente légèrement le RMS mesuré.

# %% [markdown]
# ## 7. Tableaux 2D : plusieurs acquisitions
#
# Une carte d'acquisition enregistre souvent plusieurs mesures identiques.
# On les range dans un tableau à deux dimensions : une ligne par acquisition,
# une colonne par instant. Les réductions avec `axis` indiquent sur quel axe
# on calcule.

# %%
nb_acquisitions = 8  # huit répétitions de la même mesure
bruit_2d = rng.normal(0.0, 0.2, size=(nb_acquisitions, t.size))  # une ligne de bruit par acquisition
mesures = signal_propre + bruit_2d  # diffusion : la sinusoïde est ajoutée à chaque ligne
print(mesures.shape)  # (8, 50) : 8 acquisitions de 50 échantillons

# %%
moyenne_par_acquisition = mesures.mean(axis=1)  # moyenne sur les colonnes : 8 valeurs
ecart_type_par_instant = mesures.std(axis=0)  # écart-type sur les lignes : 50 valeurs
print(np.round(moyenne_par_acquisition, 3))  # moyenne de chaque acquisition ≈ 0
print(np.round(ecart_type_par_instant[:5], 3))  # dispersion à chaque instant ≈ 0.2

# %%
moyenne_globale = mesures.mean()  # sans axis, on moyenne tous les éléments
print(f"Moyenne globale : {moyenne_globale:.4f} V")  # proche de 0, car la sinusoïde est centrée

# %% [markdown]
# **Point méthode** : `axis=0` réduit les lignes (on obtient une valeur par
# colonne), `axis=1` réduit les colonnes (une valeur par ligne).

# %% [markdown]
# ## 8. À vous de jouer
#
# **Exercice 1** — Une alimentation de 3,3 V débite dans les résistances
# `[100, 220, 330, 1000]` Ω. Calculez le courant de chaque branche avec un
# tableau NumPy et affichez-le en mA.

# %%
# TODO : créez resistances_exo avec np.array([100.0, 220.0, 330.0, 1000.0])

# TODO : créez courants_exo = 3.3 / resistances_exo

# TODO : affichez np.round(courants_exo * 1000, 1) avec un message clair

# %% [markdown]
# **Exercice 2** — Un relevé de températures contient
# `[21.5, 22.1, 24.8, 61.2, 23.0, 22.7, 60.9]` °C.
# Comptez les valeurs supérieures à 60 °C et affichez-les.

# %%
# TODO : créez temperatures avec np.array([...])

# TODO : construisez le masque temperatures > 60

# TODO : affichez les valeurs concernées et leur nombre

# %% [markdown]
# **Exercice 3** — Générez 200 échantillons d'une sinusoïde de 100 Hz,
# d'amplitude 2, échantillonnée à 2000 Hz, puis calculez son RMS.
# Le résultat théorique est 2/√2 ≈ 1,414.

# %%
# TODO : créez l'axe de temps avec np.arange et fs = 2000.0 sur 0,1 s

# TODO : calculez le signal avec np.sin

# TODO : affichez le RMS calculé et la valeur théorique 2 / np.sqrt(2)

# %% [markdown]
# ## 9. À retenir
#
# - `np.array` crée un tableau ; `shape`, `dtype`, `size` le décrivent.
# - Les opérations sont vectorisées : pas de boucle pour appliquer une formule.
# - Un masque booléen sélectionne des échantillons : `signal[signal > seuil]`.
# - `mean`, `std`, `min`, `max`, `argmax` résument un signal ; RMS = √(moyenne des carrés).
# - `np.arange` exclut la borne, `np.linspace` l'inclut.
# - `axis=0` réduit les lignes, `axis=1` réduit les colonnes.
#
# **Notebook suivant** : `07 — Tracer et analyser des signaux avec Matplotlib`.
