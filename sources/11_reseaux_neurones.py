# %% [markdown]
# # 11 — Réseaux de neurones : du neurone à PyTorch
#
# Un réseau de neurones n'est pas magique : c'est une fonction mathématique
# dont on ajuste les paramètres pour minimiser une erreur. Dans ce notebook,
# nous construisons d'abord un neurone **à la main avec NumPy**, puis nous
# laissons PyTorch faire le même travail sur des données de maintenance.
#
# **Objectifs du notebook**
#
# - comprendre le neurone artificiel : `sortie = activation(w·x + b)` ;
# - implémenter la descente de gradient sur un modèle linéaire ;
# - résoudre le problème XOR avec un réseau à une couche cachée, codé à la main ;
# - écrire le même réseau avec PyTorch ;
# - entraîner, évaluer et sauvegarder un classifieur de défauts moteur.
#
# **Prérequis** : notebooks 06 (NumPy), 08 (Pandas), 10 (scikit-learn).

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
# ## 1. Le neurone artificiel
#
# Un neurone calcule une somme pondérée de ses entrées, ajoute un biais, puis
# applique une **fonction d'activation**. La sigmoïde `σ(z) = 1 / (1 + e^-z)`
# écrase le résultat entre 0 et 1 : on peut l'interpréter comme une
# probabilité. `w` sont les poids (l'importance de chaque entrée) et `b` le
# biais (le décalage).

# %%
import numpy as np  # calculs numériques

# %%
def sigmoide(z):  # définit l'activation sigmoïde
    return 1.0 / (1.0 + np.exp(-z))  # renvoie une valeur entre 0 et 1

poids = np.array([0.8, -0.4])  # deux poids : un positif, un négatif
biais = 0.1  # décalage du neurone
entree = np.array([1.0, 0.5])  # une mesure à deux capteurs
z = np.dot(poids, entree) + biais  # somme pondérée w·x + b
sortie = sigmoide(z)  # activation : probabilité entre 0 et 1
print(f"z = {z:.3f}, sortie = {sortie:.3f}")  # affiche la valeur avant et après activation

# %% [markdown]
# Entraîner un neurone signifie **trouver les poids et le biais** qui donnent
# les bonnes réponses sur des exemples. On mesure l'écart avec une fonction de
# perte, puis on modifie les paramètres dans la direction qui réduit cette
# perte : c'est la **descente de gradient**.

# %% [markdown]
# ## 2. Descente de gradient à la main
#
# Un CAN (convertisseur analogique-numérique) 10 bits convertit une tension de
# 0 à 5 V en un code de 0 à 1023. La relation est normalement linéaire :
# `code ≈ pente × (tension / 5) + offset`. On part de paramètres faux (0) et on
# les corrige pas à pas. La normalisation de la tension entre 0 et 1 évite des
# gradients énormes et rend l'apprentissage stable.

# %%
rng = np.random.default_rng(42)  # graine fixe : résultats reproductibles
tension = np.linspace(0.0, 5.0, 80)  # 80 tensions de test réparties de 0 à 5 V
code_mesure = 204.6 * tension + 3.0 + rng.normal(0.0, 2.0, tension.size)  # réponse idéale + bruit de ±2 codes
x_norm = tension / 5.0  # normalise l'entrée dans l'intervalle [0, 1]
print(code_mesure[:5].round(1))  # aperçu des premiers codes mesurés

# %%
pente = 0.0  # paramètre initial : on ne sait rien, on part de zéro
offset = 0.0  # biais initial
taux_apprentissage = 0.5  # pas de la descente de gradient : ni trop grand, ni trop petit
historique_perte = []  # mémorise la perte pour l'observer ensuite

for iteration in range(2000):  # 2000 mises à jour des paramètres
    prediction = pente * x_norm + offset  # modèle actuel : droite pente × x + offset
    erreur = prediction - code_mesure  # écart entre prédiction et mesure réelle
    perte = np.mean(erreur**2)  # erreur quadratique moyenne (MSE)
    gradient_pente = 2.0 * np.mean(erreur * x_norm)  # dérivée de la MSE par rapport à la pente
    gradient_offset = 2.0 * np.mean(erreur)  # dérivée de la MSE par rapport à l'offset
    pente -= taux_apprentissage * gradient_pente  # descend le long du gradient pour la pente
    offset -= taux_apprentissage * gradient_offset  # descend le long du gradient pour l'offset
    historique_perte.append(perte)  # enregistre la perte de cette itération
    if iteration % 400 == 0:  # affiche seulement toutes les 400 itérations
        print(f"itération {iteration:4d} | perte {perte:9.1f} | pente {pente:7.1f} | offset {offset:6.1f}")  # suivi de l'apprentissage

# %%
print(f"Pente apprise : {pente:.1f} codes (théorique ≈ 1023)")  # une pente de 1023 couvre 0..1023 sur x ∈ [0, 1]
print(f"Offset appris : {offset:.1f} codes (théorique ≈ 3)")  # offset attendu proche de 3
print(f"Perte finale : {historique_perte[-1]:.1f}")  # proche de la variance du bruit (≈ 4)

# %%
rmse_codes = np.sqrt(historique_perte[-1])  # racine de la MSE : erreur en codes
rmse_volts = rmse_codes * 5.0 / 1023.0  # conversion en volts grâce à la résolution du CAN
print(f"Erreur typique : {rmse_codes:.1f} codes, soit {rmse_volts * 1000:.2f} mV")  # ≈ 2 codes, soit ≈ 10 mV

# %%
tension_test = 2.5  # tension inconnue à mesurer
code_predit = pente * (tension_test / 5.0) + offset  # applique le modèle appris
print(f"À {tension_test} V, le CAN devrait lire ≈ {code_predit:.0f}")  # ≈ 514 codes

# %% [markdown]
# **Ce qu'il faut retenir de la boucle** : prédire, mesurer l'erreur, calculer
# le gradient, corriger les paramètres. PyTorch automatise exactement ces
# quatre étapes, mais le principe reste identique.

# %% [markdown]
# ## 3. Pourquoi un neurone ne suffit pas : le XOR
#
# Un neurone seul ne trace qu'une frontière linéaire. Le XOR (ou exclusif) est
# le contre-exemple classique : aucune droite ne sépare les sorties 0 et 1.
# Il faut une **couche cachée**, c'est-à-dire plusieurs neurones dont les
# sorties alimentent un neurone final.

# %%
X_xor = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])  # les 4 combinaisons d'entrées
y_xor = np.array([[0.0], [1.0], [1.0], [0.0]])  # sortie attendue : 1 si les entrées diffèrent
print(X_xor.shape, y_xor.shape)  # 4 exemples, 2 entrées, 1 sortie

# %%
rng_xor = np.random.default_rng(42)  # générateur pour l'initialisation aléatoire
W1 = rng_xor.normal(0.0, 1.0, (2, 4))  # couche cachée : 2 entrées → 4 neurones
b1 = np.zeros((1, 4))  # un biais par neurone caché
W2 = rng_xor.normal(0.0, 1.0, (4, 1))  # couche de sortie : 4 neurones → 1 sortie
b2 = np.zeros((1, 1))  # biais de la sortie
taux_xor = 0.5  # pas d'apprentissage du réseau XOR

# %%
for epoque in range(10000):  # 10 000 passages sur les 4 exemples
    a1 = sigmoide(X_xor @ W1 + b1)  # couche cachée : activation des 4 neurones
    a2 = sigmoide(a1 @ W2 + b2)  # couche de sortie : probabilité finale
    delta2 = (a2 - y_xor) * a2 * (1.0 - a2)  # erreur de sortie pondérée par la dérivée de la sigmoïde
    dW2 = a1.T @ delta2  # gradient des poids de sortie
    db2 = delta2.sum(axis=0, keepdims=True)  # gradient du biais de sortie
    delta1 = (delta2 @ W2.T) * a1 * (1.0 - a1)  # rétropropagation de l'erreur vers la couche cachée
    dW1 = X_xor.T @ delta1  # gradient des poids cachés
    db1 = delta1.sum(axis=0, keepdims=True)  # gradient des biais cachés
    W1 -= taux_xor * dW1  # mise à jour des poids cachés
    b1 -= taux_xor * db1  # mise à jour des biais cachés
    W2 -= taux_xor * dW2  # mise à jour des poids de sortie
    b2 -= taux_xor * db2  # mise à jour du biais de sortie
    if epoque % 2000 == 0:  # affiche la perte de temps en temps
        perte_xor = np.mean((a2 - y_xor) ** 2)  # erreur quadratique moyenne
        print(f"époque {epoque:5d} | perte {perte_xor:.4f}")  # suivi de l'apprentissage

# %%
probabilites = a2.ravel()  # probabilités finales pour les 4 entrées
predictions = (probabilites > 0.5).astype(int)  # seuil à 0,5 : au-dessus, on prédit 1
print(np.round(probabilites, 3))  # ≈ [0, 1, 1, 0]
print(f"Précision : {np.mean(predictions == y_xor.ravel()) * 100:.0f} %")  # 4 bonnes réponses sur 4

# %% [markdown]
# Le réseau à 4 neurones cachés a appris le XOR. Les gradients `delta` sont la
# règle de la chaîne appliquée couche par couche : c'est la
# **rétropropagation du gradient**. Personne ne fait cela à la main en
# production : on utilise une bibliothèque.

# %% [markdown]
# ## 4. Le même XOR avec PyTorch
#
# PyTorch calcule les gradients automatiquement. Le schéma est toujours le
# même : `zero_grad` → prédiction → perte → `backward` → `step`.

# %%
import torch  # framework de réseaux de neurones
from torch import nn  # module des couches (Linear, ReLU, etc.)

torch.manual_seed(42)  # graine pour retrouver les mêmes résultats
X_t = torch.tensor(X_xor, dtype=torch.float32)  # convertit les entrées en tenseurs
y_t = torch.tensor(y_xor, dtype=torch.float32)  # convertit les sorties attendues
modele_xor = nn.Sequential(  # empile les couches dans l'ordre
    nn.Linear(2, 4),  # couche linéaire : 2 entrées, 4 neurones
    nn.Tanh(),  # activation non linéaire (équivalent moderne de la sigmoïde)
    nn.Linear(4, 1),  # couche de sortie : 1 neurone
    nn.Sigmoid(),  # transforme la sortie en probabilité entre 0 et 1
)
optimiseur_xor = torch.optim.SGD(modele_xor.parameters(), lr=0.2)  # SGD : descente de gradient stochastique
fonction_perte = nn.BCELoss()  # entropie croisée binaire : perte pour deux classes

# %%
for epoque in range(3000):  # 3000 passages sur les 4 exemples
    optimiseur_xor.zero_grad()  # remet les gradients à zéro (sinon ils s'accumulent)
    sortie = modele_xor(X_t)  # passe avant : calcule les prédictions
    perte = fonction_perte(sortie, y_t)  # calcule la perte
    perte.backward()  # rétropropagation : PyTorch calcule tous les gradients
    optimiseur_xor.step()  # met à jour les poids

# %%
print(modele_xor(X_t).detach().ravel().numpy().round(3))  # probabilités apprises
print(f"Perte finale : {perte.item():.4f}")  # perte très faible : XOR appris

# %% [markdown]
# ## 5. Un vrai cas : détecter un défaut moteur
#
# On reprend les mesures du notebook 10 : vibration, température, courant et
# régime. Cette fois, le modèle est un réseau de neurones à deux couches
# cachées. Trois précautions sont indispensables :
#
# 1. séparer les données en apprentissage et test ;
# 2. normaliser les entrées (les échelles sont très différentes) ;
# 3. passer le modèle en mode évaluation avant de mesurer la performance.

# %%
import pandas as pd  # lecture des données
from sklearn.model_selection import train_test_split  # découpage train/test
from sklearn.preprocessing import StandardScaler  # normalisation des colonnes
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report  # mesures de qualité

df = pd.read_csv("data/maintenance_capteurs.csv")  # charge le jeu de données simulé
features = ["vibration_rms", "temperature_C", "courant_A", "regime_tr_min"]  # les 4 entrées du modèle
X = df[features].to_numpy(dtype="float32")  # matrice des entrées (1600 × 4)
y = df["defaut"].to_numpy(dtype="int64")  # étiquettes : 0 sain, 1 défaillant
print(df.shape, np.bincount(y))  # 1600 lignes ; proportion de défauts

# %%
X_train, X_test, y_train, y_test = train_test_split(  # découpe aléatoire et stratifiée
    X, y, test_size=0.25, random_state=42, stratify=y  # 25 % pour le test, mêmes proportions de défauts
)
normaliseur = StandardScaler().fit(X_train)  # apprend moyenne et écart-type SUR LE TRAIN uniquement
X_train_n = normaliseur.transform(X_train)  # normalise les données d'apprentissage
X_test_n = normaliseur.transform(X_test)  # applique la même transformation au test
print(X_train_n.mean(axis=0).round(2))  # chaque colonne est centrée autour de 0

# %%
X_train_t = torch.tensor(X_train_n, dtype=torch.float32)  # convertit les entrées d'apprentissage en tenseur
y_train_t = torch.tensor(y_train, dtype=torch.long)  # les étiquettes doivent être des entiers 64 bits
X_test_t = torch.tensor(X_test_n, dtype=torch.float32)  # convertit les entrées de test
y_test_t = torch.tensor(y_test, dtype=torch.long)  # convertit les étiquettes de test

jeu_train = torch.utils.data.TensorDataset(X_train_t, y_train_t)  # associe entrées et étiquettes
chargeur = torch.utils.data.DataLoader(jeu_train, batch_size=64, shuffle=True)  # lots de 64, mélangés à chaque époque

# %%
torch.manual_seed(42)  # reproductibilité de l'initialisation
modele = nn.Sequential(  # réseau 4 → 16 → 8 → 2
    nn.Linear(4, 16),  # première couche cachée : 4 entrées, 16 neurones
    nn.ReLU(),  # activation ReLU : max(0, z)
    nn.Linear(16, 8),  # deuxième couche cachée : 16 → 8
    nn.ReLU(),  # seconde activation ReLU
    nn.Linear(8, 2),  # sortie : deux scores (sain / défaillant)
)
fonction_perte = nn.CrossEntropyLoss()  # perte adaptée à une classification multiclasse
optimiseur = torch.optim.Adam(modele.parameters(), lr=0.01)  # Adam : descente de gradient adaptative
print(modele)  # affiche l'architecture et le nombre de paramètres

# %%
for epoque in range(120):  # 120 passages complets sur le jeu d'apprentissage
    modele.train()  # mode apprentissage (utile pour dropout, inutile ici mais bonne habitude)
    perte_cumulee = 0.0  # remise à zéro de la perte de l'époque
    for x_lot, y_lot in chargeur:  # parcourt les lots de 64 exemples
        optimiseur.zero_grad()  # annule les gradients précédents
        sorties = modele(x_lot)  # scores bruts pour chaque classe
        perte = fonction_perte(sorties, y_lot)  # perte de ce lot
        perte.backward()  # calcule les gradients
        optimiseur.step()  # met à jour les poids
        perte_cumulee += perte.item() * len(x_lot)  # accumule la perte pondérée par la taille du lot
    if epoque % 20 == 0:  # affiche toutes les 20 époques
        print(f"époque {epoque:3d} | perte moyenne {perte_cumulee / len(X_train_t):.4f}")  # suivi

# %%
modele.eval()  # mode évaluation : désactive les comportements propres à l'entraînement
with torch.no_grad():  # pas besoin de calculer les gradients pour évaluer
    scores_test = modele(X_test_t)  # scores bruts sur le jeu de test
    predictions_test = scores_test.argmax(dim=1).numpy()  # classe la plus probable pour chaque exemple
print(f"Précision sur le test : {accuracy_score(y_test, predictions_test) * 100:.1f} %")  # part de bonnes prédictions

# %%
matrice = confusion_matrix(y_test, predictions_test)  # lignes = vérité, colonnes = prédiction
print(matrice)  # [[vrais sains, faux défauts], [défauts manqués, vrais défauts]]
print(classification_report(y_test, predictions_test, target_names=["sain", "défaillant"]))  # précision/rappel par classe

# %% [markdown]
# La précision tourne autour de 90 %. Le plafond n'est pas de 100 % : environ
# 3 % des étiquettes du fichier sont volontairement mal annotées, et certains
# cas sont réellement ambigus. En maintenance, on préfère souvent **détecter
# trop de défauts** (faux positifs) plutôt que d'en manquer : c'est un choix qui
# se règle sur les probabilités, pas seulement sur la classe prédite.

# %%
with torch.no_grad():  # calcul des probabilités
    probabilites_test = torch.softmax(modele(X_test_t), dim=1).numpy()  # softmax transforme les scores en probabilités
print(np.round(probabilites_test[:5], 3))  # [P(sain), P(défaillant)] pour 5 cas

# %% [markdown]
# ## 6. Sauvegarder et recharger un modèle
#
# Un modèle entraîné doit être réutilisé sans réentraînement, par exemple sur
# un Raspberry Pi au bord de la machine. On sauvegarde le `state_dict` (les
# poids) et on recharge la **même architecture**.

# %%
torch.save(modele.state_dict(), "models/mlp_maintenance.pt")  # enregistre les poids dans le dossier models/
print("Modèle sauvegardé dans models/mlp_maintenance.pt")  # confirme l'écriture

# %%
modele_recharge = nn.Sequential(  # il faut recréer exactement la même architecture
    nn.Linear(4, 16),  # même première couche
    nn.ReLU(),  # même activation
    nn.Linear(16, 8),  # même deuxième couche
    nn.ReLU(),  # même activation
    nn.Linear(8, 2),  # même sortie
)
etat = torch.load("models/mlp_maintenance.pt", weights_only=True)  # charge uniquement les tenseurs de poids
modele_recharge.load_state_dict(etat)  # injecte les poids dans l'architecture
modele_recharge.eval()  # passe en mode évaluation
print("Modèle rechargé et prêt à l'inférence")  # confirmation

# %%
nouvelle_mesure = {  # une mesure arrivée du capteur
    "vibration_rms": 0.95,  # vibration élevée
    "temperature_C": 71.0,  # température supérieure à la normale
    "courant_A": 3.55,  # courant un peu fort
    "regime_tr_min": 1380.0,  # régime qui baisse
}
ligne = pd.DataFrame([nouvelle_mesure])[features]  # respecte l'ordre des colonnes attendu
entree = torch.tensor(normaliseur.transform(ligne.to_numpy(dtype="float32")), dtype=torch.float32)  # normalise puis convertit
with torch.no_grad():  # inférence sans gradient
    proba = torch.softmax(modele_recharge(entree), dim=1)[0]  # probabilités des deux classes
print(f"P(défaillant) = {proba[1]:.2f}")  # probabilité de défaut pour cette mesure

# %% [markdown]
# ## 7. À vous de jouer
#
# **Exercice 1** — Reprenez le réseau XOR NumPy et remplacez les 4 neurones
# cachés par 2. Le réseau converge-t-il aussi bien ? Modifiez `W1`, `b1`, `W2`
# et les dimensions en conséquence (attention : `W2` devient 2 × 1).

# %%
# TODO : recopiez la boucle XOR avec une couche cachée de 2 neurones

# TODO : affichez les probabilités finales et la précision

# %% [markdown]
# **Exercice 2** — Dans l'entraînement PyTorch du moteur, remplacez
# `lr=0.01` par `lr=0.0001` et gardez 120 époques. Observez la perte :
# l'apprentissage est beaucoup plus lent. Conclusion sur le choix du taux ?

# %%
# TODO : créez un nouveau modèle identique avec torch.manual_seed(42)

# TODO : entraînez-le avec lr=0.0001 et affichez la perte finale

# TODO : comparez avec la précision obtenue à l'étape 5

# %% [markdown]
# **Exercice 3** — Pour la mesure `nouvelle_mesure`, abaissez le seuil de
# décision de 0,5 à 0,3 : `proba[1] > 0.3` déclenche l'alarme. Combien de
# défauts supplémentaires détecteriez-vous sur le jeu de test, et combien de
# fausses alarmes ?

# %%
# TODO : calculez predictions_seuil = (probabilites_test[:, 1] > 0.3).astype(int)

# TODO : affichez la nouvelle matrice de confusion

# TODO : concluez sur le compromis faux positifs / défauts manqués

# %% [markdown]
# ## 8. À retenir
#
# - Un neurone : `activation(w·x + b)` ; la sigmoïde donne une probabilité.
# - Entraîner = minimiser une perte par descente de gradient.
# - Une couche cachée suffit pour le XOR ; sans non-linéarité, un réseau reste linéaire.
# - PyTorch : `zero_grad`, passe avant, `backward`, `step`.
# - Toujours normaliser les entrées et évaluer sur des données jamais vues.
# - `state_dict` + même architecture = modèle rechargeable.
#
# **Notebook suivant** : `12 — Projet final : tri de composants et détection d'anomalies`,
# où nous appliquons tout cela à des images et à un pipeline complet.
