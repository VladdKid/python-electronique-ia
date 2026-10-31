"""Génère les jeux de données simulés utilisés dans les notebooks.

Les données sont déterministes (graine fixe) : relancer le script redonne
exactement les mêmes fichiers. Exécution :

    python scripts/generer_donnees.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def calibration_thermistance() -> None:
    """Courbe résistance / température d'une CTN (coefficient négatif)."""
    rng = np.random.default_rng(2026)
    temperature = np.linspace(-10.0, 80.0, 150)
    beta, t0, r0 = 3950.0, 298.15, 10_000.0
    resistance = r0 * np.exp(beta * (1.0 / (temperature + 273.15) - 1.0 / t0))
    resistance *= 1.0 + rng.normal(0.0, 0.015, temperature.size)
    table = pd.DataFrame(
        {
            "temperature_C": np.round(temperature, 1),
            "resistance_ohm": np.round(resistance, 1),
        }
    )
    table.to_csv(DATA / "calibration_thermistance.csv", index=False)


def maintenance_capteurs() -> None:
    """Mesures vibratoires et thermiques d'un moteur, étiquetées sain / défaillant.

    Les classes se recouvrent partiellement et 3 % des étiquettes sont
    volontairement erronées, comme dans un carnet de maintenance réel.
    """
    rng = np.random.default_rng(7)
    n = 1600
    defaut = rng.random(n) < 0.3

    vibration = rng.normal(0.45, 0.10, n) + defaut * rng.normal(0.30, 0.10, n)
    temperature = rng.normal(58.0, 5.0, n) + defaut * rng.normal(8.0, 3.0, n)
    courant = rng.normal(3.20, 0.22, n) + defaut * rng.normal(0.30, 0.12, n)
    regime = rng.normal(1450.0, 70.0, n) - defaut * rng.normal(120.0, 40.0, n)

    etiquettes = defaut.astype(int)
    erreurs_annotation = rng.random(n) < 0.03
    etiquettes[erreurs_annotation] = 1 - etiquettes[erreurs_annotation]

    table = pd.DataFrame(
        {
            "vibration_rms": np.round(np.clip(vibration, 0.05, None), 3),
            "temperature_C": np.round(temperature, 1),
            "courant_A": np.round(np.clip(courant, 1.0, None), 2),
            "regime_tr_min": np.round(regime, 0),
            "defaut": etiquettes,
        }
    )
    table.to_csv(DATA / "maintenance_capteurs.csv", index=False)


def mesures_capteur() -> None:
    """Journal d'un capteur d'ambiance : mesures toutes les 10 secondes sur 5 heures."""
    rng = np.random.default_rng(31)
    n = 1800
    horodatage = pd.date_range("2026-09-14 08:00:00", periods=n, freq="10s")

    temperature = 21.5 + 1.8 * np.sin(np.linspace(0, 2 * np.pi, n)) + rng.normal(0, 0.15, n)
    humidite = 48.0 + 5.0 * np.sin(np.linspace(0, np.pi, n)) + rng.normal(0, 0.8, n)
    tension = 5.02 + 0.01 * np.sin(np.linspace(0, 4 * np.pi, n)) + rng.normal(0, 0.004, n)
    courant = 12.0 + 0.9 * np.sin(np.linspace(0, 2 * np.pi, n)) + rng.normal(0, 0.12, n)

    coupures = rng.choice(n, size=6, replace=False)
    courant[coupures] = rng.uniform(45.0, 60.0, coupures.size)

    table = pd.DataFrame(
        {
            "horodatage": horodatage.strftime("%Y-%m-%d %H:%M:%S"),
            "temperature_C": np.round(temperature, 2),
            "humidite_pct": np.round(np.clip(humidite, 0, 100), 1),
            "tension_V": np.round(tension, 3),
            "courant_mA": np.round(courant, 2),
        }
    )
    table.to_csv(DATA / "mesures_capteur.csv", index=False)


if __name__ == "__main__":
    DATA.mkdir(exist_ok=True)
    calibration_thermistance()
    maintenance_capteurs()
    mesures_capteur()
    for fichier in sorted(DATA.glob("*.csv")):
        print(f"{fichier.relative_to(ROOT)} : {sum(1 for _ in fichier.open('r', encoding='utf-8')) - 1} lignes")
