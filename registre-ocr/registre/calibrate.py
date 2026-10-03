"""Calibration des seuils de qualité à partir des augmentations (flou réel connu).

Idée : une photo est « lisible » si le flou réel, ramené à l'échelle du gabarit, reste sous une limite
(par défaut 1,5 px à 150 dpi, soit environ 1/12 de la hauteur d'un chiffre manuscrit). On cherche le seuil
sur le flou ESTIMÉ qui sépare au mieux lisible / illisible, en pénalisant 2x plus les photos illisibles
acceptées (elles produisent des erreurs) que les photos lisibles refusées (elles coûtent une reprise).

Quand l'évaluation de bout en bout tournera avec le vrai VLM, remplacez la définition de « lisible »
par « exactitude des champs de la page >= 95 % » : c'est la vraie cible.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .quality import assess


def calibrate_quality(aug_dir: str, readable_sigma: float = 1.5, cost_fa: float = 2.0) -> dict:
    est, true = [], []
    for sc in sorted(Path(aug_dir).glob("*.json")):
        m = json.loads(sc.read_text())
        p = m["params"]
        H = np.array(m["H_page_to_photo"])
        s = np.sqrt(abs(np.linalg.det(H[:2, :2])))
        true.append(np.sqrt((p["defocus_px"] / 2) ** 2 + p["motion_px"] ** 2 / 12) / s)
        est.append(assess(cv2.imread(str(sc.with_suffix(".jpg")))).metrics["blur_sigma_template"])
    est, true = np.array(est), np.array(true)
    readable = true <= readable_sigma
    best = None
    for t in np.linspace(0.8, 4.0, 65):
        acc = est <= t
        cost = cost_fa * np.sum(acc & ~readable) + np.sum(~acc & readable)
        if best is None or cost < best[0]:
            best = (cost, t, np.mean(acc[readable]) if readable.any() else 0, np.mean(acc[~readable]) if (~readable).any() else 0)
    out = {"n": int(len(est)), "corr": float(np.corrcoef(est, true)[0, 1]), "blur_sigma_max": round(float(best[1]), 2),
           "lisibles_acceptees": round(float(best[2]), 3), "illisibles_acceptees": round(float(best[3]), 3)}
    print(json.dumps(out, indent=1, ensure_ascii=False))
    print("-> reporter blur_sigma_max dans QualityThresholds (registre/quality.py)")
    return out
