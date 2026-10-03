"""Lecteurs purement visuels (OpenCV), sans modèle :

- `ink_ratio`  : part de pixels d'ENCRE AJOUTÉE dans une zone, par comparaison au gabarit vierge
                 (les traits imprimés du gabarit sont exclus). Sert à dire « zone vide » vs « zone écrite »
                 de façon indépendante du modèle de lecture -> détecte les hallucinations et les oublis.
- `read_checkbox` : case cochée ou non, avec une confiance qui dépend de la distance au seuil.

Les deux travaillent sur l'image REDRESSÉE dans le repère du gabarit (sortie de align.py).
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

CHECK_THRESHOLD = 0.10   # part d'encre dans l'intérieur de la case (à calibrer : cli calibrate-checkbox)
CHECK_SPREAD = 0.05      # largeur de la zone d'incertitude autour du seuil
INK_DELTA = 45           # écart de luminance au fond pour compter un pixel comme encre


def _darkness(gray: np.ndarray) -> np.ndarray:
    """Traits fins plus sombres que leur voisinage (« black-hat » morphologique).
    Les ombres et dégradés, larges, disparaissent ; l'encre et les traits imprimés, fins, restent."""
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    return cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k).astype(np.int16)


def _gray(img: np.ndarray) -> np.ndarray:
    return img.min(axis=2) if img.ndim == 3 else img


def _crop(img: np.ndarray, bbox, pad: int = 0) -> np.ndarray:
    x0, y0, x1, y1 = [int(round(v)) for v in bbox]
    h, w = img.shape[:2]
    return img[max(0, y0 - pad):min(h, y1 + pad), max(0, x0 - pad):min(w, x1 + pad)]


def ink_ratio(aligned: np.ndarray, blank: np.ndarray, bbox, valid: np.ndarray | None = None, pad: int = 6,
              ref_contrast: float | None = None) -> float:
    """`ref_contrast` = assombrissement des traits imprimés voisins sur la photo (printed_contrast) :
    le seuil d'encre s'adapte au contraste réel (photo sombre, ombre) au lieu d'être fixe."""
    a = _gray(_crop(aligned, bbox, pad))
    t = _gray(_crop(blank, bbox, pad))
    if a.size == 0 or a.shape != t.shape:
        return 0.0
    da, dt = _darkness(a), _darkness(t)
    thr = INK_DELTA if ref_contrast is None else float(np.clip(0.45 * ref_contrast, 12, INK_DELTA))
    # traits imprimés du gabarit, élargis : absorbe 2-3 px d'erreur résiduelle d'alignement
    printed = cv2.dilate((dt > 25).astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    ink = (da > thr) & ~printed
    ink = cv2.morphologyEx(ink.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)).astype(bool)
    if valid is not None:
        v = _crop(valid, bbox, pad).astype(bool)
        if v.shape == ink.shape:
            ink &= v
    m = pad + 3  # on ignore aussi une bordure de 3 px à l'intérieur de la zone
    core = ink[m:-m or None, m:-m or None]
    return float(core.mean()) if core.size else 0.0


def printed_contrast(aligned: np.ndarray, blank: np.ndarray, bbox, pad: int = 14) -> float | None:
    """Les traits IMPRIMÉS autour de la zone sont-ils encore visibles sur la photo ?
    Si non (reflet, ombre dure, flou), on ne peut pas affirmer qu'une zone sans encre est vide.
    Renvoie l'assombrissement médian aux pixels imprimés du gabarit (None si pas de trait imprimé)."""
    a = _gray(_crop(aligned, bbox, pad))
    t = _gray(_crop(blank, bbox, pad))
    if a.size == 0 or a.shape != t.shape:
        return None
    printed = _darkness(t) > 40
    if printed.sum() < 15:
        return None
    return float(np.median(_darkness(a)[printed]))


def visible_fraction(valid: np.ndarray | None, bbox) -> float:
    if valid is None:
        return 1.0
    v = _crop(valid, bbox)
    return float(v.mean()) if v.size else 0.0


@dataclass
class CheckboxReading:
    checked: bool
    ink: float
    confidence: float


def read_checkbox(aligned: np.ndarray, blank: np.ndarray, bbox, threshold: float = CHECK_THRESHOLD) -> CheckboxReading:
    # intérieur de la case (on retire le cadre imprimé) + petite marge : une coche déborde souvent
    x0, y0, x1, y1 = bbox
    inner = (x0 + 3, y0 + 3, x1 - 3, y1 - 3)
    a = _gray(_crop(aligned, inner))
    if a.size == 0:
        return CheckboxReading(False, 0.0, 0.0)
    # fond estimé autour de la case (et non dedans : une case pleine n'a pas de fond)
    ring = _gray(_crop(aligned, bbox, pad=10)).astype(np.float32)
    bg = float(np.percentile(ring, 90))
    ink = float(((bg - a.astype(np.float32)) > INK_DELTA * 0.8).mean())
    d = (ink - threshold) / CHECK_SPREAD
    conf = float(1 / (1 + np.exp(-abs(d) * 2.2)))  # 0,5 au seuil -> ~1 loin du seuil
    return CheckboxReading(ink > threshold, ink, conf)
