"""Contrôle qualité d'une photo AVANT de l'accepter (OpenCV seul : portable sur téléphone).

Verdict : ACCEPTER, ou REPRENDRE avec des raisons lisibles par la sage-femme.

Netteté : on ESTIME LE RAYON DE FLOU (sigma, en pixels) au lieu d'un score arbitraire.
  Sur les bords les plus contrastés (texte imprimé, traits de tableau), on compare le gradient de l'image
  à celui de l'image re-floutée avec un sigma connu. Pour un bord déjà flou de sigma_b, le rapport vaut
  r = sqrt(sigma_b² + sigma_ref²) / sigma_b, d'où sigma_b = sigma_ref / sqrt(r² - 1).
  On le fait dans 4 directions et on garde la pire : le flou de bougé est directionnel.
  Le sigma est ensuite ramené à l'échelle du gabarit (150 dpi) grâce à la taille de la page dans la photo :
  c'est ce qui compte pour lire une écriture de ~20 px de haut.
  Corrélation avec le flou réel sur les augmentations : ~0,84 (la variance du laplacien : ~0).

Également : par tuile (flou partiel), exposition, reflets, cadrage (page entière et assez grande).

Les seuils sont des points de départ : `python -m registre.cli calibrate-quality` les recalcule
à partir des augmentations (voir README).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import cv2
import numpy as np

TEMPLATE_SIZE = (1240, 1754)  # (W, H) d'une page A4 à 150 dpi


@dataclass
class QualityThresholds:
    blur_sigma_max: float = 1.8        # px gabarit : au-delà, l'écriture devient illisible
    tile_blur_sigma_max: float = 2.4   # px gabarit, pour une tuile
    blurry_tile_ratio_max: float = 0.25
    luma_min: float = 60.0
    luma_max: float = 235.0
    dark_ratio_max: float = 0.40
    glare_ratio_max: float = 0.015
    page_width_min_px: float = 700.0   # largeur de la page dans la photo : en dessous, trop peu de pixels


@dataclass
class QualityReport:
    accept: bool
    reasons: list[str] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    page_quad: list | None = None
    score: float = 0.0  # 0..1, réutilisé dans la confiance des champs

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- contour de page
def order_corners(pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, np.float32)
    s = pts.sum(1)
    d = np.diff(pts, axis=1).ravel()
    return np.float32([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]])


def find_page_quad(img: np.ndarray, work: int = 1000) -> np.ndarray | None:
    """Contour de la feuille : plus grand quadrilatère convexe (coins TL, TR, BR, BL), ou None."""
    h, w = img.shape[:2]
    s = work / max(h, w)
    small = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    best = None
    for lo, hi in ((30, 90), (50, 150), (10, 60)):
        edges = cv2.dilate(cv2.Canny(gray, lo, hi), np.ones((5, 5), np.uint8))
        cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in sorted(cnts, key=cv2.contourArea, reverse=True)[:5]:
            approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
            area = cv2.contourArea(approx)
            if len(approx) == 4 and cv2.isContourConvex(approx) and area > 0.12 * small.shape[0] * small.shape[1]:
                if best is None or area > best[0]:
                    best = (area, approx.reshape(4, 2).astype(np.float32))
        if best is not None:
            break
    return None if best is None else order_corners(best[1] / s)


# --------------------------------------------------------------------------- flou
def blur_sigma(gray: np.ndarray, sigma_ref: float = 2.0, top: float = 0.01) -> float:
    """Rayon de flou estimé (px de l'image fournie), pire direction. Voir docstring du module."""
    f = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 0.7)  # écrase le bruit capteur
    b = cv2.GaussianBlur(f, (0, 0), sigma_ref)
    gx, gy = cv2.Sobel(f, cv2.CV_32F, 1, 0), cv2.Sobel(f, cv2.CV_32F, 0, 1)
    bx, by = cv2.Sobel(b, cv2.CV_32F, 1, 0), cv2.Sobel(b, cv2.CV_32F, 0, 1)
    worst = 0.0
    for a in np.deg2rad([0, 45, 90, 135]):
        m = np.abs(np.cos(a) * gx + np.sin(a) * gy)
        mb = cv2.dilate(np.abs(np.cos(a) * bx + np.sin(a) * by), np.ones((5, 5), np.uint8))
        thr = np.quantile(m, 1 - top)
        if thr <= 1e-3:
            continue
        sel = m >= thr
        r = float(np.median(m[sel] / np.maximum(mb[sel], 1e-3)))
        worst = max(worst, sigma_ref / np.sqrt(max(r * r - 1, 1e-3)))
    return float(worst)


# --------------------------------------------------------------------------- verdict
def assess(img: np.ndarray, th: QualityThresholds | None = None) -> QualityReport:
    th = th or QualityThresholds()
    H0, W0 = img.shape[:2]
    quad = find_page_quad(img)
    gray = img.min(axis=2) if img.ndim == 3 else img
    m: dict = {}

    # échelle photo -> gabarit (px photo par px gabarit)
    if quad is not None:
        page_w = float((np.linalg.norm(quad[1] - quad[0]) + np.linalg.norm(quad[2] - quad[3])) / 2)
        page_h = float((np.linalg.norm(quad[3] - quad[0]) + np.linalg.norm(quad[2] - quad[1])) / 2)
        if page_w > page_h:  # page photographiée en paysage
            page_w, page_h = page_h, page_w
        scale = page_w / TEMPLATE_SIZE[0]
        x0, y0 = np.clip(quad.min(0).astype(int), 0, None)
        x1, y1 = quad.max(0).astype(int)
        roi = gray[y0:min(y1, H0), x0:min(x1, W0)]
        margin = 0.01 * max(H0, W0)
        m["page_touches_border"] = bool(((quad[:, 0] < margin) | (quad[:, 0] > W0 - margin) |
                                         (quad[:, 1] < margin) | (quad[:, 1] > H0 - margin)).any())
    else:  # page non détourée : on suppose qu'elle remplit à peu près le cadre
        page_w = float(min(H0, W0)) * 0.9
        scale = page_w / TEMPLATE_SIZE[0]
        roi = gray
        m["page_touches_border"] = None
    m["page_width_px"] = page_w
    m["scale_photo_per_template"] = scale

    sig = blur_sigma(roi)
    m["blur_sigma_photo"] = sig
    m["blur_sigma_template"] = sig / max(scale, 1e-3)

    # tuiles 3x3 (flou partiel : mise au point ratée sur un bord, page courbée)
    h, w = roi.shape[:2]
    tiles, blurry, worst = 0, 0, None
    for i in range(3):
        for j in range(3):
            t = roi[i * h // 3:(i + 1) * h // 3, j * w // 3:(j + 1) * w // 3]
            if t.size == 0 or (cv2.Canny(t, 50, 150) > 0).mean() < 0.01:
                continue  # tuile sans contenu
            tiles += 1
            ts = blur_sigma(t) / max(scale, 1e-3)
            if ts > th.tile_blur_sigma_max:
                blurry += 1
                if worst is None or ts > worst[0]:
                    worst = (ts, i, j)
    m["content_tiles"] = tiles
    m["blurry_tile_ratio"] = blurry / tiles if tiles else 1.0

    m["luma_median"] = float(np.median(roi))
    m["dark_ratio"] = float((roi < 50).mean())
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    glare = ((hsv[:, :, 2] > 248) & (hsv[:, :, 1] < 30)).astype(np.uint8)
    glare = cv2.morphologyEx(glare, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    m["glare_ratio"] = float(glare.mean())

    reasons = []
    rows, cols = ["en haut", "au milieu", "en bas"], ["à gauche", "au centre", "à droite"]
    if m["blur_sigma_template"] > th.blur_sigma_max:
        reasons.append("Photo floue : tenez le téléphone immobile et touchez l'écran sur la page pour faire la mise au point.")
    elif m["blurry_tile_ratio"] > th.blurry_tile_ratio_max and worst:
        reasons.append(f"Une partie de la page est floue ({rows[worst[1]]} {cols[worst[2]]}) : "
                       "tenez le téléphone parallèle à la page.")
    if m["luma_median"] < th.luma_min or m["dark_ratio"] > th.dark_ratio_max:
        reasons.append("Photo trop sombre : rapprochez-vous d'une fenêtre ou d'une lampe.")
    if m["luma_median"] > th.luma_max:
        reasons.append("Photo surexposée : évitez la lumière directe sur la page.")
    if m["glare_ratio"] > th.glare_ratio_max:
        reasons.append("Reflet sur la page : inclinez légèrement le téléphone.")
    if page_w < th.page_width_min_px:
        reasons.append("Page trop petite dans la photo : rapprochez-vous.")
    if m["page_touches_border"]:
        reasons.append("La page touche le bord de la photo : vérifiez que les 4 coins sont visibles.")

    parts = [
        float(np.clip(1.0 - (m["blur_sigma_template"] - 0.8) / (th.blur_sigma_max * 1.5), 0, 1)),
        1.0 - min(1.0, m["blurry_tile_ratio"]),
        1.0 - min(1.0, m["glare_ratio"] / (th.glare_ratio_max * 3)),
        1.0 if th.luma_min <= m["luma_median"] <= th.luma_max else 0.5,
        float(np.clip(page_w / (th.page_width_min_px * 1.5), 0, 1)),
    ]
    return QualityReport(accept=not reasons, reasons=reasons, metrics=m,
                         page_quad=quad.tolist() if quad is not None else None,
                         score=float(np.clip(np.mean(parts), 0, 1)))
