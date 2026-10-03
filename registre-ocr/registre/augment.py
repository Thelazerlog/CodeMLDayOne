"""Augmentation : fabrique des « photos de terrain » à partir des pages propres, avec géométrie connue.

Chaque image générée est accompagnée d'un JSON contenant :
- H_page_to_photo : homographie exacte page propre -> photo (vérité terrain pour l'alignement)
- les paramètres de dégradation (flou, bougé, lumière, reflets, recadrage...) et un niveau de sévérité

Ce qui est simulé (ordre d'application ≈ ordre physique) :
1. papier   : pli, froissement léger, teinte jaunie
2. géométrie: rotation, perspective (téléphone pas parallèle), page petite ou décentrée, page COUPÉE par le cadre
3. fond     : table / tissu / bruit coloré autour de la feuille
4. lumière  : dégradé, ombre portée (main/téléphone), faible lumière, dominante de couleur, reflet
5. capteur  : flou de mise au point, flou de bougé, bruit, compression JPEG
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class AugParams:
    severity: str
    out_w: int
    out_h: int
    rotation_deg: float
    perspective: float        # amplitude du déplacement des coins (fraction de la page)
    page_scale: float         # taille de la page / taille de l'image
    crop_side: str | None     # côté où la page sort du cadre
    crop_frac: float
    defocus_px: float         # rayon du disque de flou
    motion_px: float
    motion_angle: float
    noise_sigma: float
    jpeg_quality: int
    gamma: float              # >1 = plus sombre
    shadow: bool
    glare: bool
    fold: bool
    color_cast: tuple


LEVELS = {
    #            rot   persp  scale         crop_p  defocus  motion  noise  jpeg       gamma
    "leger":   (4,   0.03, (0.80, 0.95), 0.0,  (0, 1.0), (0, 2),  (0, 3),  (85, 95), (1.0, 1.2)),
    "moyen":   (10,  0.07, (0.65, 0.92), 0.15, (0, 2.5), (0, 6),  (2, 7),  (60, 90), (1.0, 1.6)),
    "fort":    (20,  0.12, (0.50, 0.90), 0.35, (1, 4.0), (0, 12), (4, 12), (35, 75), (1.2, 2.3)),
}


def sample_params(rng: np.random.Generator, severity: str) -> AugParams:
    rot, persp, scale, crop_p, defocus, motion, noise, jpeg, gamma = LEVELS[severity]
    portrait = rng.random() < 0.85
    out_w, out_h = (1500, 2000) if portrait else (2000, 1500)
    crop_side = rng.choice(["top", "bottom", "left", "right"]) if rng.random() < crop_p else None
    return AugParams(
        severity=severity, out_w=out_w, out_h=out_h,
        rotation_deg=float(rng.uniform(-rot, rot) + (90 if not portrait and rng.random() < 0.5 else 0)),
        perspective=float(rng.uniform(0, persp)),
        page_scale=float(rng.uniform(*scale)),
        crop_side=crop_side, crop_frac=float(rng.uniform(0.04, 0.15)) if crop_side else 0.0,
        defocus_px=float(rng.uniform(*defocus)),
        motion_px=float(rng.uniform(*motion)), motion_angle=float(rng.uniform(0, 180)),
        noise_sigma=float(rng.uniform(*noise)), jpeg_quality=int(rng.integers(*jpeg)),
        gamma=float(rng.uniform(*gamma)),
        shadow=bool(rng.random() < (0.2 if severity == "leger" else 0.5)),
        glare=bool(rng.random() < (0.05 if severity == "leger" else 0.25)),
        fold=bool(rng.random() < 0.3),
        color_cast=tuple(float(x) for x in rng.uniform(0.85, 1.12, 3)),
    )


def _background(rng, h, w) -> np.ndarray:
    base = rng.integers(40, 200, 3).astype(np.float32)
    bg = np.ones((h, w, 3), np.float32) * base
    kind = rng.integers(0, 3)
    if kind == 0:  # bois / lignes
        y = np.arange(h)[:, None] + 15 * np.sin(np.arange(w)[None, :] / rng.uniform(60, 200))
        bg += (np.sin(y / rng.uniform(4, 15))[..., None] * rng.uniform(5, 25))
    elif kind == 1:  # tissu : bruit basse fréquence
        n = cv2.resize(rng.normal(0, 1, (h // 40 + 1, w // 40 + 1)).astype(np.float32), (w, h))
        bg += n[..., None] * rng.uniform(10, 30)
    bg += rng.normal(0, 4, bg.shape)
    return np.clip(bg, 0, 255).astype(np.uint8)


def _page_homography(rng, p: AugParams, pw: int, ph: int) -> np.ndarray:
    W, H = p.out_w, p.out_h
    s = p.page_scale * min(W / pw, H / ph)
    cx, cy = W / 2 + rng.uniform(-0.08, 0.08) * W, H / 2 + rng.uniform(-0.08, 0.08) * H
    if p.crop_side == "left":
        cx = pw * s / 2 - p.crop_frac * pw * s
    elif p.crop_side == "right":
        cx = W - pw * s / 2 + p.crop_frac * pw * s
    elif p.crop_side == "top":
        cy = ph * s / 2 - p.crop_frac * ph * s
    elif p.crop_side == "bottom":
        cy = H - ph * s / 2 + p.crop_frac * ph * s
    a = np.deg2rad(p.rotation_deg)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    src = np.float32([[0, 0], [pw, 0], [pw, ph], [0, ph]])
    centered = (src - [pw / 2, ph / 2]) * s
    jitter = rng.uniform(-p.perspective, p.perspective, (4, 2)) * [pw * s, ph * s]
    dst = (centered + jitter) @ R.T + [cx, cy]
    return cv2.getPerspectiveTransform(src, np.float32(dst))


def _disk_kernel(r: float) -> np.ndarray:
    k = max(1, int(np.ceil(r)))
    y, x = np.mgrid[-k:k + 1, -k:k + 1]
    ker = ((x ** 2 + y ** 2) <= r ** 2).astype(np.float32)
    return ker / ker.sum()


def _motion_kernel(length: float, angle: float) -> np.ndarray:
    L = max(1, int(round(length)))
    ker = np.zeros((2 * L + 1, 2 * L + 1), np.float32)
    a = np.deg2rad(angle)
    for t in np.linspace(-L / 2, L / 2, 2 * L + 1):
        ker[int(round(L + t * np.sin(a))), int(round(L + t * np.cos(a)))] = 1
    return ker / ker.sum()


def degrade(page: np.ndarray, p: AugParams, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Retourne (photo, H_page_to_photo)."""
    page = page.copy()
    ph, pw = page.shape[:2]
    # 1. papier
    if p.fold:
        x = int(rng.uniform(0.2, 0.8) * pw) if rng.random() < 0.5 else None
        ov = np.zeros((ph, pw), np.float32)
        if x is not None:
            cv2.line(ov, (x, 0), (x + int(rng.uniform(-30, 30)), ph), 1.0, 3)
        else:
            y = int(rng.uniform(0.2, 0.8) * ph)
            cv2.line(ov, (0, y), (pw, y + int(rng.uniform(-30, 30))), 1.0, 3)
        ov = cv2.GaussianBlur(ov, (0, 0), 6)
        page = np.clip(page.astype(np.float32) - ov[..., None] * 40, 0, 255).astype(np.uint8)
    # 2-3. géométrie + fond
    H = _page_homography(rng, p, pw, ph)
    bg = _background(rng, p.out_h, p.out_w)
    warped = cv2.warpPerspective(page, H, (p.out_w, p.out_h), flags=cv2.INTER_LINEAR)
    mask = cv2.warpPerspective(np.full((ph, pw), 255, np.uint8), H, (p.out_w, p.out_h))
    mask_f = (cv2.GaussianBlur(mask, (3, 3), 0).astype(np.float32) / 255)[..., None]
    img = (warped * mask_f + bg * (1 - mask_f)).astype(np.float32)
    # 4. lumière
    yy, xx = np.mgrid[0:p.out_h, 0:p.out_w].astype(np.float32)
    ang = rng.uniform(0, 2 * np.pi)
    grad = (np.cos(ang) * xx / p.out_w + np.sin(ang) * yy / p.out_h)
    grad = 1 - rng.uniform(0.05, 0.35) * (grad - grad.min()) / (np.ptp(grad) + 1e-6)
    img *= grad[..., None]
    if p.shadow:
        pts = rng.uniform([0, 0], [p.out_w, p.out_h], (int(rng.integers(3, 6)), 2)).astype(np.int32)
        sh = np.zeros((p.out_h, p.out_w), np.float32)
        cv2.fillPoly(sh, [cv2.convexHull(pts)], 1.0)
        sh = cv2.GaussianBlur(sh, (0, 0), rng.uniform(15, 60))
        img *= (1 - rng.uniform(0.3, 0.6) * sh)[..., None]
    if p.glare:
        c = (int(rng.uniform(0.2, 0.8) * p.out_w), int(rng.uniform(0.2, 0.8) * p.out_h))
        g = np.zeros((p.out_h, p.out_w), np.float32)
        cv2.ellipse(g, c, (int(rng.uniform(60, 220)), int(rng.uniform(30, 120))), rng.uniform(0, 180), 0, 360, 1.0, -1)
        g = cv2.GaussianBlur(g, (0, 0), 25)
        img = img + g[..., None] * 255 * rng.uniform(0.6, 1.0)
    img *= np.array(p.color_cast, np.float32)
    img = 255 * (np.clip(img, 0, 255) / 255) ** p.gamma
    # 5. capteur
    if p.defocus_px > 0.3:
        img = cv2.filter2D(img, -1, _disk_kernel(p.defocus_px))
    if p.motion_px > 0.8:
        img = cv2.filter2D(img, -1, _motion_kernel(p.motion_px, p.motion_angle))
    img += rng.normal(0, p.noise_sigma, img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, p.jpeg_quality])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR), H


def generate(clean_dir: str, out_dir: str, per_page: int = 3, seed: int = 0,
             severities=("leger", "moyen", "fort"), patients: list[int] | None = None,
             pattern: str = "*.png") -> int:
    rng = np.random.default_rng(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for png in sorted(Path(clean_dir).glob(pattern)):
        side = png.with_suffix(".json")
        meta = json.loads(side.read_text()) if side.exists() else {"patient": None, "page_type": None}
        if patients and meta.get("patient") not in patients:
            continue
        page = cv2.imread(str(png))
        for k in range(per_page):
            sev = severities[k % len(severities)]
            p = sample_params(rng, sev)
            img, H = degrade(page, p, rng)
            name = f"{png.stem}_aug{k:02d}_{sev}"
            cv2.imwrite(str(out / f"{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            (out / f"{name}.json").write_text(json.dumps({
                **meta, "source": png.name, "H_page_to_photo": H.tolist(), "params": asdict(p)}))
            n += 1
    return n


def blur_ladder(page: np.ndarray, radii=(0, 0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 6)) -> list[tuple[float, np.ndarray]]:
    """Même page floutée à des niveaux croissants : sert à calibrer les seuils de netteté."""
    out = []
    for r in radii:
        img = page if r <= 0.3 else cv2.filter2D(page, -1, _disk_kernel(r))
        out.append((r, img))
    return out
