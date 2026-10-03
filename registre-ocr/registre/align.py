"""Alignement d'une photo sur un gabarit, à l'aide de REPÈRES présents sur le formulaire lui-même.

Aucun repère n'est ajouté sur le papier (zéro changement pour la sage-femme) : les repères sont
les libellés imprimés (« Poids (kg) », « TA », « Glucosurie »...), les traits de tableau et les cases.

Trois étages, du grossier au fin :
1. contour de la page (si visible)            -> redressement de perspective approximatif
2. points SIFT : photo <-> gabarit vierge     -> homographie globale (robuste à rotation, échelle, 90°)
3. repères locaux : pour chaque libellé imprimé, recherche par corrélation (NCC) dans une fenêtre
   autour de la position prédite                -> correspondances précises -> homographie finale (RANSAC)

Sorties utiles en aval :
- H (gabarit -> photo), l'image redressée dans le repère du gabarit
- les repères retrouvés / perdus, l'erreur résiduelle (px) -> confiance d'alignement
- la visibilité de chaque zone (dans le cadre ou non) -> statut « hors cadre, reprendre la photo »

La classification du type de page se fait par le même mécanisme : le gabarit qui retrouve
la plus grande part de ses repères gagne (les repères propres à chaque page, comme le titre
« PRÉCOCE » / « TARDIF », départagent les pages presque identiques).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from .quality import find_page_quad

MAX_SIDE = 1700          # px : taille de travail de la photo pour SIFT
NCC_MIN = 0.55           # score minimal pour accepter un repère
SEARCH_PAD = 28          # px (repère gabarit) : fenêtre de recherche autour de la position prédite
BLUR_SIGMAS = (0, 1.5, 3.0)
MAX_RESIDUAL = 4.0       # px : au-delà, les repères « trouvés » ne s'accordent pas -> autre formulaire / page


@dataclass
class Template:
    page_type: str
    data: dict
    blank: np.ndarray            # BGR
    gray: np.ndarray             # niveaux de gris normalisés
    kp: list = field(repr=False, default=None)
    des: np.ndarray = field(repr=False, default=None)

    @property
    def size(self) -> tuple[int, int]:
        return tuple(self.data["size_px"])  # (W, H)

    def blurred(self, sigma: float) -> np.ndarray:
        if not sigma:
            return self.gray
        cache = self.__dict__.setdefault("_blur_cache", {})
        if sigma not in cache:
            cache[sigma] = cv2.GaussianBlur(self.gray, (0, 0), sigma)
        return cache[sigma]


@dataclass
class Alignment:
    page_type: str
    ok: bool
    H: np.ndarray | None = None          # gabarit -> photo
    aligned: np.ndarray | None = None    # photo redressée dans le repère du gabarit
    valid_mask: np.ndarray | None = None  # pixels du gabarit effectivement vus sur la photo
    anchors_total: int = 0
    anchors_visible: int = 0
    anchors_found: int = 0
    residual_px: float = 99.0
    sift_inliers: int = 0
    found_ids: list = field(default_factory=list)
    anchor_scores: dict = field(default_factory=dict)  # id -> NCC
    confidence: float = 0.0          # qualité GÉOMÉTRIQUE de l'alignement
    classif_confidence: float = 1.0  # certitude sur le type de page
    method: str = ""

    def summary(self) -> dict:
        return {k: getattr(self, k) for k in ("page_type", "ok", "anchors_total", "anchors_visible", "anchors_found",
                                              "residual_px", "sift_inliers", "confidence", "classif_confidence",
                                              "method")}


def _prep(gray: np.ndarray) -> np.ndarray:
    """Niveaux de gris + égalisation locale (atténue ombres et faible lumière)."""
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)


def _to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    # le minimum des canaux fait ressortir encre ET impression sur papier coloré
    return img.min(axis=2)


_sift = None


def _sift_detector():
    global _sift
    if _sift is None:
        _sift = cv2.SIFT_create(nfeatures=6000, contrastThreshold=0.02)
    return _sift


@lru_cache(maxsize=None)
def load_template(tdir: str, page_type: str) -> Template:
    data = json.loads((Path(tdir) / f"{page_type}.json").read_text())
    blank = cv2.imread(str(Path(tdir) / f"{page_type}_blank.png"))
    gray = _prep(_to_gray(blank))
    kp, des = _sift_detector().detectAndCompute(gray, None)
    return Template(page_type, data, blank, gray, kp, des)


def load_all(tdir: str) -> list[Template]:
    return [load_template(tdir, p.stem) for p in sorted(Path(tdir).glob("*.json"))]


# --------------------------------------------------------------------------- étage 2 : SIFT
def photo_features(photo: np.ndarray):
    """Points SIFT de la photo (calculés une seule fois, réutilisés pour chaque gabarit candidat)."""
    ph, pw = photo.shape[:2]
    scale = min(1.0, MAX_SIDE / max(ph, pw))
    small = cv2.resize(photo, (round(pw * scale), round(ph * scale)), interpolation=cv2.INTER_AREA)
    kp, des = _sift_detector().detectAndCompute(_prep(_to_gray(small)), None)
    return kp, des, scale, photo.shape


def _sift_homography(feats, tpl: Template):
    kp, des, scale, shape = feats
    if des is None or tpl.des is None or len(kp) < 10:
        return None, 0
    matcher = cv2.FlannBasedMatcher({"algorithm": 1, "trees": 5}, {"checks": 64})
    pairs = matcher.knnMatch(tpl.des, des, k=2)
    good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < 0.75 * n.distance]
    if len(good) < 12:
        return None, len(good)
    src = np.float32([tpl.kp[m.queryIdx].pt for m in good])
    dst = np.float32([kp[m.trainIdx].pt for m in good]) / scale  # -> pixels de la photo d'origine
    H, inl = cv2.findHomography(src, dst, cv2.USAC_MAGSAC, 6.0 / scale, maxIters=5000, confidence=0.999)
    if not plausible(H, tpl.size, shape):
        return None, 0
    return H, int(inl.sum())


def plausible(H: np.ndarray | None, size: tuple[int, int], photo_shape) -> bool:
    """Rejette les homographies dégénérées (page retournée, écrasée, minuscule ou démesurée)."""
    if H is None or not np.isfinite(H).all() or abs(np.linalg.det(H)) < 1e-9:
        return False
    W, Hh = size
    c = cv2.perspectiveTransform(np.float32([[[0, 0]], [[W, 0]], [[W, Hh]], [[0, Hh]]]), H).reshape(4, 2)
    if not cv2.isContourConvex(c.astype(np.float32)):
        return False
    area = cv2.contourArea(c)
    ph, pw = photo_shape[:2]
    return 0.05 * ph * pw < area < 4.0 * ph * pw


# --------------------------------------------------------------------------- étage 3 : repères locaux
def _refine_with_anchors(photo: np.ndarray, tpl: Template, H: np.ndarray, iters: int = 2):
    W, Hh = tpl.size
    ph, pw = photo.shape[:2]
    anchors = tpl.data["anchors"]
    found, visible, ids, res = [], 0, [], 99.0
    scores: dict = {}
    for _ in range(iters):
        aligned = cv2.warpPerspective(photo, np.linalg.inv(H), (W, Hh), flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        ag0 = _prep(_to_gray(aligned))
        # la photo peut être floue : on compare aussi à des versions adoucies (gabarit ET photo)
        ags = {sg: (cv2.GaussianBlur(ag0, (0, 0), sg) if sg else ag0) for sg in BLUR_SIGMAS}
        src, dst, ids = [], [], []
        visible = 0
        scores = {}
        for a in anchors:
            x0, y0, x1, y1 = [int(round(v)) for v in a["bbox"]]
            x0, y0, x1, y1 = x0 - 4, y0 - 3, x1 + 4, y1 + 3
            # visible ? (les 4 coins du repère projetés dans la photo)
            corners = cv2.perspectiveTransform(np.float32([[[x0, y0]], [[x1, y0]], [[x1, y1]], [[x0, y1]]]), H)
            if not ((corners[..., 0] >= 0).all() and (corners[..., 0] < pw).all()
                    and (corners[..., 1] >= 0).all() and (corners[..., 1] < ph).all()):
                continue
            visible += 1
            sx0, sy0 = max(0, x0 - SEARCH_PAD), max(0, y0 - SEARCH_PAD)
            sx1, sy1 = min(W, x1 + SEARCH_PAD), min(Hh, y1 + SEARCH_PAD)
            best = (-1.0, None)
            for sg, ag in ags.items():
                patch = tpl.blurred(sg)[max(0, y0):y1, max(0, x0):x1]
                win = ag[sy0:sy1, sx0:sx1]
                if win.shape[0] <= patch.shape[0] or win.shape[1] <= patch.shape[1] or patch.size == 0:
                    continue
                r = cv2.matchTemplate(win, patch, cv2.TM_CCOEFF_NORMED)
                _, score, _, loc = cv2.minMaxLoc(r)
                if score > best[0]:
                    best = (score, loc)
            score, loc = best
            scores[a["id"]] = float(score)
            if score < NCC_MIN:
                continue
            dx, dy = sx0 + loc[0] - max(0, x0), sy0 + loc[1] - max(0, y0)
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            # point repère dans le gabarit -> point observé dans l'image redressée
            src += [[x0, y0], [x1, y1], [cx, cy]]
            dst += [[x0 + dx, y0 + dy], [x1 + dx, y1 + dy], [cx + dx, cy + dy]]
            ids.append(a["id"])
        if len(ids) < 4:
            return H, visible, ids, 99.0, scores
        src, dst = np.float32(src), np.float32(dst)
        Hc, inl = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
        if Hc is None or not plausible(H @ Hc, tpl.size, photo.shape):
            return H, visible, ids, 99.0, scores
        pred = cv2.perspectiveTransform(src.reshape(-1, 1, 2), Hc).reshape(-1, 2)
        res = float(np.median(np.linalg.norm(pred - dst, axis=1)))
        # Hc : gabarit -> redressée ; redressée -> photo = H  => gabarit -> photo = H @ Hc
        H = H @ Hc
        found = ids
    return H, visible, found, res, scores


# --------------------------------------------------------------------------- API
def align(photo: np.ndarray, tpl: Template, use_quad: bool = True, feats=None) -> Alignment:
    ph, pw = photo.shape[:2]
    feats = feats or photo_features(photo)
    H, inliers = _sift_homography(feats, tpl)
    method = "sift"
    if H is None and use_quad:
        quad = find_page_quad(photo)
        if quad is not None:  # repli : contour de page, orientation supposée portrait
            W, Hh = tpl.size
            H = cv2.getPerspectiveTransform(np.float32([[0, 0], [W, 0], [W, Hh], [0, Hh]]), quad)
            method = "contour"
            if not plausible(H, tpl.size, photo.shape):
                H = None
    if H is None:
        return Alignment(tpl.page_type, ok=False, method="echec")
    H, visible, found, res, scores = _refine_with_anchors(photo, tpl, H)
    W, Hh = tpl.size
    aligned = cv2.warpPerspective(photo, np.linalg.inv(H), (W, Hh), flags=cv2.INTER_CUBIC)
    valid = cv2.warpPerspective(np.full((ph, pw), 255, np.uint8), np.linalg.inv(H), (W, Hh)) > 0
    n = len(tpl.data["anchors"])
    ratio_found = len(found) / max(1, visible)
    conf = float(np.clip(ratio_found, 0, 1) * np.exp(-max(0.0, res - 1.0) / 4) * min(1.0, len(found) / 8))
    return Alignment(tpl.page_type, ok=len(found) >= 6 and res <= MAX_RESIDUAL and conf >= 0.25, H=H, aligned=aligned, valid_mask=valid,
                     anchors_total=n, anchors_visible=visible, anchors_found=len(found), residual_px=res,
                     sift_inliers=inliers, found_ids=found, anchor_scores=scores, confidence=conf,
                     method=method + "+reperes")


def classify_and_align(photo: np.ndarray, templates: list[Template], top_k: int = 3) -> tuple[Alignment, list[dict]]:
    """Essaie les gabarits, garde celui qui retrouve la plus grande part de ses repères."""
    feats = photo_features(photo)
    pre = []
    for t in templates:
        _, inl = _sift_homography(feats, t)
        pre.append((inl, t))
    pre.sort(key=lambda x: -x[0])
    # les pages jumelles (précoce/tardif) ont des scores SIFT proches : on garde le top-k
    cands = [t for inl, t in pre[:top_k] if inl > 0] or [pre[0][1]]
    results = []
    for t in cands:
        a = align(photo, t, feats=feats)
        score = a.anchors_found / max(1, a.anchors_total) + 0.5 * a.anchors_found / max(1, a.anchors_visible)
        results.append((score if a.ok else -1.0 + score / 10, a))  # un alignement incohérent passe derrière
    results.sort(key=lambda x: -x[0])
    # Pages jumelles (même mise en page, libellés différents : PRÉCOCE/TARDIF, 7e/40e jour...) :
    # on compare la corrélation sur les SEULS repères qui diffèrent entre les deux gabarits.
    if len(results) > 1:
        (s1, a1), (s2, a2) = results[0], results[1]
        t1 = {x["id"] for x in next(t for t in cands if t.page_type == a1.page_type).data["anchors"]}
        t2 = {x["id"] for x in next(t for t in cands if t.page_type == a2.page_type).data["anchors"]}
        if len(t1 & t2) / max(1, len(t1 | t2)) > 0.6:
            d1 = [a1.anchor_scores[i] for i in t1 - t2 if i in a1.anchor_scores]
            d2 = [a2.anchor_scores[i] for i in t2 - t1 if i in a2.anchor_scores]
            m1, m2 = (np.mean(d1) if d1 else 0.0), (np.mean(d2) if d2 else 0.0)
            if m2 > m1:
                results[0], results[1] = results[1], results[0]
            margin = abs(m1 - m2)
            results[0][1].classif_confidence = float(np.clip(margin / 0.04, 0.0, 1.0))
        else:
            results[0][1].classif_confidence = float(np.clip((s1 - s2) / 0.1, 0.0, 1.0))
    ranking = [{"page_type": a.page_type, "score": round(s, 3), "anchors_found": a.anchors_found,
                "anchors_visible": a.anchors_visible} for s, a in results]
    return results[0][1], ranking


def corner_error(H_est: np.ndarray, H_true: np.ndarray, size: tuple[int, int]) -> float:
    """Erreur moyenne (px photo) sur les 4 coins de la page : métrique d'évaluation de l'alignement."""
    W, Hh = size
    c = np.float32([[[0, 0]], [[W, 0]], [[W, Hh]], [[0, Hh]]])
    return float(np.linalg.norm(cv2.perspectiveTransform(c, H_est) - cv2.perspectiveTransform(c, H_true), axis=2).mean())
