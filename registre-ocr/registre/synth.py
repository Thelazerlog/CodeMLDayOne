"""Générateur de pages remplies SYNTHÉTIQUES (vérité terrain exacte, volume illimité).

Pourquoi : les 10 dossiers n'utilisent que 6 polices manuscrites. Un modèle (ou un réglage de seuils)
appris uniquement dessus apprend ces polices, pas l'écriture des sages-femmes. On remplit donc le
gabarit vierge avec :
- des valeurs plausibles par champ (dates, TA, poids, vocabulaire clinique du registre) ;
- des polices manuscrites DIFFÉRENTES de celles des 10 dossiers (dossier fonts/, à compléter) ;
- des variantes réalistes : vide, tiret, « ? », gribouillis illisible, chiffres arabes, mots arabes ;
- des cases cochées de plusieurs façons (croix, coche, case noircie) ;
- de FAUX identifiants dans les zones nom/CIN/téléphone : ils servent à tester qu'ils ne fuient jamais.

Sortie : data/synth/synth_XXXXX_<type>.png + .json (vérité terrain incluse), au même format que
data/clean, donc directement utilisable par `augment` puis `evaluate` et `finetune_dataset`.
"""
from __future__ import annotations

import datetime as dt
import json
import random
import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .align import load_all

AR_DIGITS = str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩")
AR_WORDS = {"Oui": "نعم", "Non": "لا", "RAS": "لا شيء", "Normal": "عادي", "Normales": "عادية", "Néant": "لا شيء"}
FAKE_NAMES = ["Exemple Fatima", "Fictive Amina", "Test Salma", "Demo Hanane", "Synth Rachida"]

TEXT_VOCAB = [
    (r"conjonctives", ["Normales", "Pâles", "Normales"]),
    (r"seins", ["Normaux", "Normaux", "RAS"]),
    (r"oedemes|mouvements|relance|^fer__", ["Oui", "Non", "Non"]),
    (r"glucosurie|albuminurie|syphilis|serologie_vih|ag_hbs|rai_si", ["Neg", "Neg", "Neg", "Pos +"]),
    (r"rubeole__|toxoplasmose", ["Immune", "Non immune"]),
    (r"etat_du_col", ["Fermé", "Ouvert", "Fermé"]),
    (r"presentation", ["Céphalique", "Siège"]),
    (r"bassin|speculum|squelette", ["RAS", "Normal"]),
    (r"examen_fait_par|vu_par", ["Sage-femme", "Médecin", "Infirmière", "SF"]),
    (r"modalite", ["Voie basse", "Césarienne"]),
    (r"indication", ["Souffrance fœtale", "Utérus cicatriciel", "Dystocie", "Pré-éclampsie sévère"]),
    (r"lieu", ["Maternité", "Domicile", "CSC"]),
    (r"niveau_d_instruction", ["Aucun", "Primaire", "Collège", "Lycée", "Supérieur"]),
    (r"profession", ["Femme au foyer", "Agricultrice", "Couturière", "Commerçante", "Enseignante"]),
    (r"sexe", ["F", "M"]),
    (r"decision", ["Poursuivre l'allaitement exclusif", "Revoir dans 1 mois", "RAS"]),
    (r"frottis", ["Normal", "Non fait"]),
    (r"cicatrice", ["Propre", "Propre, sèche", "Inflammatoire"]),
    (r"region", ["Souss-Massa", "Oriental", "Fès-Meknès", "Marrakech-Safi"]),
    (r"province", ["Taroudant", "Berkane", "Meknès", "Al Haouz"]),
    (r"etablissement", ["CSC Al Amal", "DR Ait Ourir", "CSU Annahda"]),
    (r"fiche", None),  # numéro de fiche : généré
    (r".*", ["RAS", "Néant", "Aucun", "RAS", "Mère", "Père", "Asthme léger", "Cycles réguliers"]),
]
NUM_RANGES = [  # (regex clé, (min, max), décimales, suffixe)
    (r"^poids_kg__", (45, 98), 1, ""), (r"^hu_cm__", (10, 38), 0, ""), (r"^bcf__", (110, 170), 0, ""),
    (r"^age_probable__|^age_gestationnel", (6, 42), 0, " SA"), (r"hemoglobine", (8, 14), 1, " g/dL"),
    (r"plaquettes", (150, 400), 0, "k"), (r"bilan_glycemique", (0.6, 1.3), 2, " g/L"),
    (r"^taille$", (145, 180), 0, " cm"), (r"^age$", (15, 45), 0, ""), (r"gestation", (1, 8), 0, ""),
    (r"parite|enfants_vivants|nombre$", (0, 6), 0, ""), (r"poids_a_la_naissance|poids_nouveau", (1800, 4500), 0, " g"),
    (r"perimetre_cranien", (30, 38), 0, " cm"), (r"temperature|^t$", (36.0, 38.5), 1, ""), (r"pouls", (60, 110), 0, ""),
]


def _num_value(key: str, rng: random.Random) -> str | None:
    for pat, (lo, hi), dec, suf in NUM_RANGES:
        if re.search(pat, key):
            v = rng.uniform(lo, hi)
            return (f"{v:.{dec}f}" if dec else str(int(round(v)))) + suf
    return None


def make_value(f: dict, ptype: str, rng: random.Random) -> tuple[str | None, str]:
    """Retourne (texte écrit, statut attendu)."""
    key, ftype = f["key"], f.get("type", "text")
    r = rng.random()
    p_empty = 0.45 if f.get("table") else 0.15
    if r < p_empty:
        return None, "NON_FOURNI"
    if f.get("table") and r < p_empty + 0.07:
        return "—", "NON_FOURNI"
    if r > 0.985:
        return "?", "INCONNU"
    if r > 0.97:
        return "@@scribble", "ILLISIBLE"
    if ftype == "date" or "date" in key or key.startswith(("rendez_vous", "venue_le", "prochain", "revenir")):
        d = dt.date(2015, 1, 1) + dt.timedelta(days=rng.randrange(0, 4400))
        fmt = rng.choice(["%d/%m/%Y", "%d/%m/%Y", "%d/%m/%y", "%d-%m-%Y"])
        return d.strftime(fmt), "CONNU"
    if ftype == "bp" or re.match(r"^ta(__|$)", key):
        s = rng.randint(90, 170)
        return f"{s}/{rng.randint(50, min(110, s - 20))}", "CONNU"
    nv = _num_value(key, rng)
    if nv is not None and ftype in ("int", "quantity", "text"):
        return nv, "CONNU"
    if re.search(r"fiche", key):
        return f"{rng.randint(2024, 2027)}-{rng.randint(100, 999)}-{rng.randint(1, 999):03d}", "CONNU"
    for pat, voc in TEXT_VOCAB:
        if voc and re.search(pat, key):
            return rng.choice(voc), "CONNU"
    return "RAS", "CONNU"


def _fit_font(path: str, text: str, box_w: int, box_h: int, rng: random.Random):
    size = int(box_h * rng.uniform(0.62, 0.85))
    while size > 10:
        font = ImageFont.truetype(path, size)
        w = font.getbbox(text)[2]
        if w <= box_w * 0.95:
            return font
        size -= 2
    return ImageFont.truetype(path, 10)


def _draw_text(page: Image.Image, text: str, bbox, font_path: str, color, rng: random.Random) -> None:
    x0, y0, x1, y1 = [int(v) for v in bbox]
    bw, bh = x1 - x0, y1 - y0
    arabic = bool(re.search(r"[؀-ۿ]", text))
    font = _fit_font(font_path, text, bw - 8, min(bh, 40), rng)
    l, t, r, b = font.getbbox(text, direction="rtl" if arabic else None)
    patch = Image.new("RGBA", (r - l + 10, b - t + 10), (0, 0, 0, 0))
    ImageDraw.Draw(patch).text((5 - l, 5 - t), text, font=font, fill=color + (255,),
                               direction="rtl" if arabic else None)
    patch = patch.rotate(rng.uniform(-4, 4), expand=True, resample=Image.BICUBIC)
    px = x0 + 4 + rng.randint(0, max(0, bw - patch.width - 8) // 3 + 1)
    py = y0 + max(0, (bh - patch.height) // 2) + rng.randint(-2, 2)
    page.alpha_composite(patch, (max(0, px), max(0, py)))


def _scribble(img: np.ndarray, bbox, color, rng):
    x0, y0, x1, y1 = [int(v) for v in bbox]
    pts = [(x0 + 6 + int((x1 - x0 - 12) * t), int((y0 + y1) / 2 + rng.uniform(-1, 1) * (y1 - y0) / 3))
           for t in np.linspace(0, 1, 14)]
    cv2.polylines(img, [np.int32(pts)], False, color[::-1], 2, cv2.LINE_AA)


def _tick(img: np.ndarray, bbox, color, rng):
    x0, y0, x1, y1 = [int(v) for v in bbox]
    c = color[::-1]
    kind = rng.random()
    j = lambda: rng.randint(-2, 2)  # noqa: E731
    if kind < 0.5:  # croix
        cv2.line(img, (x0 + j(), y0 + j()), (x1 + j(), y1 + j()), c, 2, cv2.LINE_AA)
        cv2.line(img, (x0 + j(), y1 + j()), (x1 + j(), y0 + j()), c, 2, cv2.LINE_AA)
    elif kind < 0.85:  # coche ✓ qui déborde
        cv2.polylines(img, [np.int32([(x0, (y0 + y1) // 2), ((x0 + x1) // 2, y1 + 2), (x1 + 4, y0 - 5)])],
                      False, c, 2, cv2.LINE_AA)
    else:  # case noircie
        cv2.rectangle(img, (x0 + 2, y0 + 2), (x1 - 2, y1 - 2), c, -1)


def generate(n_pages: int, templates_dir: str = "templates", fonts_dir: str = "fonts", out_dir: str = "data/synth",
             seed: int = 0, page_types: list[str] | None = None) -> int:
    rng = random.Random(seed)
    fonts = sorted(str(p) for p in Path(fonts_dir).glob("*.[ot]tf"))
    if not fonts:
        raise SystemExit("Aucune police dans fonts/ : voir README (polices manuscrites Google Fonts).")
    ar_fonts = [f for f in fonts if re.search(r"aref|amiri|ruqaa|naskh|kufi|arab", f, re.I)] or fonts
    lat_fonts = [f for f in fonts if f not in ar_fonts] or fonts
    tpls = [t for t in load_all(templates_dir) if not page_types or t.page_type in page_types]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for i in range(n_pages):
        tpl = tpls[i % len(tpls)]
        font = rng.choice(lat_fonts)
        color = rng.choice([(20, 30, 140), (20, 30, 140), (25, 25, 30), (10, 60, 40)])
        arabic_page = rng.random() < 0.12
        img = cv2.cvtColor(tpl.blank, cv2.COLOR_BGR2RGB)
        gt, ticks, scribbles = {}, [], []
        pil = Image.fromarray(img).convert("RGBA")
        for f in tpl.data["fields"]:
            if f["kind"] == "checkbox":
                on = rng.random() < 0.25
                gt[f["key"]] = {"value": on, "status": "CONNU"}
                if on:
                    ticks.append(f["bbox"])
                continue
            if f.get("sensitive"):  # faux identifiant : doit être masqué, jamais lu
                fake = rng.choice(FAKE_NAMES) if "nom" in f["key"] or f["key"] == "patiente" else \
                    f"06 {rng.randint(10, 99)} {rng.randint(10, 99)} {rng.randint(10, 99)} {rng.randint(10, 99)}"
                _draw_text(pil, fake, f["bbox"], font, color, rng)
                continue
            text, status = make_value(f, tpl.page_type, rng)
            if text == "@@scribble":
                scribbles.append(f["bbox"])
                gt[f["key"]] = {"value": None, "status": "ILLISIBLE"}
                continue
            if text is None:
                gt[f["key"]] = {"value": None, "status": status}
                continue
            fpath = font
            if arabic_page and rng.random() < 0.6:
                if text in AR_WORDS:
                    text, fpath = AR_WORDS[text], rng.choice(ar_fonts)
                elif re.fullmatch(r"[\d/.\- ]+", text):
                    text = text.translate(AR_DIGITS)
                    fpath = rng.choice(ar_fonts)
            _draw_text(pil, text, f["bbox"], fpath, color, rng)
            gt[f["key"]] = {"value": text, "status": status} if status == "CONNU" else \
                {"value": None, "status": status, "raw": text}
        img = cv2.cvtColor(np.array(pil.convert("RGB")), cv2.COLOR_RGB2BGR)
        for b in ticks:
            _tick(img, b, color, rng)
        for b in scribbles:
            _scribble(img, b, color, rng)
        name = f"synth_{i:05d}_{tpl.page_type}"
        cv2.imwrite(str(out / f"{name}.png"), img)
        (out / f"{name}.json").write_text(json.dumps({
            "patient": None, "page_type": tpl.page_type, "synthetic": True, "font": Path(font).name,
            "M_page_to_template": [[1, 0, 0], [0, 1, 0]], "gt": gt}, ensure_ascii=False))
    return n_pages
