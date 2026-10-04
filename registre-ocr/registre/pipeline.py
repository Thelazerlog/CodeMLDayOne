"""Pipeline photo(s) -> dossier structuré.

Pour chaque photo :
  1. empreinte SHA-256 (l'original n'est jamais modifié)
  2. contrôle qualité (flou, lumière, reflets, cadrage)        -> raisons de reprise
  3. classification du type de page + alignement par repères   -> image redressée, zones visibles
  4. MASQUAGE des identifiants sur l'image redressée            -> plus aucun nom / CIN / téléphone en aval
  5. cases à cocher : OpenCV (sans modèle)
  6. zones texte : mesure d'encre ; seules les zones encrées sont envoyées au VLM local (mosaïque)
  7. second lecteur sur les nombres (Tesseract), fusion des signaux -> statut + confiance
Pour une session (plusieurs photos d'un même registre) :
  8. assemblage multipage, fusion des doublons (re-photo), règles de cohérence, NON_APPLICABLE
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from . import rules
from .align import Alignment, Template, classify_and_align, load_all
from .fusion import FusionParams, decide_checkbox, decide_text
from .quality import assess
from .readers.base import ReadItem, Reading, TextReader
from .readers.ink import ink_ratio, printed_contrast, read_checkbox, visible_fraction
from .schema import Champ, Dossier, ImageCapture, PageExtraite, Provenance, Statut

HINTS = {
    "date": "date JJ/MM/AAAA",
    "bp": "tension SYS/DIA, ex. 110/70",
    "int": "nombre entier",
    "quantity": "nombre, éventuellement suivi d'une unité (kg, g, cm, SA, g/dL...)",
    "text": "texte court",
}


@dataclass
class PageResult:
    capture: ImageCapture
    page: PageExtraite | None
    alignment: Alignment | None
    debug: dict = field(default_factory=dict)


def _glare_fraction(aligned: np.ndarray, bbox) -> float:
    x0, y0, x1, y1 = [int(round(v)) for v in bbox]
    c = aligned[max(0, y0):y1, max(0, x0):x1]
    if c.size == 0:
        return 0.0
    hsv = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)
    return float(((hsv[:, :, 2] > 245) & (hsv[:, :, 1] < 40)).mean())


def mask_identifiers(aligned: np.ndarray, tpl: Template, pad: int = 6) -> np.ndarray:
    """Noircit toutes les zones d'identifiants directs du gabarit (avant toute lecture)."""
    out = aligned.copy()
    boxes = list(tpl.data["redact"]) + [f["bbox"] for f in tpl.data["fields"] if f.get("sensitive")]
    for b in boxes:
        x0, y0, x1, y1 = [int(round(v)) for v in b]
        cv2.rectangle(out, (x0 - pad, y0 - pad), (x1 + pad, y1 + pad), (0, 0, 0), -1)
    return out


def _crop(img: np.ndarray, bbox, pad: int = 4) -> np.ndarray:
    x0, y0, x1, y1 = [int(round(v)) for v in bbox]
    h, w = img.shape[:2]
    return img[max(0, y0 - pad):min(h, y1 + pad), max(0, x0 - pad):min(w, x1 + pad)].copy()


def _hint(f: dict) -> str:
    parts = [p.rstrip(" :") for p in (f.get("row") or f.get("label"), f.get("col")) if p]
    lab = " / ".join(parts) if parts else f["key"]
    return f"{lab} — {HINTS.get(f.get('type', 'text'), 'texte')}"


def process_image(img: np.ndarray, templates: list[Template], reader: TextReader | None,
                  second: TextReader | None = None, sage_femme_id: str = "SF-inconnue",
                  out_dir: str | None = None, params: FusionParams = FusionParams(),
                  read_all: bool = False, image_bytes: bytes | None = None, freeform: bool = True) -> PageResult:
    sha = hashlib.sha256(image_bytes if image_bytes is not None else img.tobytes()).hexdigest()
    cap = ImageCapture(sha256=sha, capture_le=dt.datetime.now(dt.timezone.utc), sage_femme_id=sage_femme_id)

    q = assess(img)
    cap.qualite = {"accepte": q.accept, "raisons": q.reasons, "score": round(q.score, 3),
                   **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in q.metrics.items()}}

    al, ranking = classify_and_align(img, templates)
    cap.alignement = {**al.summary(), "classement": ranking}
    if al.ok and al.classif_confidence < 0.6:
        # type de page incertain (pages jumelles) : question posée à la sage-femme, pas une baisse
        # de confiance de chaque champ
        cap.alignement["type_page_a_confirmer"] = True
    if not al.ok:
        if freeform and reader is not None and hasattr(reader, "ask"):
            return _process_freeform(img, templates, reader, cap, al)
        return PageResult(cap, None, al, {"echec": "alignement"})
    tpl = next(t for t in templates if t.page_type == al.page_type)
    cap.page_type = tpl.page_type

    masked = mask_identifiers(al.aligned, tpl)
    if out_dir:
        Path(out_dir, "masque").mkdir(parents=True, exist_ok=True)
        p = Path(out_dir, "masque", f"{cap.image_id}.jpg")
        cv2.imwrite(str(p), masked, [cv2.IMWRITE_JPEG_QUALITY, 85])
        cap.chemin_masque = str(p)

    champs: dict[str, Champ] = {}
    to_read: list[tuple[dict, dict]] = []
    for f in tpl.data["fields"]:
        if f.get("sensitive"):
            continue  # jamais lu, jamais stocké
        prov = Provenance(image_id=cap.image_id, page_type=tpl.page_type, bbox=f["bbox"],
                          methode="case_opencv" if f["kind"] == "checkbox" else "vlm")
        sig = {"visible": round(visible_fraction(al.valid_mask, f["bbox"]), 3),
               "align_conf": round(al.confidence, 3), "qualite": round(q.score, 3)}
        if f["kind"] == "checkbox":
            r = read_checkbox(masked, tpl.blank, f["bbox"])
            sig["encre_case"] = round(r.ink, 3)
            champs[f["key"]] = decide_checkbox(f["key"], r.checked, r.confidence, sig, prov, params)
            continue
        pc = printed_contrast(masked, tpl.blank, f["bbox"])
        sig["contraste_imprime"] = None if pc is None else round(pc, 1)
        sig["encre"] = round(ink_ratio(masked, tpl.blank, f["bbox"], al.valid_mask, ref_contrast=pc), 4)
        sig["reflet"] = round(_glare_fraction(al.aligned, f["bbox"]), 3)
        degraded = pc is not None and pc < params.contraste_min
        if sig["visible"] < params.visible_min:
            champs[f["key"]] = decide_text(f["key"], f.get("type", "text"), None, None, sig, prov, params)
        elif read_all or degraded or sig["encre"] >= params.encre_vide or sig["reflet"] > params.reflet_max:
            to_read.append((f, {"sig": sig, "prov": prov}))
        else:
            champs[f["key"]] = decide_text(f["key"], f.get("type", "text"), None, None, sig, prov, params)

    items = [ReadItem(f["key"], _crop(masked, f["bbox"]), _hint(f), f.get("type", "text")) for f, _ in to_read]
    readings: dict[str, Reading] = {}
    readings2: dict[str, Reading] = {}
    if reader is not None and items:
        try:
            readings = reader.read(items)
        except Exception as e:  # IA indisponible : rien n'est perdu, tout part en saisie manuelle
            cap.alignement["erreur_lecteur"] = repr(e)[:300]
    if second is not None and items:
        try:
            readings2 = second.read(items)
        except Exception as e:
            cap.alignement["erreur_second_lecteur"] = repr(e)[:300]
    for f, ctx in to_read:
        r = readings.get(f["key"])
        if r is None:
            ctx["sig"]["ia_indisponible"] = True
        champs[f["key"]] = decide_text(f["key"], f.get("type", "text"), r, readings2.get(f["key"]),
                                       ctx["sig"], ctx["prov"], params)

    rules.apply_ranges(tpl.page_type, champs)
    page = PageExtraite(page_type=tpl.page_type, image_id=cap.image_id, champs=champs)
    return PageResult(cap, page, al, {"n_lus": len(items)})


def _process_freeform(img, templates, reader, cap: ImageCapture, al) -> PageResult:
    """Aucun gabarit ne correspond : lecture de la page entière par le modèle, rangée dans notre schéma."""
    from . import freeform as ff
    try:
        pt, indices = ff.classify(reader, img, [t.page_type for t in templates])
    except Exception as e:  # IA indisponible
        cap.alignement["erreur_lecteur"] = repr(e)[:300]
        return PageResult(cap, None, al, {"echec": "alignement+ia"})
    cap.alignement.update({"mode": "libre", "type_selon_modele": pt, "indices": indices})
    if pt == "autre":
        return PageResult(cap, None, al, {"echec": "page_inconnue"})
    tpl = next(t for t in templates if t.page_type == pt)
    cap.page_type = pt
    try:
        res = ff.extract(reader, img, tpl)
    except Exception as e:  # délai dépassé, serveur arrêté... : la page reste, la sage-femme saisira
        cap.alignement["erreur_lecteur"] = repr(e)[:300]
        return PageResult(cap, None, al, {"echec": "lecture_page_entiere", "type": pt})
    champs = ff.to_champs(res, tpl, cap.image_id)
    rules.apply_ranges(pt, champs)
    # pas de copie masquée possible sans gabarit : l'original reste sous accès restreint
    return PageResult(cap, PageExtraite(page_type=pt, image_id=cap.image_id, champs=champs), al,
                      {"mode": "libre", "n_valeurs": len(champs)})


# --------------------------------------------------------------------------- session multipage
def merge_pages(old: PageExtraite, new: PageExtraite) -> PageExtraite:
    """Même page photographiée deux fois : on garde la lecture la plus sûre ; si les deux sont sûres
    et différentes, conflit -> A_REVISER (la sage-femme tranche)."""
    out = {}
    for k in set(old.champs) | set(new.champs):
        a, b = old.champs.get(k), new.champs.get(k)
        if a is None or b is None:
            out[k] = a or b
            continue
        best, other = (a, b) if a.confiance >= b.confiance else (b, a)
        if a.signaux.get("mode_libre") != b.signaux.get("mode_libre"):  # gabarit > mode libre
            best = b if a.signaux.get("mode_libre") else a
        if (a.statut is Statut.CONNU and b.statut is Statut.CONNU and a.valeur != b.valeur):
            best = best.model_copy(deep=True)
            best.statut = Statut.A_REVISER
            best.raisons.append(f"Deux photos donnent des valeurs différentes : « {a.affichage} » / « {b.affichage} ».")
        out[k] = best
    return PageExtraite(page_type=new.page_type, image_id=new.image_id, champs=out)


def process_session(paths: list[str], reader: TextReader | None, second: TextReader | None = None,
                    templates_dir: str = "templates", sage_femme_id: str = "SF-inconnue",
                    code_patiente: str | None = None, out_dir: str | None = "out",
                    params: FusionParams = FusionParams(),
                    reader_for=None) -> tuple[Dossier, list[PageResult]]:
    """`reader_for(path)` : lecteur propre à chaque photo (prioritaire sur `reader` s'il est fourni)."""
    templates = load_all(templates_dir)
    d = Dossier(sage_femme_id=sage_femme_id, code_patiente=code_patiente)
    results = []
    for p in paths:
        raw = Path(p).read_bytes()
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        r = reader_for(p) if reader_for else reader
        try:
            res = process_image(img, templates, r, second, sage_femme_id, out_dir, params, image_bytes=raw)
        except Exception as e:  # aucun enregistrement perdu : l'image est gardée, l'échec est tracé
            cap = ImageCapture(sha256=hashlib.sha256(raw).hexdigest(), capture_le=dt.datetime.now(dt.timezone.utc),
                               sage_femme_id=sage_femme_id, alignement={"erreur": repr(e)[:300]})
            res = PageResult(cap, None, None, {"echec": "exception"})
        res.capture.chemin_original = str(p)
        d.images.append(res.capture)
        results.append(res)
        if res.page is None:
            why = {"lecture_page_entiere": "lecture automatique en échec (délai dépassé ou modèle indisponible)",
                   "exception": "erreur de traitement"}.get(res.debug.get("echec"), "page non reconnue")
            d.alertes.append({"regle": "ECHEC_TRAITEMENT", "message": f"{Path(p).name} : {why}, "
                              "reprendre la photo ou saisir à la main.", "champs": [], "image_id": res.capture.image_id})
            continue
        pt = res.page.page_type
        d.pages[pt] = merge_pages(d.pages[pt], res.page) if pt in d.pages else res.page
    rules.apply_dossier_rules(d)
    from .questions import build as build_questions
    d.questions = build_questions(d, templates_dir)
    d.etat = "A_REVISER" if d.champs_a_reviser() or d.alertes or d.questions else "TRAITE_IA"
    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        Path(out_dir, f"dossier_{d.dossier_id}.json").write_text(d.model_dump_json(indent=1))
    return d, results


def to_json(d: Dossier) -> str:
    return json.dumps(json.loads(d.model_dump_json()), ensure_ascii=False, indent=1)
