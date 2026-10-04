"""Banc d'évaluation champ par champ.

Entrée : un dossier d'images + pour chaque image un JSON « sidecar » {patient, page_type}
(les augmentations en ont un ; pour les vraies photos du Drive, voir README « Où intervenir »).
Vérité terrain : data/gt/patient_XX.json.

Ce qui est mesuré (c'est ce que le jury regarde) :
- exactitude      : la valeur finale est-elle juste (statut vide compris), tous statuts confondus
- sans révision   : part des champs que l'agent tranche seul (CONNU, NON_FOURNI, NON_APPLICABLE, INCONNU)
- erreur silencieuse : part de ces champs tranchés seuls qui sont FAUX -> doit tendre vers 0
                       (« un agent qui ne cache jamais ses doutes »)
- utilité de la révision : parmi les champs A_REVISER/ILLISIBLE, combien étaient réellement faux
- courbe seuil -> (couverture, erreur silencieuse) pour choisir FusionParams.seuil_connu
Détail par type de champ (case, date, TA, nombre, texte), par sévérité de dégradation, par page.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from .align import load_all
from .fusion import FusionParams
from .normalize import normalize, same_value
from .pipeline import process_image
from .readers.secondary import OracleReader
from .schema import Statut

EMPTY_STATUTS = {Statut.NON_FOURNI, Statut.NON_APPLICABLE}


def field_correct(gt: dict, champ, ftype: str) -> bool:
    if gt["status"] == "NON_FOURNI":
        return champ.statut in EMPTY_STATUTS
    if gt["status"] == "INCONNU":
        return champ.statut is Statut.INCONNU
    if gt["status"] == "ILLISIBLE":  # attendu : l'agent avoue son doute
        return champ.statut in (Statut.ILLISIBLE, Statut.A_REVISER)
    if champ.statut in EMPTY_STATUTS or champ.statut in (Statut.ILLISIBLE, Statut.INCONNU):
        return False
    if ftype == "bool":
        return champ.valeur == gt["value"]
    a = normalize(str(gt["value"]), ftype)
    b = normalize(champ.texte_brut if champ.texte_brut is not None else str(champ.valeur), ftype)
    return same_value(a, b)


def run(images_dir: str, reader_factory, second=None, templates_dir: str = "templates", gt_dir: str = "data/gt",
        limit: int | None = None, out: str = "out/eval", params: FusionParams = FusionParams(),
        pattern: str = "*.json") -> dict:
    templates = load_all(templates_dir)
    rows = []
    pat = pattern if pattern.endswith(".json") else pattern + ".json"
    sidecars = sorted(Path(images_dir).glob(pat))[:limit]
    for i, sc in enumerate(sidecars):
        meta = json.loads(sc.read_text())
        img_path = next((p for p in (sc.with_suffix(e) for e in (".jpg", ".jpeg", ".png")) if p.exists()), None)
        if img_path is None:
            continue
        if "gt" in meta:  # page synthétique : vérité terrain embarquée
            gt = meta["gt"]
        else:
            gt_all = json.loads(Path(gt_dir, f"patient_{int(meta['patient']):02d}.json").read_text())
            gt = gt_all.get(meta["page_type"], {})
        img = cv2.imread(str(img_path))
        res = process_image(img, templates, reader_factory(gt), second, params=params)
        sev = meta.get("params", {}).get("severity", meta.get("severity", "reel"))
        classified = res.page is not None and res.page.page_type == meta["page_type"]
        tpl = next(t for t in templates if t.page_type == meta["page_type"])
        types = {f["key"]: f.get("type", "text") for f in tpl.data["fields"]}
        for key, g in gt.items():
            c = res.page.champs.get(key) if classified else None
            libre = classified and any(x.signaux.get("mode_libre") for x in res.page.champs.values())
            if c is None and libre:
                # mode libre : le modèle ne rend que ce qu'il voit écrit ; un champ absent n'est juste que s'il
                # était vide sur le registre
                rows.append({"image": img_path.name, "page": meta["page_type"], "sev": sev, "key": key,
                             "type": types.get(key, "text"), "statut": "NON_LU_MODE_LIBRE",
                             "ok": g["status"] == "NON_FOURNI" or (types.get(key) == "bool" and g["value"] is False),
                             "conf": 0.0, "gt_empty": g["status"] == "NON_FOURNI", "attendu": g.get("value")})
                continue
            if c is None:
                rows.append({"image": img_path.name, "page": meta["page_type"], "sev": sev, "key": key,
                             "type": types.get(key, "text"), "statut": "PAGE_NON_LUE", "ok": False, "conf": 0.0,
                             "gt_empty": g["status"] == "NON_FOURNI"})
                continue
            rows.append({"image": img_path.name, "page": meta["page_type"], "sev": sev, "key": key,
                         "type": types.get(key, "text"), "statut": c.statut.value,
                         "ok": field_correct(g, c, types.get(key, "text")), "conf": c.confiance,
                         "gt_empty": g["status"] == "NON_FOURNI", "lu": c.texte_brut,
                         "attendu": g.get("value"), "raisons": c.raisons[:2],
                         "encre": c.signaux.get("encre"), "contraste": c.signaux.get("contraste_imprime")})
        print(f"[{i + 1}/{len(sidecars)}] {img_path.name}: page={'ok' if classified else 'ERREUR'}", flush=True)
    report = summarize(rows)
    Path(out).mkdir(parents=True, exist_ok=True)
    Path(out, "rows.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False, default=str) for r in rows))
    Path(out, "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    Path(out, "report.md").write_text(to_markdown(report))
    return report


def _stats(rs: list[dict]) -> dict:
    if not rs:
        return {}
    n = len(rs)
    known = [r for r in rs if r["statut"] == "CONNU"]
    review = [r for r in rs if r["statut"] in ("A_REVISER", "ILLISIBLE", "PAGE_NON_LUE")]
    auto = [r for r in rs if r["statut"] not in ("A_REVISER", "ILLISIBLE", "PAGE_NON_LUE")]
    return {
        "n": n,
        "exactitude": round(sum(r["ok"] for r in rs) / n, 4),
        "sans_revision": round(len(auto) / n, 4),
        "erreur_silencieuse": round(sum(not r["ok"] for r in auto) / len(auto), 4) if auto else None,
        "taux_revision": round(len(review) / n, 4),
        "revision_utile": round(sum(not r["ok"] for r in review) / len(review), 4) if review else None,
    }


def summarize(rows: list[dict]) -> dict:
    by = lambda k: {v: _stats([r for r in rows if r[k] == v]) for v in sorted({r[k] for r in rows})}  # noqa: E731
    curve = []
    filled = [r for r in rows if r["statut"] in ("CONNU", "A_REVISER")]
    for t in np.linspace(0.5, 0.99, 11):
        known = [r for r in filled if r["conf"] >= t]
        curve.append({"seuil": round(float(t), 3), "couverture": round(len(known) / max(1, len(rows)), 4),
                      "erreur_silencieuse": round(sum(not r["ok"] for r in known) / max(1, len(known)), 4)})
    return {"global": _stats(rows), "par_type": by("type"), "par_severite": by("sev"), "par_page": by("page"),
            "courbe_seuil": curve,
            "pires_erreurs_silencieuses": [r for r in rows if r["statut"] == "CONNU" and not r["ok"]][:30]}


def to_markdown(rep: dict) -> str:
    lines = ["# Rapport d'évaluation", "", "| périmètre | n | exactitude | sans révision | erreur silencieuse | "
             "taux révision | révision utile |", "|---|---|---|---|---|---|---|"]

    def row(name, s):
        if s:
            lines.append(f"| {name} | {s['n']} | {s['exactitude']:.3f} | {s['sans_revision']:.3f} | "
                         f"{s['erreur_silencieuse'] if s['erreur_silencieuse'] is not None else '–'} | "
                         f"{s['taux_revision']:.3f} | {s['revision_utile'] if s['revision_utile'] is not None else '–'} |")
    row("**global**", rep["global"])
    for grp in ("par_type", "par_severite", "par_page"):
        for k, s in rep[grp].items():
            row(f"{grp.split('_')[1]} : {k}", s)
    lines += ["", "## Seuil de confiance -> couverture / erreurs silencieuses", "", "| seuil | couverture | erreur silencieuse |",
              "|---|---|---|"] + [f"| {c['seuil']} | {c['couverture']} | {c['erreur_silencieuse']} |" for c in rep["courbe_seuil"]]
    return "\n".join(lines) + "\n"


def oracle_factory(error_rate: float = 0.08):
    """Lecteur SIMULÉ (vérité terrain bruitée) : teste la plomberie, ne mesure PAS la lecture."""
    return lambda gt: OracleReader(gt, error_rate=error_rate)
