"""Opérations sur le dossier extrait (forme JSON, telle que stockée chiffrée) : validations de la
sage-femme, saisie manuelle. Une validation est conservée à part : si la page est rephotographiée et
relue par l'IA, ce que la sage-femme a confirmé ou corrigé est réappliqué, jamais perdu."""
from __future__ import annotations

import datetime as dt
import json
from functools import lru_cache
from pathlib import Path

from registre.labels import field_label
from registre.normalize import normalize


@lru_cache(maxsize=None)
def champs_gabarit(templates: str, page_type: str) -> dict[str, dict]:
    p = Path(templates, f"{page_type}.json")
    return {f["key"]: f for f in json.loads(p.read_text())["fields"]} if p.exists() else {}


def libelle(templates: str, page_type: str, cle: str) -> str:
    f = champs_gabarit(templates, page_type).get(cle)
    return field_label(f) if f else cle


def type_champ(templates: str, page_type: str, cle: str) -> str:
    f = champs_gabarit(templates, page_type).get(cle, {})
    return "bool" if f.get("kind") == "checkbox" else f.get("type", "text")


def champ(dossier: dict, page: str, cle: str) -> dict | None:
    return dossier.get("pages", {}).get(page, {}).get("champs", {}).get(cle)


def saisir(dossier: dict, page: str, cle: str, texte: str | None, sage_femme: str, templates: str,
           statut: str | None = None) -> tuple[bool, str]:
    """Valeur donnée par la sage-femme (correction ou saisie). Renvoie (ok, message d'erreur de format)."""
    ftype = type_champ(templates, page, cle)
    pages = dossier.setdefault("pages", {})
    pg = pages.setdefault(page, {"page_type": page, "image_id": "saisie_manuelle", "champs": {}})
    old = pg["champs"].get(cle) or {"cle": cle, "type": ftype, "statut": "A_REVISER", "raisons": [], "signaux": {}}
    c = dict(old)
    if statut is not None:  # « laisser vide » / « inconnu »
        c.update(valeur=None, affichage="", statut=statut)
    elif ftype == "bool":
        v = (texte or "").strip().lower() in ("oui", "x", "coché", "coche", "1", "vrai")
        c.update(valeur=v, affichage="☒" if v else "☐", statut="CONNU")
    else:
        n = normalize(texte, ftype)
        if not n.format_ok:
            return False, n.note or "format inattendu"
        c.update(valeur=n.value, affichage=n.display, texte_brut=texte, statut="CONNU")
    c["confiance"] = 1.0
    c["valide_par"] = sage_femme
    c["raisons"] = list(old.get("raisons", [])) + [f"Saisi par la sage-femme le {dt.date.today():%d/%m/%Y}."]
    prov = dict(old.get("provenance") or {"image_id": pg["image_id"], "page_type": page, "bbox": [0, 0, 0, 0]})
    prov["methode"] = "manuel"
    c["provenance"] = prov
    pg["champs"][cle] = c
    return True, ""


def confirmer(dossier: dict, page: str, cle: str, sage_femme: str) -> None:
    c = champ(dossier, page, cle)
    if c is None:
        return
    c["statut"] = "CONNU" if c.get("valeur") is not None else c["statut"]
    c["valide_par"] = sage_femme
    c.setdefault("raisons", []).append("Confirmé par la sage-femme.")


def memoriser(rec: dict, page: str, cle: str) -> None:
    c = champ(rec["dossier"], page, cle)
    if c is not None:
        rec.setdefault("validations", {})[f"{page}.{cle}"] = c


def reappliquer(rec: dict) -> None:
    """Après une nouvelle lecture IA : les champs déjà validés reprennent la valeur de la sage-femme et
    les questions qui les concernent disparaissent."""
    vals = rec.get("validations") or {}
    d = rec.get("dossier")
    if not vals or not d:
        return
    for ref, c in vals.items():
        page, cle = ref.split(".", 1)
        d.setdefault("pages", {}).setdefault(page, {"page_type": page, "image_id": "saisie_manuelle",
                                                     "champs": {}})["champs"][cle] = c
    d["questions"] = [q for q in d.get("questions", []) if f"{q.get('page')}.{q.get('champ')}" not in vals]


def compter(dossier: dict) -> dict:
    from collections import Counter
    return dict(Counter(c["statut"] for p in dossier.get("pages", {}).values() for c in p["champs"].values()))
