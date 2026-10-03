"""Export analytique : une ligne par dossier, colonnes alignées sur le CSV synthétique de 200 lignes.

Seuls les champs CONNU (ou validés par la sage-femme) alimentent l'export : une valeur douteuse
devient vide plutôt que fausse. La source de chaque colonne se lit dans `row()`.
Remarque : le formulaire contient Ag HBs (hépatite B) et non l'hépatite C du CSV ; la colonne
« hepatitis c test result » reste donc vide, c'est assumé et documenté.
"""
from __future__ import annotations

import re
import statistics

from .rules import checked, num
from .schema import Dossier, Statut

OK = {Statut.CONNU}
EDU = {"aucun": 0, "analphabete": 0, "primaire": 0, "college": 1, "lycee": 1, "secondaire": 1, "superieur": 2}


def _c(d: Dossier, pt: str, k: str):
    c = d.pages.get(pt).champs.get(k) if pt in d.pages else None
    return c if c is not None and (c.statut in OK or c.valide_par) else None


def _vals(d: Dossier, pt: str, prefix: str):
    page = d.pages.get(pt)
    if not page:
        return []
    return [c for k, c in sorted(page.champs.items()) if k.startswith(prefix) and (c.statut in OK or c.valide_par)]


def _posneg(c) -> int | None:
    if c is None or not isinstance(c.valeur, str):
        return None
    v = c.valeur.lower()
    return 1 if v.startswith("pos") else 0 if v.startswith("neg") else None


def row(d: Dossier) -> dict:
    from .textutil import strip_accents
    r: dict = {"dossier_id": d.dossier_id}
    r["age (years)"] = num(_c(d, "identification", "age"))
    edu = _c(d, "identification", "niveau_d_instruction")
    r["education level"] = EDU.get(strip_accents(str(edu.valeur)).lower()) if edu else None
    for col, key in (("consanguinity", "consanguinite"), ("desired pregnancy", "grossesse_desiree")):
        c = _c(d, "identification", key)
        r[col] = int(checked(c)) if c else None
    for col, key in (("gravidity", "gestation"), ("parity", "parite"), ("living children", "nombre_d_enfants_vivants"),
                     ("abortions", "avortement__nombre")):
        r[col] = num(_c(d, "identification", key))
    modes = [c for c in _vals(d, "identification", "modalite_d_extraction__")]
    r["previous cesarean"] = int(any("sar" in str(c.valeur).lower() for c in modes)) if modes else None

    taille = num(_c(d, "grossesse_actuelle", "taille"))
    poids = [num(c) for c in _vals(d, "grossesse_actuelle", "poids_kg__")]
    r["bmi (approx., 1st visit)"] = round(poids[0] / (taille / 100) ** 2, 2) if poids and taille else None
    tas = [c.valeur for c in _vals(d, "grossesse_actuelle", "ta__") if isinstance(c.valeur, dict)]
    r["mean systolic bp"] = round(statistics.mean(t["sys"] for t in tas), 1) if tas else None
    r["mean diastolic bp"] = round(statistics.mean(t["dia"] for t in tas), 1) if tas else None
    hb = [num(c) for c in _vals(d, "grossesse_actuelle", "hemoglobine__")]
    r["hemoglobin (g/dl)"] = hb[0] if hb else None
    gl = [num(c) for c in _vals(d, "grossesse_actuelle", "bilan_glycemique__")]
    r["first fasting glucose (mg/dl)"] = round(gl[0] * 100, 1) if gl and gl[0] < 5 else (gl[0] if gl else None)
    alb = [_posneg(c) for c in _vals(d, "grossesse_actuelle", "albuminurie__")]
    r["proteinuria"] = int(any(a == 1 for a in alb)) if any(a is not None for a in alb) else None
    for col, pre in (("hiv test result", "serologie_vih__"), ("syphilis test result", "syphilis_tpha_vdrl__"),
                     ("hepatitis b (Ag HBs)", "ag_hbs__")):
        v = [_posneg(c) for c in _vals(d, "grossesse_actuelle", pre)]
        r[col] = int(any(x == 1 for x in v)) if any(x is not None for x in v) else None
    r["hepatitis c test result"] = None
    sa = [num(c) for c in _vals(d, "grossesse_actuelle", "age_probable__")]
    r["gestational age at enrollment (weeks)"] = min(sa) if sa else None

    ga = num(_c(d, "accouchement", "age_gestationnel"))
    r["gestational age at birth (weeks)"] = ga
    r["preterm birth"] = int(ga < 37) if ga is not None else None
    ces = [_c(d, "accouchement", k) for k in ("cesarienne_programmee", "urgence")]
    vb = [_c(d, "accouchement", k) for k in ("voie_basse_non_instrumen", "voie_basse_instrumentale")]
    if any(c is not None for c in ces + vb):
        r["type of delivery (0=vaginal,1=cesarean)"] = int(any(checked(c) for c in ces))
    sexe = _c(d, "accouchement", "sexe")
    r["newborn sex (0=female,1=male)"] = {"F": 0, "M": 1}.get(str(sexe.valeur).strip().upper()[:1]) if sexe else None
    r["child birth weight (g)"] = num(_c(d, "accouchement", "poids_a_la_naissance"))
    r["head circumference (cm)"] = num(_c(d, "accouchement", "perimetre_cranien_a_la_naissance"))
    allait = [_c(d, "pp_precoce_nne", k) for k in ("exclusivement_au_sein", "mixte")]
    r["breastfeeding initiated"] = int(any(checked(c) for c in allait)) if any(c is not None for c in allait) else None
    tr = [_c(d, pt, "transfert") for pt in ("pp_precoce_nne", "pp_tardif_nne")]
    r["referral to higher care"] = int(any(checked(c) for c in tr)) if any(c is not None for c in tr) else None
    return {k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()}

