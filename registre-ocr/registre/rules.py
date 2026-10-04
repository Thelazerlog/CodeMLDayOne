"""Règles de validation : elles SIGNALENT, elles ne corrigent jamais.

1. Bornes physiologiques par champ (une valeur hors bornes passe en A_REVISER).
2. Cohérence entre champs d'une page et entre pages (alertes au niveau du dossier,
   chaque champ impliqué passe en A_REVISER avec la raison).
3. Déductions NON_APPLICABLE (ex. indication de césarienne quand l'accouchement est par voie basse).

Aucune règle ne fait de diagnostic ni de triage : hors périmètre du défi.
"""
from __future__ import annotations

import datetime as dt
import re

from .schema import Champ, Dossier, Statut

# (regex type de page, regex clé) -> (min, max) sur la valeur numérique
RANGES = [
    (r"identification", r"^age$", (12, 55)),
    (r"identification", r"^(gestation)$", (1, 20)),
    (r"identification", r"^(parite|nombre_d_enfants_vivants)$", (0, 20)),
    (r"grossesse_actuelle", r"^poids_kg__", (30, 200)),
    (r"grossesse_actuelle", r"^hu_cm__", (4, 45)),
    (r"grossesse_actuelle", r"^bcf__", (100, 180)),
    (r"grossesse_actuelle", r"^hemoglobine__", (4, 20)),
    (r"grossesse_actuelle", r"^age_probable__", (4, 44)),
    (r"grossesse_actuelle", r"^taille$", (120, 200)),
    (r"accouchement", r"^poids_a_la_naissance$", (400, 6500)),
    (r"accouchement", r"^perimetre_cranien", (20, 45)),
    (r"accouchement", r"^age_gestationnel$", (20, 45)),
    (r"_mere$", r"^temperature$", (34, 42)),
    (r"_mere$", r"^pouls$", (40, 160)),
    (r"_mere$", r"^poids$", (30, 200)),
    (r"_nne$", r"^temperature$", (34, 42)),
    (r"_nne$", r"^poids$", (400, 9000)),
    (r"_nne$", r"^(taille)$", (30, 75)),
    (r"_nne$", r"^perimetre_cranien$", (20, 50)),
]


# (regex type de page, regex clé) -> unités acceptées ("" = pas d'unité écrite)
UNITS = [
    (r"grossesse_actuelle", r"^(hu_cm__|taille$)", {"", "cm"}),
    (r"grossesse_actuelle", r"^poids_kg__", {"", "kg"}),
    (r"grossesse_actuelle", r"^age_probable__", {"", "SA"}),
    (r"grossesse_actuelle", r"^hemoglobine__", {"", "g/dL"}),
    (r"accouchement", r"^perimetre_cranien", {"", "cm"}),
    (r"accouchement", r"^poids_a_la_naissance$", {"", "g", "kg"}),
    (r"accouchement", r"^age_gestationnel$", {"", "SA"}),
    (r"_nne$", r"^(taille|perimetre_cranien)$", {"", "cm"}),
    (r"_nne$", r"^poids$", {"", "g", "kg"}),
    (r"_mere$", r"^poids$", {"", "kg"}),
]


def _lookup(table, page_type: str, key: str):
    for pt_re, key_re, val in table:
        if re.search(pt_re, page_type) and re.search(key_re, key):
            return val
    return None


def bounds_for(page_type: str, key: str):
    return _lookup(RANGES, page_type, key)


def plausibility(page_type: str, key: str, value) -> str | None:
    """None si la valeur est plausible, sinon la raison (en clair, pour la question à la sage-femme).
    `value` : valeur normalisée (dict {value, unit}, nombre, {sys, dia}) ou texte brut."""
    lo_hi = bounds_for(page_type, key)
    units = _lookup(UNITS, page_type, key)
    if lo_hi is None and units is None:
        return None
    if isinstance(value, dict) and "value" in value:
        x, unit = float(value["value"]), value.get("unit", "")
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        x, unit = float(value), ""
    else:
        from .normalize import normalize
        n = normalize(str(value), "quantity")
        if not isinstance(n.value, dict):
            return None
        x, unit = float(n.value["value"]), n.value.get("unit", "")
    if units is not None and unit not in units:
        attendu = " ou ".join(sorted(u for u in units if u)) or "sans unité"
        return f"Unité inattendue « {unit} » (attendu : {attendu})."
    if lo_hi is not None:
        lo, hi = lo_hi
        if unit == "kg" and lo >= 100:  # poids du nouveau-né écrit en kg : on compare en grammes
            x *= 1000
        if not lo <= x <= hi:
            return f"Valeur {x:g} hors des bornes plausibles [{lo}–{hi}] : à confirmer."
    return None


def num(c: Champ | None) -> float | None:
    if c is None or c.valeur is None or c.statut not in (Statut.CONNU, Statut.A_REVISER):
        return None
    v = c.valeur
    if isinstance(v, dict) and "value" in v:
        return float(v["value"])
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v)
    m = re.search(r"\d+(?:[.,]\d+)?", str(v))
    return float(m.group().replace(",", ".")) if m else None


def date(c: Champ | None) -> dt.date | None:
    if c is None or not isinstance(c.valeur, str):
        return None
    try:
        return dt.date.fromisoformat(c.valeur)
    except ValueError:
        return None


def checked(c: Champ | None) -> bool:
    return bool(c is not None and c.valeur is True)


def _flag(c: Champ | None, raison: str) -> None:
    if c is None:
        return
    if raison not in c.raisons:
        c.raisons.append(raison)
    if c.statut is Statut.CONNU:
        c.statut = Statut.A_REVISER


def apply_ranges(page_type: str, champs: dict[str, Champ]) -> None:
    for k, c in champs.items():
        if c.valeur is None or c.statut not in (Statut.CONNU, Statut.A_REVISER) or c.type == "bool":
            continue
        why = plausibility(page_type, k, c.valeur)
        if why:
            c.signaux["valeur_aberrante"] = True
            _flag(c, why)
    for k, c in champs.items():  # TA : bornes déjà vérifiées à la normalisation
        if c.type == "bp" and isinstance(c.valeur, dict) and c.valeur.get("sys", 0) <= c.valeur.get("dia", 0):
            _flag(c, "Systolique ≤ diastolique : inversion ou erreur de lecture ?")
    apply_page_dates(page_type, champs)


VISITES = ["t1_v1", "t1_v2", "t1_v3", "t2_v1", "t2_v2", "t2_v3", "t3_m7", "t3_m8", "t3_m9"]
RDV_MAX_JOURS = 120  # un rendez-vous de suivi se donne en jours ou semaines, pas en années


def apply_page_dates(page_type: str, champs: dict[str, Champ]) -> None:
    """Chronologie des dates d'une même page. Sur photo dégradée, une année mal lue (2024 pour 2026)
    garde un format valide et une forte confiance : seule la cohérence entre dates la trahit."""
    g = champs.get

    def check(ok: bool, raison: str, *cles: str):
        if not ok:
            for k in cles:
                _flag(g(k), raison)

    if page_type == "grossesse_actuelle":
        ddr, prev = date(g("ddr")), None
        for col in VISITES:
            v_k, r_k = f"venue_le__{col}", f"rendez_vous__{col}"
            v, rdv = date(g(v_k)), date(g(r_k))
            if v and rdv:
                check(0 < (rdv - v).days <= RDV_MAX_JOURS,
                      f"Rendez-vous à {(rdv - v).days:+d} j de la visite : date ou année mal lue ?", v_k, r_k)
            if v and ddr:
                check(0 <= (v - ddr).days <= 44 * 7, "Visite hors de la grossesse d'après la DDR : date mal lue ?",
                      v_k, "ddr")
            if v and prev:
                check(v > prev[1], "Visites hors d'ordre chronologique : date mal lue ?", v_k, prev[0])
            if v:
                prev = (v_k, v)
    elif page_type.startswith("pp_"):
        dc, rdv = date(g("date_de_la_consultation")), date(g("prochain_rendez_vous_le"))
        if dc and rdv:
            check(0 < (rdv - dc).days <= RDV_MAX_JOURS,
                  f"Prochain rendez-vous à {(rdv - dc).days:+d} j de la consultation : date mal lue ?",
                  "date_de_la_consultation", "prochain_rendez_vous_le")


def apply_dossier_rules(d: Dossier) -> None:
    P = {pt: p.champs for pt, p in d.pages.items()}
    g = lambda pt, k: P.get(pt, {}).get(k)  # noqa: E731

    def alert(code: str, msg: str, fields: list[tuple[str, str]]):
        d.alertes.append({"regle": code, "message": msg, "champs": [f"{pt}.{k}" for pt, k in fields]})
        for pt, k in fields:
            _flag(g(pt, k), msg)

    # --- histoire obstétricale : gestité >= parité + avortements
    G, Pa = num(g("identification", "gestation")), num(g("identification", "parite"))
    ab = num(g("identification", "avortement__nombre")) or 0
    if G is not None and Pa is not None:
        enceinte = 1 if "grossesse_actuelle" in P or "accouchement" in P else 0
        if G < Pa + ab + enceinte:
            alert("G_P_AB", f"Gestité ({G:g}) < parité ({Pa:g}) + avortements ({ab:g}) + grossesse en cours.",
                  [("identification", "gestation"), ("identification", "parite")])
        if G > Pa + ab + 1 + 0.5:  # grossesses non expliquées
            alert("G_P_AB_MANQUE", f"Gestité ({G:g}) > parité + avortements + 1 : antécédent manquant ?",
                  [("identification", "gestation"), ("identification", "avortement__nombre")])
    ev = num(g("identification", "nombre_d_enfants_vivants"))
    if ev is not None and Pa is not None and ev > Pa + 1:
        alert("ENFANTS_VIVANTS", "Plus d'enfants vivants que d'accouchements.",
              [("identification", "nombre_d_enfants_vivants"), ("identification", "parite")])
    ag = num(g("identification", "avortement__age_gestationnel_sa"))
    if ag is not None and ag >= 22:
        alert("AVORTEMENT_TERME", f"« Avortement » à {ag:g} SA : au-delà de 22 SA, accouchement prématuré / MFIU ?",
              [("identification", "avortement__age_gestationnel_sa")])

    # --- datation
    ddr, dpa = date(g("grossesse_actuelle", "ddr")), date(g("grossesse_actuelle", "date_prevue_d_accouchement"))
    dep = date(g("grossesse_actuelle", "date_de_depassement_de_terme"))
    if ddr and dpa and abs((dpa - ddr).days - 280) > 3:
        alert("DPA", f"DPA ≠ DDR + 280 j (écart {(dpa - ddr).days - 280:+d} j).",
              [("grossesse_actuelle", "ddr"), ("grossesse_actuelle", "date_prevue_d_accouchement")])
    if dpa and dep and abs((dep - dpa).days - 7) > 2:
        alert("TERME", "Date de dépassement ≠ DPA + 7 j.", [("grossesse_actuelle", "date_de_depassement_de_terme")])
    if ddr:
        for k, c in P.get("grossesse_actuelle", {}).items():
            if k.startswith("venue_le__"):
                col = k.split("__", 1)[1]
                v, sa = date(c), num(g("grossesse_actuelle", f"age_probable__{col}"))
                if v and sa is not None and abs((v - ddr).days / 7 - sa) > 2.5:
                    alert("SA_VISITE", f"Âge gestationnel {sa:g} SA incohérent avec la date de visite "
                                       f"({(v - ddr).days / 7:.1f} SA d'après la DDR).",
                          [("grossesse_actuelle", k), ("grossesse_actuelle", f"age_probable__{col}")])
    d_acc = date(g("accouchement", "date_de_l_accouchement"))
    sa_acc = num(g("accouchement", "age_gestationnel"))
    if ddr and d_acc and sa_acc is not None and abs((d_acc - ddr).days / 7 - sa_acc) > 2.5:
        alert("SA_NAISSANCE", "Âge gestationnel à la naissance incohérent avec DDR et date d'accouchement.",
              [("accouchement", "age_gestationnel"), ("accouchement", "date_de_l_accouchement")])

    # --- accouchement : mode
    acc = P.get("accouchement", {})
    if acc:
        modes = ["voie_basse_non_instrumen", "voie_basse_instrumentale", "cesarienne_programmee", "urgence"]
        n_modes = sum(checked(acc.get(m)) for m in modes)
        voie_basse = checked(acc.get("voie_basse_non_instrumen")) or checked(acc.get("voie_basse_instrumentale"))
        cesar = checked(acc.get("cesarienne_programmee")) or checked(acc.get("urgence"))
        if n_modes == 0 or (voie_basse and cesar):
            alert("MODE_ACC", "Mode d'accouchement absent ou contradictoire.", [("accouchement", m) for m in modes])
        ind = acc.get("preciser_l_indication")
        if ind is not None and voie_basse and not cesar and ind.statut is Statut.NON_FOURNI:
            ind.statut, ind.raisons = Statut.NON_APPLICABLE, ["Pas de césarienne : indication sans objet."]
        if ind is not None and isinstance(ind.valeur, str) and "cicatriciel" in ind.valeur.lower():
            prev = [c for k, c in P.get("identification", {}).items()
                    if k.startswith("modalite_d_extraction__") and isinstance(c.valeur, str)]
            if prev and not any("sar" in c.valeur.lower() for c in prev):
                alert("UTERUS_CICATRICIEL", "Indication « utérus cicatriciel » sans césarienne antérieure notée.",
                      [("accouchement", "preciser_l_indication")])

    # --- RAI sans objet si Rh+
    if checked(g("grossesse_actuelle", "rh_pos")):
        for k, c in P.get("grossesse_actuelle", {}).items():
            if k.startswith("rai_si_rh_negatif__") and c.statut is Statut.NON_FOURNI:
                c.statut, c.raisons = Statut.NON_APPLICABLE, ["Rh positif : RAI sans objet."]

    # --- post-partum : délais et cicatrice
    for pt, (lo, hi) in {"pp_precoce_mere": (4, 14), "pp_tardif_mere": (30, 70),
                         "pp_precoce_nne": (4, 14), "pp_tardif_nne": (30, 70)}.items():
        dc = date(g(pt, "date_de_la_consultation"))
        if d_acc and dc and not (lo <= (dc - d_acc).days <= hi):
            alert("DELAI_PP", f"{pt} : consultation à J{(dc - d_acc).days} de l'accouchement (attendu J{lo}–J{hi}).",
                  [(pt, "date_de_la_consultation")])
        if d_acc and dc and dc < d_acc:
            alert("CHRONO", f"{pt} : consultation avant l'accouchement.", [(pt, "date_de_la_consultation")])
    if acc and not (checked(acc.get("cesarienne_programmee")) or checked(acc.get("urgence"))):
        for pt in ("pp_precoce_mere", "pp_tardif_mere"):
            c = g(pt, "etat_de_la_cicatrice")
            if c is not None and c.statut is Statut.NON_FOURNI:
                c.statut, c.raisons = Statut.NON_APPLICABLE, ["Pas de césarienne : cicatrice sans objet."]
