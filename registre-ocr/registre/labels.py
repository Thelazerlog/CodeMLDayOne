"""Libellés lisibles par la sage-femme : pages, colonnes de visites, champs."""
from __future__ import annotations

PAGE_LABELS = {
    "couverture": "Couverture",
    "identification": "Identification et antécédents",
    "grossesse_actuelle": "Grossesse actuelle",
    "accouchement": "Accouchement",
    "pp_precoce_mere": "Post-partum précoce (mère)",
    "pp_precoce_nne": "Post-partum précoce (nouveau-né)",
    "pp_tardif_mere": "Post-partum tardif (mère)",
    "pp_tardif_nne": "Post-partum tardif (nouveau-né)",
}

# Ce que contient chaque page : sert à classer une photo quand la mise en page est inconnue
PAGE_CONTENT = {
    "couverture": "couverture : n° de fiche, région, province, établissement, type d'établissement, grossesse à risque",
    "identification": "identification et antécédents : âge, instruction, antécédents familiaux et de la femme, "
                      "anomalies des grossesses antérieures, accouchements antérieurs, gestation/parité, VAT, rubéole",
    "grossesse_actuelle": "grossesse actuelle : DDR, groupage, tableau des visites par trimestre (poids, TA, HU, "
                          "BCF, examens biologiques)",
    "accouchement": "déroulement de l'accouchement : lieu, date, mode, complications, état du nouveau-né",
    "pp_precoce_mere": "consultation du post-partum PRÉCOCE pour la MÈRE (7e-8e jour)",
    "pp_precoce_nne": "consultation du post-partum PRÉCOCE pour le NOUVEAU-NÉ (7e-8e jour)",
    "pp_tardif_mere": "consultation du post-partum TARDIF pour la MÈRE (après 6 semaines)",
    "pp_tardif_nne": "consultation du post-partum TARDIF pour le NOUVEAU-NÉ (après 6 semaines)",
}

COLUMN_LABELS = {
    "t1_v1": "1er trimestre, visite 1", "t1_v2": "1er trimestre, visite 2", "t1_v3": "1er trimestre, visite 3",
    "t2_v1": "2e trimestre, visite 1", "t2_v2": "2e trimestre, visite 2", "t2_v3": "2e trimestre, visite 3",
    "t3_m7": "7e mois", "t3_m8": "8e mois", "t3_m9": "9e mois",
}

KEY_LABELS = {
    "date_vacc_rubeole": "Date du vaccin contre la rubéole",
    "date_vacc_hepatite_b": "Date du vaccin contre l'hépatite B",
    "rh_neg": "Rhésus négatif", "rh_pos": "Rhésus positif",
    "a": "Groupe A", "b": "Groupe B", "o": "Groupe O", "ab": "Groupe AB",
    "vat_1": "VAT 1", "vat_2": "VAT 2", "vat_3": "VAT 3", "vat_4": "VAT 4", "vat_5": "VAT 5",
    "temperature": "Température", "pf_motif_refus": "Motif du refus de contraception",
    "signes_graves_autres": "Autres signes de gravité", "traumatismes_autres": "Autres traumatismes / malformations",
}


def field_label(field: dict) -> str:
    key = field["key"]
    if key in KEY_LABELS:
        return KEY_LABELS[key]
    row, col = field.get("row"), field.get("col")
    if row:
        suffix = key.split("__", 1)[1] if "__" in key else ""
        col_txt = COLUMN_LABELS.get(suffix, col or suffix)
        return f"{row.rstrip(' :')} ({col_txt})"
    lab = (field.get("label") or col or key).rstrip(" :")
    return lab.capitalize() if lab.isupper() else lab
