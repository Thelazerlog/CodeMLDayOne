"""Questions de suivi à poser à la sage-femme (consommées par la couche conversationnelle).

Chaque question est un petit objet autonome :
    {id, priorite, type, page, champ, texte, valeur_lue, options}
- type « reprendre_photo »   : photo refusée par le contrôle qualité, ou page non reconnue
- type « type_page »         : pages jumelles (précoce / tardif) mal départagées
- type « coherence »         : règle violée entre plusieurs champs (G/P, DPA, délais...)
- type « confirmer »         : valeur lue mais douteuse (raison donnée), ou valeur aberrante
- type « saisir »            : illisible, hors cadre, IA indisponible : rien n'a été lu
- type « confirmer_page »    : page lue en mode libre -> une confirmation groupée de tout ce qui a été lu
- type « manquant »          : champ important non trouvé sur une page lue en mode libre
- type « page_manquante »    : page du registre pas encore photographiée (priorité basse)

L'ordre (priorité) suit le coût d'une erreur : d'abord ce qui invalide une page entière, puis les
incohérences, puis les champs importants, puis le reste.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

from .labels import PAGE_LABELS, field_label
from .schema import Dossier, Statut

# Champs dont l'absence ou le doute mérite une question en priorité (variables clés du défi)
ESSENTIELS = [
    r"^(age|gestation|parite|nombre_d_enfants_vivants)$",
    r"^(ddr|date_prevue_d_accouchement)$",
    r"^(ta|poids_kg|age_probable|serologie_vih|syphilis_tpha_vdrl|ag_hbs|hemoglobine)__",
    r"^(date_de_l_accouchement|sexe|poids_a_la_naissance|perimetre_cranien_a_la_naissance|age_gestationnel)$",
    r"^(temperature|ta|poids|date_de_la_consultation)$",
]
OPTS_CONFIRMER = ["Confirmer", "Corriger", "Reprendre la photo"]
OPTS_SAISIR = ["Saisir la valeur", "Reprendre la photo", "Laisser vide"]


def essentiel(key: str) -> bool:
    return any(re.search(p, key) for p in ESSENTIELS)


@lru_cache(maxsize=None)
def _labels(templates_dir: str, page_type: str) -> dict[str, str]:
    p = Path(templates_dir, f"{page_type}.json")
    if not p.exists():
        return {}
    return {f["key"]: field_label(f) for f in json.loads(p.read_text())["fields"]}


def build(d: Dossier, templates_dir: str = "templates", pages_attendues: list[str] | None = None) -> list[dict]:
    qs: list[dict] = []

    def add(prio, typ, texte, page=None, champ=None, valeur=None, options=None):
        qs.append({"id": f"q{len(qs) + 1}", "priorite": prio, "type": typ, "page": page, "champ": champ,
                   "texte": texte, "valeur_lue": valeur, "options": options or []})

    # 1. photos
    for im in d.images:
        q = im.qualite or {}
        if not q.get("accepte", True):
            add(1, "reprendre_photo", "La photo de la page " + PAGE_LABELS.get(im.page_type or "", "?") +
                " n'est pas assez bonne : " + " ".join(q.get("raisons", [])), page=im.page_type,
                options=["Reprendre la photo", "Continuer quand même"])
        if im.page_type is None:
            add(1, "reprendre_photo", "Je ne reconnais pas cette page. Pouvez-vous la reprendre en photo, "
                "bien à plat, avec les 4 coins visibles ?", options=["Reprendre la photo", "Saisir à la main"])
        elif (im.alignement or {}).get("type_page_a_confirmer"):
            add(2, "type_page", f"Cette page est-elle bien « {PAGE_LABELS.get(im.page_type, im.page_type)} » ?",
                page=im.page_type, options=["Oui", "Non, c'est une autre page"])

    # 2. incohérences
    for a in d.alertes:
        if a.get("regle") == "ECHEC_TRAITEMENT":
            if "page non reconnue" not in a["message"]:  # la page non reconnue a déjà sa question (plus haut)
                add(1, "saisir", a["message"], options=["Réessayer", "Reprendre la photo", "Saisir à la main"])
            continue
        add(3, "coherence", a["message"] + " Pouvez-vous vérifier ?", champ=",".join(a.get("champs", [])),
            options=["C'est exact", "Corriger"])

    # 3. champs
    for pt, page in d.pages.items():
        labels = _labels(templates_dir, pt)
        libre = [c for c in page.champs.values() if c.signaux.get("mode_libre")]
        if libre:
            lus = "; ".join(f"{labels.get(c.cle, c.cle)} : {c.affichage}" for c in libre[:25])
            add(4, "confirmer_page", f"Page « {PAGE_LABELS.get(pt, pt)} » (mise en page non reconnue). "
                f"J'ai lu : {lus}{' …' if len(libre) > 25 else ''}. Est-ce correct ?", page=pt,
                options=["Tout est correct", "Corriger un champ", "Reprendre la photo"])
            trouves = set(page.champs)
            for k, lab in labels.items():
                if essentiel(k) and k not in trouves and "__" not in k:
                    add(5, "manquant", f"Je n'ai pas trouvé « {lab} » sur la page {PAGE_LABELS.get(pt, pt)}. "
                        "Pouvez-vous l'indiquer ?", page=pt, champ=k, options=OPTS_SAISIR)
        for k, c in page.champs.items():
            if c.signaux.get("mode_libre"):
                continue
            lab = labels.get(k, k)
            prio = 4 if essentiel(k) else 6
            if c.statut is Statut.A_REVISER and c.valeur is not None:
                add(prio, "confirmer", f"{PAGE_LABELS.get(pt, pt)} — {lab} : j'ai lu « {c.affichage} ». "
                    + " ".join(c.raisons), page=pt, champ=k, valeur=c.affichage, options=OPTS_CONFIRMER)
            elif c.statut in (Statut.A_REVISER, Statut.ILLISIBLE):
                add(prio, "saisir", f"{PAGE_LABELS.get(pt, pt)} — {lab} : " + " ".join(c.raisons),
                    page=pt, champ=k, options=OPTS_SAISIR)

    # 4. pages pas encore photographiées
    for pt in pages_attendues or []:
        if pt not in d.pages:
            add(9, "page_manquante", f"Je n'ai pas encore la page « {PAGE_LABELS.get(pt, pt)} ». "
                "Voulez-vous la photographier ?", page=pt, options=["Photographier", "Plus tard", "Pas encore remplie"])

    qs.sort(key=lambda q: q["priorite"])
    for i, q in enumerate(qs, 1):
        q["id"] = f"q{i}"
    return qs
