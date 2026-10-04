"""Mode libre : lecture d'une page dont la mise en page ne correspond à aucun gabarit.

Cas visé : les photos du vrai carnet (mise en page différente du spécimen), une page très abîmée, une
nouvelle édition du registre. Plutôt que d'abandonner, on demande au modèle local :
1. quel type de page c'est (parmi nos 8, ou « autre ») ;
2. ce qu'il voit d'écrit, rangé dans NOTRE schéma : mêmes clés de champs que les gabarits (lignes et
   colonnes des tableaux, champs simples, cases cochées). Le dossier garde donc une structure cohérente,
   quelle que soit la mise en page.

Garde-fous :
- sans gabarit, aucune vérification géométrique ni mesure d'encre : TOUT est « à confirmer » (A_REVISER),
  et la sage-femme reçoit une confirmation groupée par page (voir questions.py) ;
- le schéma de sortie ne contient AUCUN champ d'identité (nom, CIN, téléphone, adresse) : le modèle ne peut
  pas en écrire ; un filtre supprime en plus toute valeur qui ressemble à un CIN ou un téléphone ;
- ce qui n'est pas lu n'est pas inventé : absent du dossier, il fera l'objet d'une question.
"""
from __future__ import annotations

import re

import cv2
import numpy as np

from .align import Template
from .labels import COLUMN_LABELS, PAGE_CONTENT, field_label
from .normalize import normalize
from .quality import find_page_quad
from .schema import Champ, Provenance, Statut

CIN_RE = re.compile(r"\b[A-Z]{1,2}\s?\d{5,7}\b")
PHONE_RE = re.compile(r"(?:\+212|0)\s?[5-7](?:[\s.-]?\d{2}){4}")

SYSTEM = ("Tu lis des pages de registres de suivi de grossesse, remplies à la main. Tu ne recopies que ce qui "
          "est réellement écrit. Tu ne transcris jamais de nom de personne, de numéro de téléphone, d'adresse "
          "ni de numéro d'identité.")


def rectify(img: np.ndarray, max_side: int = 1600) -> np.ndarray:
    """Redresse la feuille si son contour est visible, puis réduit à une taille raisonnable pour le modèle."""
    quad = find_page_quad(img)
    if quad is not None:
        w = int(max(np.linalg.norm(quad[1] - quad[0]), np.linalg.norm(quad[2] - quad[3])))
        h = int(max(np.linalg.norm(quad[3] - quad[0]), np.linalg.norm(quad[2] - quad[1])))
        M = cv2.getPerspectiveTransform(quad, np.float32([[0, 0], [w, 0], [w, h], [0, h]]))
        img = cv2.warpPerspective(img, M, (w, h))
    s = min(1.0, max_side / max(img.shape[:2]))
    return cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img


def classify(reader, img: np.ndarray, page_types: list[str]) -> tuple[str, str]:
    """Type de page selon le modèle (image réduite : appel court). Retourne (type, justification)."""
    small = rectify(img, max_side=900)
    choices = "\n".join(f"- {pt} : {PAGE_CONTENT.get(pt, pt)}" for pt in page_types)
    prompt = (f"Cette photo montre une page d'un registre de suivi de grossesse. Quel type de page est-ce ?\n"
              f"{choices}\n- autre : aucune de ces pages, ou photo inexploitable\n"
              "Indique aussi en quelques mots le titre ou les rubriques que tu vois (sans aucun nom de personne).")
    schema = {"type": "object", "properties": {
        "type": {"type": "string", "enum": page_types + ["autre"]},
        "indices": {"type": "string"}}, "required": ["type", "indices"]}
    parsed, *_ = reader.ask(prompt, small, schema, 120, system=SYSTEM)
    pt = parsed.get("type", "autre")
    return (pt if pt in page_types else "autre"), str(parsed.get("indices", ""))[:200]


def _vocab(tpl: Template):
    """Vocabulaire du gabarit : champs simples, cases, et tableaux (lignes, colonnes, cellule -> champ)."""
    simple, boxes, tables = {}, {}, {}
    for f in tpl.data["fields"]:
        if f.get("sensitive"):
            continue
        k = f["key"]
        if f["kind"] == "checkbox":
            boxes[k] = field_label(f)
        elif f.get("row") and "__" in k:
            r, c = k.split("__", 1)
            t = tables.setdefault(f.get("table") or "t", {"rows": {}, "cols": {}, "cells": {}})
            t["rows"][r] = f["row"].rstrip(" :")
            t["cols"][c] = COLUMN_LABELS.get(c, f.get("col") or c)
            t["cells"][(r, c)] = k
        else:
            simple[k] = field_label(f)
    return simple, boxes, tables


def _match(name, choices: dict[str, str], seuil: int = 80) -> str | None:
    """Clé du schéma correspondant à ce que le modèle a écrit : la clé elle-même, ou un libellé proche
    (« Accouchement prématuré » -> accouchement_premature, « 2ème trimestre Visite 1 » -> t2_v1)."""
    from rapidfuzz import fuzz, process
    from .textutil import norm_text
    name = str(name or "").strip()
    if not name:
        return None
    if name in choices:
        return name
    q = norm_text(name).replace("eme", "e").replace("ème", "e")
    opts = {k: norm_text(v).replace("eme", "e") for k, v in choices.items()}
    opts.update({f"{k}\x00key": norm_text(k.replace("_", " ")) for k in choices})
    best = process.extractOne(q, opts, scorer=fuzz.WRatio)
    if not best or best[1] < seuil:
        return None
    return best[2].split("\x00")[0]


def _ask(reader, prompt, img, schema, max_tokens):
    parsed, *_ = reader.ask(prompt, img, schema, max_tokens, system=SYSTEM, timeout=900)
    return parsed if isinstance(parsed, dict) else {}


def extract(reader, img: np.ndarray, tpl: Template, max_tokens: int = 1500) -> dict:
    """Lecture d'une page de mise en page inconnue, rangée dans le schéma du gabarit `tpl`.
    Un appel pour les champs simples et les cases, puis un appel PAR TABLEAU : le modèle transcrit le
    tableau ligne par ligne avec les en-têtes qu'il voit (ce qu'il fait bien), et c'est nous qui
    rapprochons ces libellés du schéma (ce qu'il fait mal). Une photo peut ne montrer qu'une moitié de
    page : seuls les lignes et colonnes visibles sont renvoyées.
    Retourne {"valeurs": {cle: texte}, "cochees": [cles], "brut": [réponses]}."""
    simple, boxes, tables = _vocab(tpl)
    page = rectify(img, max_side=1600)
    regles = ("Règles : recopie exactement ce qui est écrit à la main (pas de correction, pas de traduction, pas "
              "de conversion d'unité) ; une écriture tracée en travers de plusieurs cases (ex. « RAS » en diagonale) "
              "vaut pour chacune des cases qu'elle couvre ; n'invente rien ; ne recopie jamais de nom, téléphone, "
              "adresse ni numéro d'identité. Réponds uniquement avec le JSON, sur une seule ligne.")
    vals: dict[str, str] = {}
    checked: list[str] = []
    brut = []

    if simple or boxes:
        lines = (["Champs simples (libellé imprimé) :"] + [f"  - {v}" for v in simple.values()]
                 + (["Cases à cocher :"] + [f"  - {v}" for v in boxes.values()] if boxes else []))
        prompt = ("Voici une photo d'une page de registre de suivi de grossesse (mise en page possiblement différente). "
                  "Pour les champs suivants, s'ils sont visibles sur la photo, relève ce qui est écrit à la main à côté "
                  "du libellé imprimé, et liste les cases cochées, entourées ou soulignées.\n" + "\n".join(lines) +
                  '\n\nFormat : {"c": [[libellé, texte écrit], ...], "x": [libellés des cases cochées]}. '
                  "Omets un champ vide ou absent de la photo. " + regles)
        schema = {"type": "object", "properties": {
            "c": {"type": "array", "items": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 2}},
            "x": {"type": "array", "items": {"type": "string"}}}, "required": ["c", "x"]}
        r = _ask(reader, prompt, page, schema, 600)
        brut.append(r)
        for x in r.get("c", []) or []:
            if isinstance(x, list) and len(x) == 2 and str(x[1]).strip():
                k = _match(x[0], simple)
                if k:
                    vals[k] = str(x[1]).strip()
        checked = [k for k in (_match(x, boxes) for x in r.get("x", []) or []) if k]

    for t in tables.values():
        rows = "\n".join(f"  - {v}" for v in t["rows"].values())
        cols = "\n".join(f"  - {v}" for v in dict.fromkeys(t["cols"].values()))
        prompt = ("Voici une photo d'une page de registre de suivi de grossesse. Elle peut contenir un tableau dont "
                  f"les lignes s'appellent par exemple :\n{rows}\net les colonnes :\n{cols}\n"
                  "Si ce tableau (ou une partie) est visible, transcris-le TEL QU'IL APPARAÎT SUR LA PHOTO : d'abord les "
                  "en-têtes des colonnes de VALEURS visibles, de gauche à droite (sans la colonne des libellés), chacun "
                  "précédé de son groupe s'il y en a un (ex. « 2ème trimestre - Visite 1 », « 3ème trimestre - 8ème mois ») ; "
                  "puis chaque ligne visible, DE HAUT EN BAS DANS L'ORDRE DE LA PHOTO, en recopiant le libellé imprimé tel "
                  "qu'il est écrit sur la photo, suivi d'une valeur par colonne, dans le même ordre (\"\" si la case est "
                  "vide). Ne crée pas de ligne qui n'est pas imprimée sur la photo ; si les libellés des lignes ne sont "
                  "pas visibles (page coupée), renvoie des listes vides.\n"
                  'Format : {"colonnes": [...], "lignes": [[libellé, valeur1, valeur2, ...], ...]}. '
                  "Si le tableau n'est pas sur la photo, renvoie des listes vides. " + regles)
        schema = {"type": "object", "properties": {
            "colonnes": {"type": "array", "items": {"type": "string"}},
            "lignes": {"type": "array", "items": {"type": "array", "items": {"type": "string"}}}},
            "required": ["colonnes", "lignes"]}
        r = _ask(reader, prompt, page, schema, 2500)
        brut.append(r)
        heads = [str(c) for c in r.get("colonnes", []) or []]
        lignes = [ln for ln in r.get("lignes", []) or [] if isinstance(ln, list) and len(ln) >= 2]
        if heads and lignes and len(heads) == len(lignes[0]) and _match(heads[0], t["cols"]) is None:
            heads = heads[1:]  # le modèle a inclus l'en-tête de la colonne des libellés (« Nature »...)
        colkeys = [_match(c, t["cols"]) for c in heads]
        # réponse dégénérée (même valeur recopiée dans presque toutes les lignes) : colonne rejetée
        for j in range(len(heads)):
            col = [str(ln[j + 1]).strip() for ln in lignes if len(ln) > j + 1 and str(ln[j + 1]).strip()]
            if len(lignes) >= 6 and len(col) >= 0.6 * len(lignes) and len(set(col)) <= 2:
                colkeys[j] = None
        for ln in lignes:
            if not isinstance(ln, list) or len(ln) < 2:
                continue
            rk = _match(ln[0], t["rows"])
            if rk is None:
                continue
            for ck, v in zip(colkeys, ln[1:]):
                k = t["cells"].get((rk, ck)) if ck else None
                if k and str(v).strip():
                    vals[k] = str(v).strip()
    # filet de sécurité confidentialité
    vals = {k: v for k, v in vals.items() if not CIN_RE.search(v) and not PHONE_RE.search(v)}
    return {"valeurs": vals, "cochees": checked, "brut": brut}


def to_champs(result: dict, tpl: Template, image_id: str) -> dict[str, Champ]:
    types = {f["key"]: f.get("type", "text") for f in tpl.data["fields"]}
    raison = "Lu sans gabarit (mise en page différente du modèle connu) : à confirmer."
    champs: dict[str, Champ] = {}
    for k, txt in result["valeurs"].items():
        n = normalize(txt, types.get(k, "text"))
        statut = Statut.INCONNU if n.marker == "inconnu" else (
            Statut.NON_FOURNI if n.marker == "vide" else Statut.A_REVISER)
        champs[k] = Champ(cle=k, type=types.get(k, "text"), valeur=n.value, affichage=n.display or txt,
                          texte_brut=txt, statut=statut, confiance=0.6 if n.format_ok else 0.4,
                          raisons=[raison] + ([f"Format inattendu : {n.note}."] if not n.format_ok else []),
                          signaux={"mode_libre": True},
                          provenance=Provenance(image_id=image_id, page_type=tpl.page_type, bbox=[0, 0, 0, 0],
                                                methode="vlm_page_entiere"))
    for k in result["cochees"]:
        champs[k] = Champ(cle=k, type="bool", valeur=True, affichage="☒", statut=Statut.A_REVISER, confiance=0.6,
                          raisons=[raison], signaux={"mode_libre": True},
                          provenance=Provenance(image_id=image_id, page_type=tpl.page_type, bbox=[0, 0, 0, 0],
                                                methode="vlm_page_entiere"))
    return champs
