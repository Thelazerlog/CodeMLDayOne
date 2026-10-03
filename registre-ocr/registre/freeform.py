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
    """Vocabulaire du gabarit : champs simples, lignes et colonnes des tableaux, cases."""
    simple, rows, cols, boxes, by_rc = {}, {}, {}, {}, {}
    for f in tpl.data["fields"]:
        if f.get("sensitive"):
            continue
        k = f["key"]
        if f["kind"] == "checkbox":
            boxes[k] = field_label(f)
        elif f.get("row") and "__" in k:
            r, c = k.split("__", 1)
            rows[r] = f["row"].rstrip(" :")
            cols[c] = COLUMN_LABELS.get(c, f.get("col") or c)
            by_rc[(r, c)] = f
        else:
            simple[k] = field_label(f)
    return simple, rows, cols, boxes, by_rc


def extract(reader, img: np.ndarray, tpl: Template, max_tokens: int = 1500) -> dict:
    """Lecture de la page entière, rangée dans le schéma du gabarit `tpl`.
    Retourne {"valeurs": {cle: texte}, "cochees": [cles], "brut": parsed}."""
    simple, rows, cols, boxes, by_rc = _vocab(tpl)
    lines = ["Champs simples (clé = libellé) :"] + [f"  {k} = {v}" for k, v in simple.items()]
    if rows:
        lines += ["Lignes de tableau :"] + [f"  {k} = {v}" for k, v in rows.items()]
        lines += ["Colonnes de tableau :"] + [f"  {k} = {v}" for k, v in cols.items()]
    if boxes:
        lines += ["Cases à cocher (cochées, entourées ou soulignées) :"] + [f"  {k} = {v}" for k, v in boxes.items()]
    prompt = (
        "Voici une page de registre de suivi de grossesse (mise en page possiblement différente de la liste ci-dessous). "
        "Relève UNIQUEMENT ce qui est écrit à la main, en le rangeant avec les clés suivantes.\n"
        + "\n".join(lines) +
        "\n\nFormat de réponse (compact) :\n"
        '  "c" : liste de paires [clé, texte] pour les champs simples\n'
        + ('  "t" : liste de triplets [ligne, colonne, texte] pour les tableaux\n' if rows else "")
        + ('  "x" : liste des clés des cases cochées\n' if boxes else "")
        + "Règles : recopie exactement (pas de correction, pas de traduction) ; une écriture qui couvre "
        "plusieurs cases (ex. « RAS » en travers) vaut pour chacune ; n'invente rien ; ignore ce qui n'a pas de clé ; "
        "ne recopie jamais de nom, téléphone, adresse ni numéro d'identité. Réponds uniquement avec le JSON, "
        "sur une seule ligne.")
    pair = {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 2}
    triple = {"type": "array", "items": {"type": "string"}, "minItems": 3, "maxItems": 3}
    props = {"c": {"type": "array", "items": pair}}
    if rows:
        props["t"] = {"type": "array", "items": triple}
    if boxes:
        props["x"] = {"type": "array", "items": {"type": "string", "enum": list(boxes)}}
    schema = {"type": "object", "properties": props, "required": list(props)}
    # page entière : réponse longue -> délai d'attente généreux (Mac lent), mais limite de jetons
    parsed, *_ = reader.ask(prompt, rectify(img, max_side=1400), schema, max_tokens, system=SYSTEM,
                            timeout=900)

    vals: dict[str, str] = {}
    for x in parsed.get("c", []) or []:
        if isinstance(x, list) and len(x) == 2 and x[0] in simple and str(x[1]).strip():
            vals[x[0]] = str(x[1]).strip()
    for x in parsed.get("t", []) or []:
        if not (isinstance(x, list) and len(x) == 3):
            continue
        f = by_rc.get((x[0], x[1]))
        if f and str(x[2]).strip():
            vals[f["key"]] = str(x[2]).strip()
    checked = [k for k in parsed.get("x", []) or [] if k in boxes]
    # filet de sécurité confidentialité
    vals = {k: v for k, v in vals.items() if not CIN_RE.search(v) and not PHONE_RE.search(v)}
    return {"valeurs": vals, "cochees": checked, "brut": parsed}


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
