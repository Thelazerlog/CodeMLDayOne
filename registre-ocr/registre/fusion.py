"""Fusion des signaux -> statut + confiance par champ.

Signaux disponibles pour un champ texte :
- lecture du VLM (texte, état vide/écrit/illisible, confiance logprobs si le serveur la fournit)
- second lecteur (Tesseract sur les nombres) : accord / désaccord
- encre détectée dans la zone (indépendant de tout modèle) : contredit « vide » ou « écrit »
- format valide après normalisation (date réelle, TA SYS/DIA plausible...)
- visibilité de la zone (hors cadre ?), reflet, confiance d'alignement, qualité photo

Principe : un champ n'est CONNU que si plusieurs signaux concordent. Dans le doute -> A_REVISER,
avec la raison en clair : c'est ce que la sage-femme verra dans la conversation.

Les seuils sont dans FusionParams et se recalibrent sur l'évaluation (`cli evaluate` produit la courbe
couverture / erreurs silencieuses en fonction du seuil).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .normalize import Normalized, normalize, same_value
from .readers.base import Reading
from .schema import Champ, Provenance, Statut

# Arabe (lettres et chiffres arabes-indiens, y compris persans) et cyrillique
UNEXPECTED_SCRIPT = re.compile(r"[؀-ۿݐ-ݿЀ-ӿ]")


@dataclass
class FusionParams:
    seuil_connu: float = 0.85          # confiance minimale pour CONNU
    encre_vide: float = 0.0005         # en dessous : zone vide sans appel au modèle (au-dessus : on lit)
    encre_ecrit: float = 0.008         # au-dessus : il y a sûrement de l'écriture
    visible_min: float = 0.90
    reflet_max: float = 0.20
    contraste_min: float = 15.0        # traits imprimés à peine visibles -> zone dégradée
    conf_vlm_defaut: float = 0.80      # quand le serveur ne fournit pas de logprobs


def decide_text(key: str, ftype: str, r: Reading | None, r2: Reading | None, sig: dict,
                prov: Provenance, p: FusionParams = FusionParams()) -> Champ:
    raisons: list[str] = []
    base = dict(cle=key, type=ftype, provenance=prov, signaux=sig)

    if sig.get("visible", 1.0) < p.visible_min:
        return Champ(**base, statut=Statut.A_REVISER, confiance=0.0,
                     raisons=["Zone hors de la photo : reprendre la photo ou saisir à la main."])
    ink = sig.get("encre", 0.0)
    if r is None:  # aucun lecteur n'a été appelé (encre quasi nulle, ou IA indisponible)
        pc = sig.get("contraste_imprime")
        if ink < p.encre_vide and not sig.get("ia_indisponible") and (pc is None or pc >= p.contraste_min):
            # « vide » n'est sûr que si la photo montre bien la zone (traits imprimés nets, bonne qualité)
            clarity = 1.0 if pc is None else min(1.0, pc / 40.0)
            conf = float(min(0.97, (0.55 + 0.42 * clarity) * (0.7 + 0.3 * sig.get("align_conf", 1.0))
                             * (0.8 + 0.2 * sig.get("qualite", 1.0))))
            if conf >= p.seuil_connu:
                return Champ(**base, statut=Statut.NON_FOURNI, confiance=conf, raisons=["Zone vide (aucune encre)."])
            return Champ(**base, statut=Statut.A_REVISER, confiance=conf,
                         raisons=["Zone probablement vide, mais la photo est peu contrastée à cet endroit."])
        if sig.get("ia_indisponible"):
            return Champ(**base, statut=Statut.A_REVISER, confiance=0.0,
                         raisons=["Lecture IA indisponible : saisie manuelle."])
        return Champ(**base, statut=Statut.A_REVISER, confiance=0.0,
                     raisons=["Zone peu lisible sur la photo (reflet, ombre ou flou) : impossible d'affirmer "
                              "qu'elle est vide. Reprendre la photo ou vérifier."])

    n = normalize(r.text, ftype) if r.etat != "vide" else Normalized(None, "", True, marker="vide")
    conf = r.confidence if r.confidence is not None else p.conf_vlm_defaut
    methode = prov.methode

    # --- états explicites
    if r.etat == "erreur":  # panne du lecteur, pas un jugement sur l'écriture
        return Champ(**base, statut=Statut.A_REVISER, confiance=0.0,
                     raisons=[r.extra.get("raison", "Lecture IA en échec.") + " À relire ou saisir."])
    if r.etat == "illisible":
        return Champ(**base, statut=Statut.ILLISIBLE, texte_brut=r.text, affichage=r.text or "",
                     confiance=min(conf, 0.5), raisons=["Le modèle n'arrive pas à lire cette écriture."])
    if n.marker == "inconnu":
        return Champ(**base, statut=Statut.INCONNU, texte_brut=r.text, affichage=r.text or "?",
                     confiance=conf, raisons=["Noté « inconnu » sur le registre."])
    if n.marker == "vide":
        pc = sig.get("contraste_imprime")
        if r.etat == "vide" and pc is not None and pc < p.contraste_min:
            return Champ(**base, statut=Statut.A_REVISER, confiance=0.3,
                         raisons=["Zone dégradée (reflet/ombre) : « vide » non confirmé."])
        if ink > p.encre_ecrit and r.etat == "vide":
            return Champ(**base, statut=Statut.A_REVISER, confiance=0.3,
                         raisons=["Le modèle voit une zone vide mais il y a de l'encre : à vérifier."])
        return Champ(**base, statut=Statut.NON_FOURNI, texte_brut=r.text, affichage=r.text or "",
                     confiance=conf, raisons=["Tiret : examen non fait." if r.text else "Zone vide."])

    # --- une valeur a été lue
    if r.extra.get("etat_contradictoire"):
        conf *= 0.9
        sig["etat_contradictoire"] = True
    if ink < p.encre_vide:
        conf *= 0.4
        raisons.append("Valeur lue alors que la zone semble vide (hallucination possible).")
    if UNEXPECTED_SCRIPT.search(r.text or ""):
        # Sur photo dégradée, le modèle bascule parfois en chiffres arabes-indiens ou en cyrillique :
        # sur l'évaluation Narval, 2 lectures sur 3 de ce genre étaient fausses malgré une confiance > 0,85.
        conf *= 0.5
        sig["ecriture_inattendue"] = True
        raisons.append("Écriture inattendue dans la lecture (chiffres arabes-indiens ou cyrillique) : à vérifier.")
    if not n.format_ok:
        conf *= 0.55
        raisons.append(f"Format inattendu : {n.note}.")
    elif n.note:
        raisons.append(n.note)
    if r2 is not None and r2.text:
        from .rules import plausibility
        n2 = normalize(r2.text, ftype)
        if not n2.format_ok or plausibility(prov.page_type, key, n2.value if n2.value is not None else r2.text):
            # second avis aberrant (ex. Tesseract lit « 324 m » pour « 34 cm ») : on l'écarte
            sig["second_lecteur_ecarte"] = r2.text
        elif same_value(n, n2):
            conf = max(conf, 0.97)
            sig["accord_second_lecteur"] = True
            methode = f"{methode}+{r2.source}"
        else:
            sig["accord_second_lecteur"] = False
            sig["second_lecteur"] = r2.text
            # Tesseract est faible sur l'écriture manuscrite : son accord rassure, son désaccord pèse peu
            conf *= 0.95
    if sig.get("reflet", 0.0) > p.reflet_max:
        conf *= 0.6
        raisons.append("Reflet sur la zone.")
    conf *= 0.6 + 0.4 * sig.get("align_conf", 1.0)
    conf *= 0.7 + 0.3 * sig.get("qualite", 1.0)
    conf = float(max(0.0, min(1.0, conf)))

    statut = Statut.CONNU if conf >= p.seuil_connu else Statut.A_REVISER
    if statut is Statut.A_REVISER and not raisons:
        raisons.append(f"Lecture incertaine (confiance {conf:.2f}).")
    prov.methode = methode
    return Champ(**base, valeur=n.value, affichage=n.display, texte_brut=r.text, statut=statut,
                 confiance=conf, raisons=raisons)


def decide_checkbox(key: str, checked: bool, conf_read: float, sig: dict, prov: Provenance,
                    p: FusionParams = FusionParams()) -> Champ:
    if sig.get("visible", 1.0) < p.visible_min:
        return Champ(cle=key, type="bool", statut=Statut.A_REVISER, confiance=0.0, provenance=prov, signaux=sig,
                     raisons=["Case hors de la photo."])
    conf = conf_read * (0.6 + 0.4 * sig.get("align_conf", 1.0))
    raisons = [] if conf >= p.seuil_connu else [f"Case ambiguë (encre {sig.get('encre_case', 0):.2f})."]
    return Champ(cle=key, type="bool", valeur=checked, affichage="☒" if checked else "☐",
                 statut=Statut.CONNU if conf >= p.seuil_connu else Statut.A_REVISER,
                 confiance=float(conf), raisons=raisons, provenance=prov, signaux=sig)
