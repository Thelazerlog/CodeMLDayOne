"""Cycle de vie d'un enregistrement (une session de photos d'un même registre) : machine à états.

    CAPTURE ─► EN_ATTENTE_IA ─► TRAITE_IA ─► A_REVISER ─► VALIDE ─► PATIENTE_LIEE ─► ENREGISTRE ─► SYNCHRONISE
                    │  ▲              │                      ▲
                    ▼  │ (réessai)    └──────────────────────┘ (rien à réviser)
              ECHEC_TRAITEMENT ─► REVISION_MANUELLE_REQUISE ─► VALIDE   (saisie à la main)
    ENREGISTRE ─► ECHEC_SYNC ─► ENREGISTRE (réessai au retour du réseau)
    CAPTURE / EN_ATTENTE_IA ─► DOUBLON_SUSPECT ─► EN_ATTENTE_IA (« ce n'est pas un doublon ») | ANNULE

Toute transition est journalisée (date, raison). Une transition non prévue lève une erreur : un
enregistrement ne peut pas « sauter » la révision ni être synchronisé sans patiente.
"""
from __future__ import annotations

import datetime as dt
from enum import Enum


class Etat(str, Enum):
    CAPTURE = "CAPTURE"
    EN_ATTENTE_IA = "EN_ATTENTE_IA"
    TRAITE_IA = "TRAITE_IA"
    A_REVISER = "A_REVISER"
    VALIDE = "VALIDE"
    PATIENTE_LIEE = "PATIENTE_LIEE"
    ENREGISTRE = "ENREGISTRE"
    SYNCHRONISE = "SYNCHRONISE"
    # états d'échec
    ECHEC_TRAITEMENT = "ECHEC_TRAITEMENT"
    ECHEC_SYNC = "ECHEC_SYNC"
    DOUBLON_SUSPECT = "DOUBLON_SUSPECT"
    REVISION_MANUELLE_REQUISE = "REVISION_MANUELLE_REQUISE"
    ANNULE = "ANNULE"


E = Etat
TRANSITIONS: dict[Etat, set[Etat]] = {
    E.CAPTURE: {E.EN_ATTENTE_IA, E.DOUBLON_SUSPECT, E.REVISION_MANUELLE_REQUISE, E.ANNULE},
    E.EN_ATTENTE_IA: {E.TRAITE_IA, E.ECHEC_TRAITEMENT, E.DOUBLON_SUSPECT, E.REVISION_MANUELLE_REQUISE},
    E.TRAITE_IA: {E.A_REVISER, E.VALIDE},
    E.A_REVISER: {E.VALIDE, E.EN_ATTENTE_IA},           # « reprendre la photo » renvoie en file
    E.VALIDE: {E.PATIENTE_LIEE},
    E.PATIENTE_LIEE: {E.ENREGISTRE},
    E.ENREGISTRE: {E.SYNCHRONISE, E.ECHEC_SYNC},
    E.ECHEC_SYNC: {E.ENREGISTRE},
    E.SYNCHRONISE: {E.A_REVISER},                       # renumérisation : mise à jour d'un dossier existant
    E.ECHEC_TRAITEMENT: {E.EN_ATTENTE_IA, E.REVISION_MANUELLE_REQUISE},
    E.REVISION_MANUELLE_REQUISE: {E.VALIDE, E.EN_ATTENTE_IA},
    E.DOUBLON_SUSPECT: {E.EN_ATTENTE_IA, E.ANNULE},
    E.ANNULE: set(),
}

# Libellés montrés à la sage-femme
LIBELLES = {
    E.CAPTURE: "Photo enregistrée sur le téléphone",
    E.EN_ATTENTE_IA: "En attente de traitement IA",
    E.TRAITE_IA: "Lu par l'IA",
    E.A_REVISER: "À vérifier",
    E.VALIDE: "Vérifié",
    E.PATIENTE_LIEE: "Rattaché à la patiente",
    E.ENREGISTRE: "Enregistré (en attente d'envoi)",
    E.SYNCHRONISE: "Envoyé au serveur",
    E.ECHEC_TRAITEMENT: "Échec de la lecture IA (sera réessayée)",
    E.ECHEC_SYNC: "Échec d'envoi (sera réessayé)",
    E.DOUBLON_SUSPECT: "Doublon possible",
    E.REVISION_MANUELLE_REQUISE: "Saisie manuelle requise",
    E.ANNULE: "Annulé",
}

MAX_ESSAIS_IA = 3


class TransitionInterdite(ValueError):
    pass


def transition(rec: dict, nouvel: Etat, raison: str = "") -> dict:
    """Change l'état d'un enregistrement (dict) en vérifiant la machine à états et en journalisant."""
    actuel = Etat(rec["etat"])
    if nouvel not in TRANSITIONS[actuel]:
        raise TransitionInterdite(f"{actuel.value} -> {nouvel.value} interdit")
    rec["etat"] = nouvel.value
    rec.setdefault("historique", []).append(
        {"de": actuel.value, "vers": nouvel.value, "le": dt.datetime.now(dt.timezone.utc).isoformat(),
         "raison": raison})
    return rec
