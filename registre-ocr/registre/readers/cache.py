"""Lectures ENREGISTRÉES : rejouer, sans GPU, ce que le vrai modèle a lu sur Narval.

Un évaluateur sans carte graphique peut ainsi faire tourner le prototype avec les vraies sorties du
modèle (pas une simulation) sur les images de démo. La clé est l'empreinte SHA-256 de la photo : une
lecture enregistrée ne sert que pour exactement la même photo.

    # sur Narval (nœud GPU, vLLM lancé) :
    python -m registre.cli record data/demo --out data/demo/lectures.json
    # n'importe où, sans GPU :
    python -m app.server --lecteur enregistre
"""
from __future__ import annotations

from .base import ReadItem, Reading


class RecordingReader:
    """Enveloppe un vrai lecteur et garde chaque réponse (lectures de zones et appels du mode libre)."""

    def __init__(self, inner):
        self.inner, self.name = inner, f"enregistrement({inner.name})"
        self.lectures: dict[str, list] = {}
        self.appels: list = []

    def read(self, items: list[ReadItem]) -> dict[str, Reading]:
        out = self.inner.read(items)
        for k, r in out.items():
            self.lectures[k] = [r.text, r.etat, r.confidence, r.extra]
        return out

    def ask(self, *a, **kw):
        res = self.inner.ask(*a, **kw)
        self.appels.append(res[0])
        return res

    def dump(self) -> dict:
        return {"lectures": self.lectures, "appels": self.appels}


class CacheReader:
    """Rejoue les réponses enregistrées pour UNE photo. Une zone absente de l'enregistrement est rendue
    comme une panne de lecture (« erreur ») : elle part en saisie, jamais inventée."""
    name = "enregistre"

    def __init__(self, enregistrement: dict):
        self.lectures = enregistrement.get("lectures", {})
        self.appels = list(enregistrement.get("appels", []))

    def read(self, items: list[ReadItem]) -> dict[str, Reading]:
        out = {}
        for it in items:
            x = self.lectures.get(it.key)
            out[it.key] = (Reading(x[0], x[1], x[2], "vlm_enregistre", x[3] or {}) if x else
                           Reading(None, "erreur", 0.0, "vlm_enregistre", {"raison": "Lecture non enregistrée."}))
        return out

    def ask(self, *a, **kw):
        if not self.appels:
            raise RuntimeError("aucune réponse enregistrée pour ce mode libre")
        return self.appels.pop(0), "", None, {}
