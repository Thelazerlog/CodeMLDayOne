"""Lecteurs secondaires.

- TesseractReader : second avis local sur les champs NUMÉRIQUES (dates, TA, poids...). Faible sur l'écriture
  manuscrite en général, mais quand il est d'accord avec le VLM sur un nombre, c'est un signal fort.
  Installation Mac : `brew install tesseract tesseract-lang`.
- OracleReader    : SIMULATION à partir de la vérité terrain, avec un taux d'erreur réglable. Sert uniquement
  à tester la plomberie (statuts, fusion, évaluation, flux de révision) sans GPU ni modèle.
  Ne jamais l'utiliser pour rapporter une performance.
"""
from __future__ import annotations

import os
import random
import shutil
import subprocess
import tempfile

import cv2

from .base import ReadItem, Reading

NUMERIC_TYPES = {"date", "bp", "int", "quantity"}


class TesseractReader:
    name = "tesseract"

    def __init__(self, lang: str = "eng", types: set[str] = NUMERIC_TYPES):
        self.ok = shutil.which("tesseract") is not None
        self.lang, self.types = lang, types

    def read(self, items: list[ReadItem]) -> dict[str, Reading]:
        out = {}
        if not self.ok:
            return out
        for it in items:
            if it.ftype not in self.types:
                continue
            g = it.crop.min(axis=2) if it.crop.ndim == 3 else it.crop
            g = cv2.resize(g, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
            g = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15)
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                cv2.imwrite(f.name, g)
                path = f.name
            try:
                txt = subprocess.run(
                    ["tesseract", path, "stdout", "--psm", "7", "-l", self.lang,
                     "-c", "tessedit_char_whitelist=0123456789/.,:-kgcmSAdLj° "],
                    capture_output=True, text=True, timeout=20).stdout.strip()
            except subprocess.TimeoutExpired:
                txt = ""
            finally:
                os.unlink(path)
            out[it.key] = Reading(txt or None, "ecrit" if txt else "vide", None, self.name)
        return out


class OracleReader:
    """SIMULATION : renvoie la vérité terrain, corrompue avec une probabilité `error_rate`."""
    name = "oracle_simulation"

    def __init__(self, truth: dict[str, dict], error_rate: float = 0.08, seed: int = 0):
        self.truth, self.p, self.rng = truth, error_rate, random.Random(seed)

    def read(self, items: list[ReadItem]) -> dict[str, Reading]:
        out = {}
        for it in items:
            g = self.truth.get(it.key, {})
            txt = g.get("value") if g.get("status") == "CONNU" else g.get("raw")
            r = self.rng.random()
            if r < self.p / 2 and txt:  # erreur de lecture
                pos = self.rng.randrange(len(txt))
                txt = txt[:pos] + self.rng.choice("0123456789aeo") + txt[pos + 1:]
                out[it.key] = Reading(txt, "ecrit", self.rng.uniform(0.4, 0.9), self.name)
            elif r < self.p:  # doute exprimé
                out[it.key] = Reading(txt, "illisible", self.rng.uniform(0.2, 0.6), self.name)
            else:
                out[it.key] = Reading(txt, "ecrit" if txt else "vide", self.rng.uniform(0.9, 1.0), self.name)
        return out
