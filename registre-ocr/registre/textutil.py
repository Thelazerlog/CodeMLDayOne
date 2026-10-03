"""Petites fonctions texte partagées (slug, normalisation Unicode, chiffres arabes)."""
from __future__ import annotations

import re
import unicodedata

ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")


def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


LIGATURES = str.maketrans({"Œ": "OE", "œ": "oe", "Æ": "AE", "æ": "ae"})


def slug(s: str, maxlen: int = 40) -> str:
    s = strip_accents(s.translate(LIGATURES)).lower()
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s[:maxlen].rstrip("_") or "x"


def norm_text(s: str | None) -> str:
    """Forme canonique pour comparer deux lectures : NFKC, chiffres occidentaux,
    minuscules, sans accents, espaces compactés."""
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", s).translate(ARABIC_DIGITS)
    s = strip_accents(s).lower().replace(",", ".")
    s = re.sub(r"\s+", " ", s).strip()
    return s
