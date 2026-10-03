"""Normalisation des lectures brutes -> valeurs typées.

Chaque champ texte a un type (inféré à la construction du gabarit, modifiable dans templates/*.json) :
date, bp (tension), int, quantity (nombre + unité), text.

La normalisation renvoie aussi si le FORMAT est valide : un format invalide fait baisser la confiance
et envoie le champ en révision (jamais de correction silencieuse).
"""
from __future__ import annotations

import datetime as dt
import re
import unicodedata
from dataclasses import dataclass

from rapidfuzz import fuzz, process

from .textutil import ARABIC_DIGITS, norm_text

EMPTY_MARKERS = {"—", "-", "–", "--", "/", "ø", "x"}
UNKNOWN_MARKERS = {"?", "??", "nsp", "ne sait pas", "inconnu", "inconnue", "non connu", "؟", "unknown"}

# Vocabulaire contrôlé : formes canoniques fréquentes sur le registre (complété par build_vocab()).
BASE_VOCAB = [
    "RAS", "Néant", "Aucun", "Aucune", "Neg", "Pos +", "Pos ++", "Oui", "Non", "Normal", "Normale", "Normales",
    "Normaux", "Pâles", "Fermé", "Ouvert", "Céphalique", "Siège", "Transverse", "Immune", "Non immune",
    "Voie basse", "Césarienne", "Cycles réguliers", "Cycles irréguliers", "Non fait", "Propre", "Propre, sèche",
    "Poursuivre l'allaitement exclusif", "Souffrance fœtale", "Utérus cicatriciel", "Pré-éclampsie sévère",
    "Sage-femme", "Lycée", "Collège", "Primaire", "Supérieur", "Analphabète",
]


@dataclass
class Normalized:
    value: object          # valeur typée (str ISO pour les dates, dict pour la TA, float/int...)
    display: str           # forme lisible pour la sage-femme
    format_ok: bool
    marker: str | None = None   # "vide" | "inconnu" | None
    note: str = ""


def _clean(raw: str) -> str:
    s = unicodedata.normalize("NFKC", raw).translate(ARABIC_DIGITS).strip()
    return re.sub(r"\s+", " ", s)


def parse_date(s: str) -> dt.date | None:
    s = s.replace(" ", "")
    m = re.match(r"^(\d{1,2})[/.\-](\d{1,2})[/.\-](\d{2,4})$", s)
    if not m:
        return None
    d, mo, y = map(int, m.groups())
    if y < 100:
        y += 2000 if y < 50 else 1900
    try:
        out = dt.date(y, mo, d)
    except ValueError:
        return None
    return out if 1950 <= out.year <= 2040 else None


def parse_number(s: str) -> float | None:
    m = re.search(r"-?\d+(?:[.,]\d+)?", s)
    return float(m.group().replace(",", ".")) if m else None


UNIT_ALIASES = {
    "g": "g", "gr": "g", "kg": "kg", "cm": "cm", "sa": "SA", "s.a": "SA", "sem": "SA", "°c": "°C", "c": "°C",
    "g/dl": "g/dL", "g/l": "g/L", "k": "k", "000": "k", "jours": "jours", "j": "jours", "jour": "jours",
}


def normalize(raw: str | None, ftype: str, vocab: list[str] | None = None) -> Normalized:
    if raw is None or not str(raw).strip():
        return Normalized(None, "", True, marker="vide")
    s = _clean(str(raw))
    low = s.lower()
    if s in EMPTY_MARKERS:
        return Normalized(None, s, True, marker="vide")
    if low in UNKNOWN_MARKERS:
        return Normalized(None, s, True, marker="inconnu")

    if ftype == "date":
        d = parse_date(s)
        return Normalized(d.isoformat() if d else s, d.strftime("%d/%m/%Y") if d else s, d is not None,
                          note="" if d else "date invalide")
    if ftype == "bp":
        m = re.match(r"^(\d{2,3})\s*[/\\|-]\s*(\d{2,3})$", s)
        if not m:
            return Normalized(s, s, False, note="tension illisible au format SYS/DIA")
        sys_, dia = int(m.group(1)), int(m.group(2))
        ok = 60 <= sys_ <= 260 and 30 <= dia <= 160 and sys_ > dia
        return Normalized({"sys": sys_, "dia": dia}, f"{sys_}/{dia}", ok, note="" if ok else "tension hors bornes")
    if ftype == "int":
        n = parse_number(s)
        ok = n is not None and float(n).is_integer() and re.fullmatch(r"\d+", s.replace(" ", "")) is not None
        return Normalized(int(n) if n is not None else s, s, ok, note="" if ok else "entier attendu")
    if ftype == "quantity":
        n = parse_number(s)
        unit = re.sub(r"[-\d.,\s]+", "", s, count=1).strip().lower()
        unit = UNIT_ALIASES.get(unit, unit) if unit else ""
        if n is None:
            return Normalized(s, s, False, note="nombre attendu")
        return Normalized({"value": n, "unit": unit}, f"{n:g} {unit}".strip(), True)

    # texte libre : rapprochement d'un vocabulaire contrôlé, sans forcer
    voc = list(dict.fromkeys((vocab or []) + BASE_VOCAB))
    if voc:
        best = process.extractOne(norm_text(s), {v: norm_text(v) for v in voc}, scorer=fuzz.ratio)
        if best and best[1] >= 88:
            canon = best[2]
            return Normalized(canon, canon, True, note="" if norm_text(canon) == norm_text(s) else f"lu « {s} »")
    return Normalized(s, s, True)


def same_value(a: Normalized, b: Normalized) -> bool:
    """Égalité de deux lectures normalisées (sert à l'accord entre lecteurs et à l'évaluation)."""
    if a.marker or b.marker:
        return a.marker == b.marker
    if isinstance(a.value, dict) and isinstance(b.value, dict):
        if "value" in a.value and "value" in b.value:
            return abs(a.value["value"] - b.value["value"]) < 1e-6
        return a.value == b.value
    return norm_text(str(a.value)) == norm_text(str(b.value))
