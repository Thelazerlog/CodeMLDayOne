"""Interface commune des lecteurs de texte."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import cv2
import numpy as np


@dataclass
class ReadItem:
    key: str
    crop: np.ndarray          # recadrage de la zone (repère gabarit), identifiants déjà exclus
    hint: str                 # libellé imprimé + colonne + format attendu
    ftype: str


@dataclass
class Reading:
    text: str | None
    etat: str                 # "ecrit" | "vide" | "illisible" | "erreur" (réponse absente / tronquée)
    confidence: float | None = None   # confiance propre au lecteur (logprobs...), None si inconnue
    source: str = ""
    extra: dict = field(default_factory=dict)


class TextReader(Protocol):
    name: str

    def read(self, items: list[ReadItem]) -> dict[str, Reading]: ...


def mosaic(crops: list[np.ndarray], scale: float = 2.0, max_w: int = 1400) -> np.ndarray:
    """Empile des recadrages numérotés dans une seule image (compatible avec tout serveur VLM,
    association numéro -> champ sans ambiguïté)."""
    scaled = []
    for c in crops:
        c = cv2.resize(c, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        if c.shape[1] > max_w:
            f = max_w / c.shape[1]
            c = cv2.resize(c, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        scaled.append(c)
    W = max((c.shape[1] for c in scaled), default=10)
    rows = []
    for i, c in enumerate(scaled, 1):
        h = max(c.shape[0], 44)
        tag = np.full((h, 70, 3), 255, np.uint8)
        cv2.putText(tag, str(i), (8, h // 2 + 12), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 2, cv2.LINE_AA)
        body = np.full((h, W, 3), 255, np.uint8)
        body[:c.shape[0], :c.shape[1]] = c
        row = np.hstack([tag, body])
        rows.append(row)
        rows.append(np.full((6, row.shape[1], 3), 90, np.uint8))  # séparateur
    return np.vstack(rows[:-1]) if rows else np.zeros((10, 10, 3), np.uint8)
