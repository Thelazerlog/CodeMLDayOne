"""Modèle de données du dossier : statuts explicites et provenance pour chaque champ.

Un champ n'est JAMAIS un simple « N/A » : il porte un statut qui dit pourquoi la valeur manque,
une confiance, la raison d'un doute, et d'où il vient (image, zone, méthode).
"""
from __future__ import annotations

import datetime as dt
import uuid
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class Statut(str, Enum):
    CONNU = "CONNU"                    # lu avec confiance suffisante
    INCONNU = "INCONNU"                # la sage-femme a écrit « ? », « NSP », « inconnu »
    NON_FOURNI = "NON_FOURNI"          # zone vide ou tiret
    ILLISIBLE = "ILLISIBLE"            # encre présente, lecture impossible
    NON_APPLICABLE = "NON_APPLICABLE"  # déduit par règle (ex. indication de césarienne si voie basse)
    A_REVISER = "A_REVISER"            # lu mais doute, règle violée, hors cadre, conflit entre photos


class Provenance(BaseModel):
    image_id: str
    page_type: str
    bbox: list[float]                  # zone dans le repère du gabarit (px, 150 dpi)
    methode: str                       # "case_opencv" | "vlm" | "vlm+tesseract" | "regle" | "manuel"


class Champ(BaseModel):
    cle: str
    type: str
    valeur: Optional[Any] = None       # valeur normalisée (date ISO, {"sys","dia"}, nombre+unité, texte, bool)
    affichage: str = ""                # ce qu'on montre à la sage-femme
    texte_brut: Optional[str] = None   # lecture brute du modèle
    statut: Statut
    confiance: float = 0.0
    raisons: list[str] = Field(default_factory=list)   # pourquoi ce statut / ce doute
    signaux: dict[str, Any] = Field(default_factory=dict)  # encre, accord des lecteurs, alignement...
    provenance: Optional[Provenance] = None
    valide_par: Optional[str] = None   # rempli lors de la révision conversationnelle


class ImageCapture(BaseModel):
    image_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    sha256: str
    capture_le: dt.datetime
    sage_femme_id: str
    page_type: Optional[str] = None
    qualite: dict = Field(default_factory=dict)
    alignement: dict = Field(default_factory=dict)
    chemin_original: Optional[str] = None   # stockage chiffré + accès par rôle : étape suivante
    chemin_masque: Optional[str] = None     # copie redressée avec identifiants masqués


class PageExtraite(BaseModel):
    page_type: str
    image_id: str
    champs: dict[str, Champ]


class Dossier(BaseModel):
    dossier_id: str = Field(default_factory=lambda: uuid.uuid4().hex)  # jamais dérivé d'une donnée personnelle
    code_patiente: Optional[str] = None     # code aléatoire écrit par la sage-femme sur le registre
    sage_femme_id: str
    cree_le: dt.datetime = Field(default_factory=lambda: dt.datetime.now(dt.timezone.utc))
    etat: str = "TRAITE_IA"
    images: list[ImageCapture] = Field(default_factory=list)
    pages: dict[str, PageExtraite] = Field(default_factory=dict)
    alertes: list[dict] = Field(default_factory=list)   # règles de cohérence violées
    questions: list[dict] = Field(default_factory=list)  # questions de suivi, par priorité (questions.py)

    def champs_a_reviser(self) -> list[tuple[str, Champ]]:
        out = []
        for pt, page in self.pages.items():
            for k, c in page.champs.items():
                if c.statut in (Statut.A_REVISER, Statut.ILLISIBLE):
                    out.append((f"{pt}.{k}", c))
        return out

    def resume(self) -> dict:
        from collections import Counter
        cnt = Counter(c.statut.value for p in self.pages.values() for c in p.champs.values())
        return {"dossier_id": self.dossier_id, "pages": sorted(self.pages), "statuts": dict(cnt),
                "a_reviser": len(self.champs_a_reviser()), "alertes": len(self.alertes)}
