from enum import Enum
from pydantic import BaseModel, Field
from typing import Any, Optional, List

class Statut(str, Enum):
    CONNU = "CONNU"
    INCONNU = "INCONNU"
    NON_FOURNI = "NON_FOURNI"
    ILLISIBLE = "ILLISIBLE"
    NON_APPLICABLE = "NON_APPLICABLE"
    A_REVISER = "A_REVISER"

class Champ(BaseModel):
    valeur: Optional[Any] = None
    statut: Statut
    confiance: float = Field(ge=0.0, le=1.0)
    texte_brut: Optional[str] = None
    methode: str # "opencv_case", "vlm_crop", "paddle_ocr", "regle"
    alertes: List[str] = []

class IdentificationPatient(BaseModel):
    age: Champ
    niveau_instruction: Champ
    profession: Champ
    consanguinite: Champ
    grossesse_desiree: Champ
    # On omet volontairement nom, téléphone et CIN du schéma final

class DossierGrossesse(BaseModel):
    id_dossier: str
    identification: Optional[IdentificationPatient]
    # Ajouter ici les autres pages (Antécédents, Visites, Accouchement...)