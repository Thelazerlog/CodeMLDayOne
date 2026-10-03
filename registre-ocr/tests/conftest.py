"""Les pages propres (data/clean) ne sont pas livrées dans l'archive : on les régénère si besoin."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def pytest_sessionstart(session):
    if not (ROOT / "data/clean/patient_01_couverture.png").exists():
        from registre.build_templates import build
        build(str(ROOT / "data/raw/dossiers_specimen_10_patientes.pdf"), str(ROOT / "templates"),
              str(ROOT / "data/gt"), str(ROOT / "data/clean"))
