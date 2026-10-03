"""Tests rapides (sans GPU ni modèle). Lancer : pytest -q

Pré-requis : `python -m registre.cli build-templates` (gabarits + vérité terrain + pages propres).
"""
from __future__ import annotations

import json
import re
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import cv2
import numpy as np
import pytest

from registre.align import classify_and_align, corner_error, load_all
from registre.augment import degrade, sample_params
from registre.normalize import normalize, same_value
from registre.pipeline import process_session
from registre.quality import assess, blur_sigma
from registre.readers.base import ReadItem
from registre.readers.secondary import OracleReader
from registre.readers.vlm import VLMReader
from registre.schema import Statut

ROOT = Path(__file__).resolve().parents[1]
TPL = str(ROOT / "templates")


@pytest.fixture(scope="session")
def templates():
    return load_all(TPL)


def clean(patient: int, ptype: str) -> np.ndarray:
    return cv2.imread(str(ROOT / f"data/clean/patient_{patient:02d}_{ptype}.png"))


# ------------------------------------------------------------------ normalisation
@pytest.mark.parametrize("raw,ftype,expected", [
    ("12/05/2022", "date", "2022-05-12"),
    ("١٢/٠٥/٢٠٢٢", "date", "2022-05-12"),   # chiffres arabes orientaux
    ("110/71", "bp", {"sys": 110, "dia": 71}),
    ("11,8 g/dl", "quantity", {"value": 11.8, "unit": "g/dL"}),
    ("Ferm", "text", "Fermé"),                # glyphe manquant -> vocabulaire contrôlé
])
def test_normalize(raw, ftype, expected):
    assert normalize(raw, ftype).value == expected


def test_markers():
    assert normalize("—", "date").marker == "vide"
    assert normalize("?", "int").marker == "inconnu"
    assert not normalize("31/02/2025", "date").format_ok
    assert not normalize("70/110", "bp").format_ok  # systolique < diastolique


# ------------------------------------------------------------------ qualité
def test_blur_estimate_is_monotonic():
    page = clean(1, "grossesse_actuelle")
    sig = [blur_sigma(cv2.GaussianBlur(page, (0, 0), s).min(axis=2)) if s else blur_sigma(page.min(axis=2))
           for s in (0, 1, 2, 3)]
    assert all(b > a for a, b in zip(sig, sig[1:])), sig


def test_quality_rejects_heavy_blur():
    page = clean(1, "identification")
    assert not assess(cv2.GaussianBlur(page, (0, 0), 4)).accept


# ------------------------------------------------------------------ alignement + classification
@pytest.mark.parametrize("ptype", ["grossesse_actuelle", "pp_precoce_nne", "pp_tardif_nne"])
def test_align_degraded_photo(templates, ptype):
    rng = np.random.default_rng(3)
    page = clean(3, ptype)
    meta = json.loads((ROOT / f"data/clean/patient_03_{ptype}.json").read_text())
    img, H = degrade(page, sample_params(rng, "moyen"), rng)
    al, _ = classify_and_align(img, templates)
    assert al.ok and al.page_type == ptype
    M = np.vstack([np.array(meta["M_page_to_template"]), [0, 0, 1]])
    tpl = next(t for t in templates if t.page_type == ptype)
    assert corner_error(al.H, H @ np.linalg.inv(M), tpl.size) < 3.0


# ------------------------------------------------------------------ session complète + confidentialité
def test_session_privacy_and_statuses(tmp_path):
    gt = json.loads((ROOT / "data/gt/patient_01.json").read_text())
    truth = {k: v for page in gt.values() for k, v in page.items()}
    paths = []
    for pt in ("couverture", "identification", "accouchement", "pp_precoce_mere"):
        p = tmp_path / f"{pt}.jpg"
        cv2.imwrite(str(p), clean(1, pt))
        paths.append(str(p))
    d, _ = process_session(paths, OracleReader(truth, error_rate=0.0), templates_dir=TPL,
                           sage_femme_id="SF-test", code_patiente="K7Q2", out_dir=str(tmp_path))
    blob = d.model_dump_json()
    # aucun identifiant direct de la patiente 1 (nom, CIN, téléphone, adresse, mari)
    for secret in ("Tazi", "Meryem", "CB609814", "06 00 76 13 48", "Rue Al Qods", "Mohamed"):
        assert secret not in blob
    assert not re.search(r"\b0[5-7](\s?\d{2}){4}\b", blob)
    for page in d.pages.values():
        assert not {"cin", "telephone", "adresse", "nom_du_mari", "patiente", "nom_prenom_de_la_parturiente"} & set(page.champs)
    # identifiant interne aléatoire, jamais dérivé du code ou d'une donnée personnelle
    assert re.fullmatch(r"[0-9a-f]{32}", d.dossier_id) and "K7Q2" not in d.dossier_id
    # les valeurs connues sont justes sur une page propre
    acc = d.pages["accouchement"].champs
    assert acc["date_de_l_accouchement"].valeur == "2026-02-03"
    assert acc["voie_basse_non_instrumen"].valeur is True
    # indication de césarienne : sans objet après accouchement par voie basse
    assert acc["preciser_l_indication"].statut is Statut.NON_APPLICABLE
    # la règle G/P/avortements signale l'incohérence connue de la patiente 1 (G3 P1 sans avortement)
    assert any(a["regle"].startswith("G_P_AB") for a in d.alertes)


def test_ai_unavailable_means_manual_entry(tmp_path):
    p = tmp_path / "x.jpg"
    cv2.imwrite(str(p), clean(2, "accouchement"))
    d, _ = process_session([str(p)], reader=None, templates_dir=TPL, out_dir=str(tmp_path))
    champs = d.pages["accouchement"].champs
    assert champs["poids_a_la_naissance"].statut is Statut.A_REVISER   # rien n'est inventé
    assert champs["vivant"].statut is Statut.CONNU                      # les cases restent lues


# ------------------------------------------------------------------ protocole VLM (faux serveur local)
class _FakeVLM(BaseHTTPRequestHandler):
    calls: list = []

    def log_message(self, *a):
        pass

    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'{"data":[]}')

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _FakeVLM.calls.append(body)
        if "response_format" in body:  # simule un serveur sans json_schema -> le client doit se rabattre
            self.send_response(400)
            self.end_headers()
            return
        content = json.dumps({"lectures": [{"n": 1, "texte": "110/70", "etat": "ecrit"},
                                           {"n": 2, "texte": None, "etat": "vide"}]})
        self.send_response(200)
        self.end_headers()
        self.wfile.write(json.dumps({"choices": [{"message": {"content": "```json\n" + content + "\n```"}}]}).encode())


def test_vlm_protocol_and_fallback():
    srv = HTTPServer(("127.0.0.1", 0), _FakeVLM)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    r = VLMReader(f"http://127.0.0.1:{srv.server_port}/v1", "fake")
    crop = np.full((30, 120, 3), 255, np.uint8)
    out = r.read([ReadItem("ta", crop, "TA", "bp"), ReadItem("hu", crop, "HU", "int")])
    srv.shutdown()
    assert out["ta"].text == "110/70" and out["hu"].etat == "vide"
    assert "response_format" not in _FakeVLM.calls[-1]  # repli sans schéma


def test_vlm_refuses_remote_url():
    with pytest.raises(ValueError):
        VLMReader("https://api.example.com/v1", "x")


class _FakeOllama(BaseHTTPRequestHandler):
    calls: list = []
    truncate = False

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _FakeOllama.calls.append((self.path, body))
        content = '{"lectures":[{"n":1,"etat":"lu","texte":"03/02/2026"},{"n":2,"texte":"F","etat":"vide"}]}'
        if _FakeOllama.truncate:
            content = content[:40]
        resp = {"message": {"role": "assistant", "content": content}, "done_reason": "length" if _FakeOllama.truncate
                else "stop", "eval_count": 30, "prompt_eval_count": 700, "load_duration": 1e9,
                "prompt_eval_duration": 2e9, "eval_duration": 3e9}
        self.send_response(200)
        self.end_headers()
        self.wfile.write(json.dumps(resp).encode())


def test_ollama_native_no_thinking_and_truncation():
    from registre.fusion import decide_text
    from registre.schema import Provenance
    srv = HTTPServer(("127.0.0.1", 0), _FakeOllama)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    r = VLMReader(f"http://127.0.0.1:{srv.server_port}/v1", "fake", backend="ollama")
    crop = np.full((30, 120, 3), 255, np.uint8)
    items = [ReadItem("date", crop, "Date", "date"), ReadItem("sexe", crop, "Sexe", "text")]
    out = r.read(items)
    path, body = _FakeOllama.calls[-1]
    assert path == "/api/chat"
    assert body["messages"][1]["images"] and body["think"] is False and "format" in body
    assert out["date"].text == "03/02/2026" and r.last_debug[-1]["generation_s"] == 3.0
    _FakeOllama.truncate = True
    out = r.read(items)
    srv.shutdown()
    assert out["sexe"].etat == "erreur"
    c = decide_text("sexe", "text", out["sexe"], None, {"encre": 0.05},
                    Provenance(image_id="x", page_type="accouchement", bbox=[0, 0, 1, 1], methode="vlm"))
    assert c.statut is Statut.A_REVISER and "tronquée" in c.raisons[0]


def test_contradictory_state_keeps_text():
    """Cas réel (qwen3-vl:8b-instruct) : texte lu correctement mais « etat »: « vide »."""
    from registre.fusion import decide_text
    from registre.readers.vlm import _parse_json
    from registre.schema import Provenance
    raw = '{"lectures":[{"n":1,"texte":"03/02/2026","etat":"vide"}]}'
    x = _parse_json(raw)["lectures"][0]
    assert x["texte"] == "03/02/2026"
    from registre.readers.base import Reading
    r = Reading("03/02/2026", "ecrit", None, "vlm", {"etat_contradictoire": True})
    c = decide_text("date_de_l_accouchement", "date", r, None, {"encre": 0.08},
                    Provenance(image_id="x", page_type="accouchement", bbox=[0, 0, 1, 1], methode="vlm"))
    assert c.valeur == "2026-02-03" and c.statut is not Statut.NON_FOURNI


class _FakeFreeReader:
    """Simule le modèle pour le mode libre (page dont la mise en page est inconnue)."""
    name = "fake"

    def __init__(self, fail=False):
        self.calls, self.fail = [], fail

    def ask(self, prompt, image, schema, max_tokens, system="", timeout=None):
        self.calls.append(schema)
        if "type" in schema["properties"]:
            return {"type": "grossesse_actuelle", "indices": "GROSSESSE ACTUELLE, DDR, Visites"}, "", None, {}
        if self.fail:
            raise TimeoutError("timed out")
        return ({"c": [["ddr", "19/05/2025"], ["taille", "06 61 23 45 67"]],   # téléphone : doit être filtré
                 "t": [["poids_kg", "t1_v1", "62"], ["hu_cm", "t1_v2", "340"],  # 340 : aberrant -> question
                       ["inexistante", "t1_v1", "x"]],
                 "x": ["rh_pos"]}, "", None, {})

    def read(self, items):
        return {}


def test_freeform_unknown_layout_gives_structure_and_questions(tmp_path):
    # une « page » qu'aucun gabarit ne reconnaît : feuille avec des traits sans rapport
    img = np.full((1600, 1100, 3), 235, np.uint8)
    for y in range(200, 1500, 60):
        cv2.line(img, (100, y), (1000, y + 10), (40, 40, 40), 2)
    p = tmp_path / "inconnue.jpg"
    cv2.imwrite(str(p), img)
    d, res = process_session([str(p)], _FakeFreeReader(), templates_dir=TPL, out_dir=str(tmp_path))
    page = d.pages["grossesse_actuelle"].champs
    assert page["ddr"].valeur == "2025-05-19" and page["ddr"].statut is Statut.A_REVISER
    assert page["poids_kg__t1_v1"].signaux["mode_libre"] and page["rh_pos"].valeur is True
    assert "taille" not in page                                    # valeur ressemblant à un téléphone : supprimée
    assert "Valeur 340" in " ".join(page["hu_cm__t1_v2"].raisons)  # borne -> confirmation
    types = [q["type"] for q in d.questions]
    assert "confirmer_page" in types and "manquant" in types       # confirmation groupée + champs essentiels absents
    assert d.images[0].alignement["mode"] == "libre"


def test_questions_for_doubtful_field():
    from registre.questions import build
    from registre.schema import Champ, Dossier, PageExtraite, Provenance
    prov = Provenance(image_id="x", page_type="accouchement", bbox=[0, 0, 1, 1], methode="vlm")
    c = Champ(cle="perimetre_cranien_a_la_naissance", type="quantity", valeur={"value": 324, "unit": "m"},
              affichage="324 m", statut=Statut.A_REVISER, confiance=0.5,
              raisons=["Unité inattendue « m » (attendu : cm)."], provenance=prov)
    d = Dossier(sage_femme_id="SF", pages={"accouchement": PageExtraite(page_type="accouchement", image_id="x",
                                                                         champs={c.cle: c})})
    qs = build(d, TPL, pages_attendues=["accouchement", "pp_precoce_mere"])
    assert qs[0]["type"] == "confirmer" and "324 m" in qs[0]["texte"] and "Périmètre crânien" in qs[0]["texte"]
    assert qs[-1]["type"] == "page_manquante"


def test_freeform_timeout_does_not_lose_the_record(tmp_path):
    img = np.full((1600, 1100, 3), 235, np.uint8)
    for y in range(200, 1500, 60):
        cv2.line(img, (100, y), (1000, y + 10), (40, 40, 40), 2)
    p = tmp_path / "inconnue.jpg"
    cv2.imwrite(str(p), img)
    d, _ = process_session([str(p)], _FakeFreeReader(fail=True), templates_dir=TPL, out_dir=str(tmp_path))
    assert len(d.images) == 1 and d.images[0].sha256                      # l'image est gardée
    assert any("échec" in a["message"] for a in d.alertes)
    assert any(q["type"] == "saisir" and "Réessayer" in q["options"] for q in d.questions)
