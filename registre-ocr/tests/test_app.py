"""Couche application : cycle de vie, hors ligne, chiffrement, liaison patiente, renumérisation.
Lecteur SIMULÉ (vérité terrain bruitée) : ces tests vérifient le parcours, pas la qualité de lecture."""
from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from app.lifecycle import Etat, TransitionInterdite, transition
from app.linking import candidats, differences, fusionner
from app.server import App

PAGES = ["data/clean/patient_01_identification.png", "data/clean/patient_01_grossesse_actuelle.png"]


@pytest.fixture
def app(tmp_path):
    a = App(str(tmp_path), "0000", "simulation", "templates", "SF-01")
    a.travailleur.arreter()          # les tests pilotent le travailleur à la main (tick)
    return a


def capturer(app, code, pages):
    ag = app.agent
    ag.bouton("📷 Nouveau registre")
    ag.texte(code)
    for p in pages:
        ag.photo(Path(p).read_bytes(), Path(p).name)
    ag.bouton("✅ J'ai terminé")
    return app.store.tous()[-1]


def tout_verifier(ag):
    while ag.mode == "revision":
        bs = ag.msgs[-1]["boutons"]
        ag.bouton(next((b for b in ("Continuer quand même", "Confirmer", "C'est exact", "Laisser vide") if b in bs),
                       "⏭ Passer"))


def etats(app):
    return [r["etat"] for r in app.store.tous()]


def test_transitions_interdites():
    rec = {"etat": "CAPTURE"}
    with pytest.raises(TransitionInterdite):
        transition(rec, Etat.SYNCHRONISE)          # pas d'envoi sans IA, révision ni patiente
    transition(rec, Etat.EN_ATTENTE_IA)
    with pytest.raises(TransitionInterdite):
        transition(rec, Etat.VALIDE)               # la révision ne se saute pas
    assert rec["historique"][-1]["vers"] == "EN_ATTENTE_IA"


def test_capture_hors_ligne_puis_parcours_complet(app):
    app.reseau.regler(False)
    rec = capturer(app, "K7Q2", PAGES)
    assert rec["etat"] == "EN_ATTENTE_IA" and "Pas de réseau" in app.agent.msgs[-1]["texte"]
    app.travailleur.tick()                         # hors ligne : rien ne bouge, rien n'est perdu
    assert etats(app) == ["EN_ATTENTE_IA"]
    app.reseau.regler(True)
    app.travailleur.tick()
    assert etats(app) == ["A_REVISER"] and app.agent.mode == "revision"
    tout_verifier(app.agent)
    assert app.agent.mode == "lien" and "Aucune, créer" in app.agent.msgs[-1]["boutons"]
    assert app.store.patientes() == []             # jamais de création sans la sage-femme
    app.agent.bouton("Aucune, créer")
    assert etats(app) == ["ENREGISTRE"]
    app.travailleur.tick()
    assert etats(app) == ["SYNCHRONISE"] and len(app.serveur.dossiers()) == 1


def test_coupure_pendant_traitement_et_envoi(app):
    lp = app.travailleur.lecteur_pour

    def coupe_au_premier_appel(nom, chemin=None):
        app.reseau.regler(False)                   # le réseau tombe en pleine lecture IA
        return lp(nom, chemin)
    app.travailleur.lecteur_pour = coupe_au_premier_appel
    capturer(app, "K7Q2", PAGES[:1])
    app.travailleur.tick()
    rec = app.store.tous()[0]
    assert rec["etat"] == "ECHEC_TRAITEMENT" and rec["essais_ia"] == 1
    app.travailleur.lecteur_pour = lp
    app.reseau.regler(True)
    app.travailleur.tick()                         # nouvel essai automatique au retour du réseau
    assert etats(app) == ["A_REVISER"]
    tout_verifier(app.agent)
    app.agent.bouton("Aucune, créer")
    app.reseau.latence_s = 0.4
    t = threading.Thread(target=app.travailleur.tick)
    t.start()
    time.sleep(0.1)
    app.reseau.regler(False)                       # coupure pendant l'envoi
    t.join()
    assert etats(app) == ["ECHEC_SYNC"] and app.serveur.dossiers() == []
    app.reseau.latence_s = 0
    app.reseau.regler(True)
    app.travailleur.tick()
    app.travailleur.tick()                         # idempotent : pas de doublon côté serveur
    assert etats(app) == ["SYNCHRONISE"] and len(app.serveur.dossiers()) == 1


def test_chiffrement_au_repos(app, tmp_path):
    rec = capturer(app, "K7Q2", PAGES[:1])
    brut = Path(PAGES[0]).read_bytes()
    enc = next((tmp_path / "telephone" / "images").glob("*.enc")).read_bytes()
    assert enc != brut and brut[:8] not in enc
    assert app.store.lire_image(rec["images"][0]["image_id"]) == brut      # l'original n'est pas modifié
    db = (tmp_path / "telephone" / "telephone.db").read_bytes()
    assert b"K7Q2" not in db
    from app.store import LocalStore
    with pytest.raises(Exception):
        LocalStore(tmp_path / "telephone", "mauvais-pin").get(rec["id"])


def test_doublon_et_seconde_visite(app):
    capturer(app, "K7Q2", PAGES[:1])
    app.travailleur.tick()
    tout_verifier(app.agent)
    app.agent.bouton("Aucune, créer")
    # même photo renvoyée -> doublon suspecté, la sage-femme décide
    capturer(app, "K7Q2", PAGES[:1])
    assert app.store.tous()[-1]["etat"] == "DOUBLON_SUSPECT"
    app.agent.bouton("Annuler")
    # visite suivante, code mal recopié (K7O2) : correspondance PROPOSÉE, pas imposée
    capturer(app, "K7O2", PAGES[1:])
    app.travailleur.tick()
    tout_verifier(app.agent)
    texte, boutons = app.agent.msgs[-1]["texte"], app.agent.msgs[-1]["boutons"]
    assert "Patiente 1" in boutons and "Aucune, créer" in boutons and "Je ne sais pas" in boutons
    assert "caractères qui se ressemblent" in texte
    app.agent.bouton("Patiente 1")
    p = app.store.patientes()
    assert len(p) == 1 and len(p[0]["visites"]) == 2
    assert set(p[0]["dossier_courant"]["pages"]) == {"identification", "grossesse_actuelle"}


def test_renumerisation_ne_remplace_rien_sans_accord():
    c = lambda v: {"statut": "CONNU", "valeur": v, "affichage": str(v)}  # noqa: E731
    ancien = {"pages": {"g": {"champs": {"poids": c(55), "ta": c("110/70")}}}}
    nouveau = {"pages": {"g": {"champs": {"poids": c(56), "ta": c("110/70"), "hu": c(24)}}}}
    ajouts, changements = differences(ancien, nouveau)
    assert [a[1] for a in ajouts] == ["hu"] and [x[1] for x in changements] == ["poids"]
    garde_ancien = fusionner(ancien, nouveau, set())
    assert garde_ancien["pages"]["g"]["champs"]["poids"]["valeur"] == 55
    assert garde_ancien["pages"]["g"]["champs"]["hu"]["valeur"] == 24
    assert fusionner(ancien, nouveau, {("g", "poids")})["pages"]["g"]["champs"]["poids"]["valeur"] == 56


def test_candidats_par_code():
    pats = [{"id": "a", "code": "K7Q2", "profil": {"age": 26, "ddr": "2025-04-26"}},
            {"id": "b", "code": "K7Q3", "profil": {}}, {"id": "c", "code": "ZZZZ", "profil": {}}]
    d = {"pages": {"identification": {"champs": {"age": {"statut": "CONNU", "valeur": 26}}}}}
    out = candidats("K7Q2", d, pats)
    assert [x["patiente"]["id"] for x in out] == ["a", "b"] and "même âge" in out[0]["raisons"]


def test_saisie_manuelle_sans_ia(tmp_path):
    a = App(str(tmp_path), "0000", "aucun", "templates", "SF-01")
    a.travailleur.arreter()
    ag = a.agent
    ag.bouton("✍️ Saisie manuelle")
    ag.texte("M3X9")
    ag.bouton("Accouchement")
    ag.texte("03/02/2026")                           # premier champ : date de l'accouchement
    ag.bouton("⏭ Page terminée")
    ag.bouton("✅ Saisie terminée")
    rec = a.store.tous()[0]
    c = rec["dossier"]["pages"]["accouchement"]["champs"]["date_de_l_accouchement"]
    assert c["valeur"] == "2026-02-03" and c["valide_par"] == "SF-01" and c["provenance"]["methode"] == "manuel"
    assert rec["etat"] == "VALIDE" and ag.mode == "lien"


def test_lectures_enregistrees_rejouees_sans_gpu(tmp_path):
    """Ce que le modèle a lu sur Narval est rejoué à l'identique, pour la même photo uniquement."""
    import hashlib
    import json

    import cv2

    from registre.align import load_all
    from registre.pipeline import process_image
    from registre.readers.cache import RecordingReader
    from registre.readers.secondary import OracleReader
    p = Path(PAGES[0])
    gt = json.loads(Path("data/gt/patient_01.json").read_text())["identification"]
    rec = RecordingReader(OracleReader(gt, error_rate=0.0))
    rec.name = "oracle"
    templates = load_all("templates")
    raw = p.read_bytes()
    ref = process_image(cv2.imread(str(p)), templates, rec, image_bytes=raw)
    lectures = tmp_path / "lectures.json"
    lectures.write_text(json.dumps({hashlib.sha256(raw).hexdigest(): {"modele": "test", **rec.dump()}}, default=str))
    a = App(str(tmp_path / "app"), "0000", "enregistre", "templates", "SF-01", str(lectures))
    a.travailleur.arreter()
    assert "RÉELLES" in a.lecteur_desc
    capturer(a, "K7Q2", [str(p)])
    a.travailleur.tick()
    d = a.store.tous()[0]["dossier"]["pages"]["identification"]["champs"]
    assert {k: c["valeur"] for k, c in d.items()} == {k: c.valeur for k, c in ref.page.champs.items()}
