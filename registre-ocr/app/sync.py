"""Réseau simulé, serveur central simulé, et travailleur de file d'attente.

- `Reseau` : un interrupteur en ligne / hors ligne (piloté depuis l'interface de démo). Chaque coupure
  incrémente un compteur : un traitement ou un envoi pendant lequel le réseau est tombé est considéré
  comme ÉCHOUÉ, même s'il a « fini » localement (on ne sait pas ce que le serveur a reçu).
- `Serveur` : la base du système de santé. Il reçoit des dossiers déjà validés, rattachés à un
  identifiant patiente interne, sans aucun identifiant direct. L'envoi est idempotent (clé = id de
  l'enregistrement) : renvoyer après un échec ne crée jamais de doublon.
- `Travailleur` : tant que le réseau est là, il vide la file « En attente de traitement IA » puis la
  file d'envoi. Au retour du réseau, tout repart sans action de la sage-femme.
"""
from __future__ import annotations

import json
import sqlite3
import tempfile
import threading
import time
import traceback
from pathlib import Path

from registre.pipeline import process_session

from .dossier import reappliquer
from .lifecycle import MAX_ESSAIS_IA, Etat
from .store import LocalStore


class Reseau:
    def __init__(self, en_ligne: bool = True):
        self.en_ligne = en_ligne
        self.coupures = 0
        self.latence_s = 0.0          # pour la démo : rend visible l'envoi en cours

    def regler(self, en_ligne: bool) -> None:
        if self.en_ligne and not en_ligne:
            self.coupures += 1
        self.en_ligne = en_ligne


class Serveur:
    def __init__(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.execute("CREATE TABLE IF NOT EXISTS dossiers (enregistrement_id TEXT PRIMARY KEY, "
                        "patiente_id TEXT, recu_le TEXT, contenu TEXT)")
        self.lock = threading.Lock()

    def recevoir(self, rec: dict) -> None:
        payload = {k: rec[k] for k in ("id", "patiente_id", "sage_femme_id", "images", "dossier", "historique")}
        with self.lock:
            self.db.execute("INSERT OR REPLACE INTO dossiers VALUES (?,?,datetime('now'),?)",
                            (rec["id"], rec["patiente_id"], json.dumps(payload, ensure_ascii=False, default=str)))
            self.db.commit()

    def dossiers(self) -> list[dict]:
        return [json.loads(r[0]) for r in self.db.execute("SELECT contenu FROM dossiers ORDER BY recu_le")]


class Travailleur:
    """`lecteur_pour(nom_image, chemin)` renvoie le lecteur à utiliser pour cette photo (VLM local,
    lectures enregistrées, simulation, ou None = aucun lecteur : tout part en saisie manuelle)."""

    def __init__(self, store: LocalStore, reseau: Reseau, serveur: Serveur, lecteur_pour, templates="templates",
                 notifier=None):
        self.store, self.reseau, self.serveur = store, reseau, serveur
        self.lecteur_pour, self.templates = lecteur_pour, templates
        self.notifier = notifier or (lambda rec, msg: None)
        self._stop = threading.Event()
        self.occupe = False

    # ------------------------------------------------------------------ boucle
    def demarrer(self, periode_s: float = 1.0) -> threading.Thread:
        def boucle():
            while not self._stop.is_set():
                try:
                    self.tick()
                except Exception:
                    traceback.print_exc()
                self._stop.wait(periode_s)
        t = threading.Thread(target=boucle, daemon=True)
        t.start()
        return t

    def arreter(self) -> None:
        self._stop.set()

    def tick(self) -> None:
        if not self.reseau.en_ligne:
            return
        self.occupe = True
        try:
            for rec in self.store.par_etat(Etat.ECHEC_TRAITEMENT):
                self.store.changer_etat(rec, Etat.EN_ATTENTE_IA, "nouvel essai")
            for rec in self.store.par_etat(Etat.EN_ATTENTE_IA):
                if not self.reseau.en_ligne:
                    return
                self.traiter(rec)
            for rec in self.store.par_etat(Etat.ECHEC_SYNC):
                self.store.changer_etat(rec, Etat.ENREGISTRE, "nouvel essai d'envoi")
            for rec in self.store.par_etat(Etat.ENREGISTRE):
                if not self.reseau.en_ligne:
                    return
                self.synchroniser(rec)
        finally:
            self.occupe = False

    # ------------------------------------------------------------------ traitement IA
    def traiter(self, rec: dict) -> None:
        coupures_avant = self.reseau.coupures
        rec["essais_ia"] = rec.get("essais_ia", 0) + 1
        try:
            with tempfile.TemporaryDirectory() as tmp:
                chemins, noms = [], {}
                for im in rec["images"]:
                    p = Path(tmp, f"{im['image_id']}.jpg")
                    p.write_bytes(self.store.lire_image(im["image_id"]))
                    chemins.append(str(p))
                    noms[str(p)] = im.get("nom", "")
                d, _ = process_session(chemins, None, templates_dir=self.templates,
                                       sage_femme_id=rec["sage_femme_id"], code_patiente=rec.get("code_patiente"),
                                       out_dir=None, reader_for=lambda p: self.lecteur_pour(noms[p], p))
        except Exception as e:
            self._echec_ia(rec, f"erreur : {e!r}"[:200])
            return
        if self.reseau.coupures != coupures_avant or not self.reseau.en_ligne:
            self._echec_ia(rec, "réseau coupé pendant le traitement")
            return
        # l'image d'origine reste liée à l'enregistrement : on remet nos identifiants d'image
        for cap, im in zip(d.images, rec["images"]):
            cap.chemin_original = f"chiffre:{im['image_id']}"
        rec["dossier"] = json.loads(d.model_dump_json())
        gardees = set(rec.get("photos_gardees") or [])
        if gardees:  # photo douteuse gardée en connaissance de cause : pas de seconde question « reprendre »
            pages = {cap.page_type for cap, im in zip(d.images, rec["images"]) if im["image_id"] in gardees}
            rec["dossier"]["questions"] = [q for q in rec["dossier"]["questions"]
                                           if not (q["type"] == "reprendre_photo" and q.get("page") in pages
                                                   and "pas assez bonne" in q["texte"])]
        reappliquer(rec)  # une page rephotographiée garde ce que la sage-femme a déjà validé
        self.store.changer_etat(rec, Etat.TRAITE_IA, f"{len(d.pages)} page(s) lue(s)")
        n_q = len(rec["dossier"]["questions"])
        self.store.changer_etat(rec, Etat.A_REVISER if n_q else Etat.VALIDE,
                                f"{n_q} question(s)" if n_q else "rien à vérifier")
        self.notifier(rec, "traite")

    def _echec_ia(self, rec: dict, raison: str) -> None:
        self.store.changer_etat(rec, Etat.ECHEC_TRAITEMENT, raison)
        if rec["essais_ia"] >= MAX_ESSAIS_IA:
            self.store.changer_etat(rec, Etat.REVISION_MANUELLE_REQUISE, f"{rec['essais_ia']} essais en échec")
            self.notifier(rec, "manuel")
        else:
            self.store.save(rec)

    # ------------------------------------------------------------------ envoi
    def synchroniser(self, rec: dict) -> None:
        coupures_avant = self.reseau.coupures
        time.sleep(self.reseau.latence_s)
        if self.reseau.coupures != coupures_avant or not self.reseau.en_ligne:
            self.store.changer_etat(rec, Etat.ECHEC_SYNC, "réseau coupé pendant l'envoi")
            return
        self.serveur.recevoir(rec)
        self.store.changer_etat(rec, Etat.SYNCHRONISE, "reçu par le serveur")
        self.notifier(rec, "synchronise")
