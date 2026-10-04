"""Stockage local CHIFFRÉ du téléphone (simulé sur disque) : enregistrements, images, patientes.

- Base SQLite : seules quelques colonnes techniques sont en clair (id, état, dates) pour piloter la
  file d'attente. Tout le contenu (dossier extrait, profil patiente, historique) est chiffré (Fernet :
  AES-128-CBC + HMAC-SHA256).
- Images d'origine : fichiers chiffrés, jamais modifiés (on garde leur SHA-256), liées à
  l'enregistrement, à la date de capture, à la sage-femme et au statut de traitement.
- Clé : dérivée (PBKDF2, 390 000 itérations) du code PIN de l'appareil et d'un sel aléatoire. Sur un vrai
  téléphone, elle serait protégée par le Keystore Android / Keychain iOS.

Écriture d'abord, traitement ensuite : une photo est chiffrée et enregistrée AVANT toute autre étape,
d'où « aucun enregistrement perdu » même si l'application est coupée juste après.
"""
from __future__ import annotations

import base64
import datetime as dt
import hashlib
import json
import os
import sqlite3
import threading
import uuid
from pathlib import Path

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from .lifecycle import Etat, transition

SCHEMA = """
CREATE TABLE IF NOT EXISTS enregistrements (
    id TEXT PRIMARY KEY, etat TEXT NOT NULL, sage_femme_id TEXT NOT NULL,
    cree_le TEXT NOT NULL, maj_le TEXT NOT NULL, essais_ia INTEGER DEFAULT 0, contenu BLOB NOT NULL);
CREATE TABLE IF NOT EXISTS images (
    id TEXT PRIMARY KEY, enregistrement_id TEXT NOT NULL, sha256 TEXT NOT NULL, capture_le TEXT NOT NULL,
    sage_femme_id TEXT NOT NULL, statut TEXT NOT NULL, fichier TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS patientes (
    id TEXT PRIMARY KEY, sage_femme_id TEXT NOT NULL, cree_le TEXT NOT NULL,
    contenu BLOB NOT NULL);
CREATE TABLE IF NOT EXISTS journal_acces (
    le TEXT NOT NULL, role TEXT NOT NULL, utilisateur TEXT NOT NULL, image_id TEXT NOT NULL, autorise INTEGER);
"""


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def derive_key(pin: str, salt: bytes) -> bytes:
    kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=390_000)
    return base64.urlsafe_b64encode(kdf.derive(pin.encode()))


class LocalStore:
    def __init__(self, root: str | Path, pin: str):
        self.root = Path(root)
        (self.root / "images").mkdir(parents=True, exist_ok=True)
        salt_p = self.root / "sel"
        if not salt_p.exists():
            salt_p.write_bytes(os.urandom(16))
        self.f = Fernet(derive_key(pin, salt_p.read_bytes()))
        self.db = sqlite3.connect(self.root / "telephone.db", check_same_thread=False)
        self.db.executescript(SCHEMA)
        self.lock = threading.RLock()

    # ------------------------------------------------------------------ chiffrement
    def _enc(self, obj) -> bytes:
        return self.f.encrypt(json.dumps(obj, ensure_ascii=False, default=str).encode())

    def _dec(self, blob: bytes):
        return json.loads(self.f.decrypt(blob))

    # ------------------------------------------------------------------ enregistrements
    def nouvel_enregistrement(self, sage_femme_id: str, code_patiente: str | None) -> dict:
        rec = {"id": uuid.uuid4().hex, "etat": Etat.CAPTURE.value, "sage_femme_id": sage_femme_id,
               "code_patiente": code_patiente, "images": [], "dossier": None, "patiente_id": None,
               "historique": [{"vers": Etat.CAPTURE.value, "le": _now(), "raison": "session ouverte"}]}
        self.save(rec, cree=True)
        return rec

    def save(self, rec: dict, cree: bool = False) -> None:
        with self.lock:
            if cree:
                self.db.execute("INSERT INTO enregistrements VALUES (?,?,?,?,?,?,?)",
                                (rec["id"], rec["etat"], rec["sage_femme_id"], _now(), _now(),
                                 rec.get("essais_ia", 0), self._enc(rec)))
            else:
                self.db.execute("UPDATE enregistrements SET etat=?, maj_le=?, essais_ia=?, contenu=? WHERE id=?",
                                (rec["etat"], _now(), rec.get("essais_ia", 0), self._enc(rec), rec["id"]))
            for im in rec["images"]:
                self.db.execute("UPDATE images SET statut=? WHERE id=?", (rec["etat"], im["image_id"]))
            self.db.commit()

    def get(self, rec_id: str) -> dict:
        row = self.db.execute("SELECT contenu FROM enregistrements WHERE id=?", (rec_id,)).fetchone()
        if row is None:
            raise KeyError(rec_id)
        return self._dec(row[0])

    def par_etat(self, *etats: Etat) -> list[dict]:
        q = "SELECT contenu FROM enregistrements WHERE etat IN (%s) ORDER BY cree_le" % ",".join("?" * len(etats))
        return [self._dec(r[0]) for r in self.db.execute(q, [e.value for e in etats])]

    def tous(self) -> list[dict]:
        return [self._dec(r[0]) for r in self.db.execute("SELECT contenu FROM enregistrements ORDER BY cree_le")]

    def changer_etat(self, rec: dict, nouvel: Etat, raison: str = "") -> dict:
        with self.lock:
            transition(rec, nouvel, raison)
            self.save(rec)
        return rec

    # ------------------------------------------------------------------ images
    def ajouter_image(self, rec: dict, data: bytes, nom: str = "") -> dict:
        """Chiffre et enregistre la photo AVANT tout traitement. Renvoie ses métadonnées."""
        sha = hashlib.sha256(data).hexdigest()
        image_id = uuid.uuid4().hex
        fichier = self.root / "images" / f"{image_id}.enc"
        fichier.write_bytes(self.f.encrypt(data))
        meta = {"image_id": image_id, "sha256": sha, "capture_le": _now(), "sage_femme_id": rec["sage_femme_id"],
                "nom": Path(nom).name}
        with self.lock:
            self.db.execute("INSERT INTO images VALUES (?,?,?,?,?,?,?)",
                            (image_id, rec["id"], sha, meta["capture_le"], rec["sage_femme_id"], rec["etat"],
                             str(fichier)))
            rec["images"].append(meta)
            self.save(rec)
        return meta

    def image_deja_vue(self, sha256: str, sauf: str) -> str | None:
        row = self.db.execute("SELECT enregistrement_id FROM images WHERE sha256=? AND enregistrement_id<>?",
                              (sha256, sauf)).fetchone()
        return row[0] if row else None

    def lire_image(self, image_id: str) -> bytes:
        row = self.db.execute("SELECT fichier FROM images WHERE id=?", (image_id,)).fetchone()
        if row is None:
            raise KeyError(image_id)
        return self.f.decrypt(Path(row[0]).read_bytes())

    def info_image(self, image_id: str) -> dict | None:
        row = self.db.execute("SELECT enregistrement_id, sage_femme_id, capture_le, statut FROM images WHERE id=?",
                              (image_id,)).fetchone()
        return dict(zip(["enregistrement_id", "sage_femme_id", "capture_le", "statut"], row)) if row else None

    def journaliser_acces(self, role: str, utilisateur: str, image_id: str, autorise: bool) -> None:
        with self.lock:
            self.db.execute("INSERT INTO journal_acces VALUES (?,?,?,?,?)",
                            (_now(), role, utilisateur, image_id, int(autorise)))
            self.db.commit()

    # ------------------------------------------------------------------ patientes
    def nouvelle_patiente(self, code: str, sage_femme_id: str, profil: dict) -> dict:
        p = {"id": uuid.uuid4().hex, "code": code, "sage_femme_id": sage_femme_id, "cree_le": _now(),
             "visites": [], **profil}
        with self.lock:
            self.db.execute("INSERT INTO patientes VALUES (?,?,?,?)",  # le code reste dans la partie chiffrée
                            (p["id"], sage_femme_id, p["cree_le"], self._enc(p)))
            self.db.commit()
        return p

    def save_patiente(self, p: dict) -> None:
        with self.lock:
            self.db.execute("UPDATE patientes SET contenu=? WHERE id=?", (self._enc(p), p["id"]))
            self.db.commit()

    def patientes(self) -> list[dict]:
        return [self._dec(r[0]) for r in self.db.execute("SELECT contenu FROM patientes ORDER BY cree_le")]

    def patiente(self, pid: str) -> dict:
        row = self.db.execute("SELECT contenu FROM patientes WHERE id=?", (pid,)).fetchone()
        if row is None:
            raise KeyError(pid)
        return self._dec(row[0])
