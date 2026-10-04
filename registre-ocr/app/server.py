"""Prototype de démonstration : un « téléphone » (conversation type WhatsApp) + un panneau de démo
(interrupteur réseau, file d'attente, patientes, accès aux images par rôle, agrégats anonymisés).

    python -m app.server --lecteur simulation      # démo rapide : lecteur SIMULÉ (vérité terrain bruitée)
    python -m app.server --lecteur vlm             # vraie lecture : modèle local (Ollama), lent sur Mac
    python -m app.server --lecteur enregistre      # VRAIES lectures du modèle, enregistrées sur Narval (sans GPU)
    python -m app.server --lecteur aucun           # IA indisponible : tout en saisie manuelle

Tout tourne en local (http://localhost:8000), sans internet. Bibliothèque standard uniquement.
"""
from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import statistics
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from registre.config import load as load_config
from registre.export import row as export_row
from registre.schema import Dossier

from .conversation import Agent
from .lifecycle import LIBELLES, Etat
from .store import LocalStore
from .sync import Reseau, Serveur, Travailleur

STATIC = Path(__file__).parent / "static"
DEMO_DIRS = ["data/demo", "../data/Paper Registry", "data/clean", "data/augmented", "data/drive"]
ROLES = {"sage-femme": "ses propres images", "superviseur": "toutes les images", "analyste": "aucune image"}


# --------------------------------------------------------------------------- lecteurs
def fabrique_lecteur(kind: str, templates: str, lectures: str = "data/demo/lectures.json"):
    if kind == "aucun":
        return lambda nom, chemin=None: None, "aucun (saisie manuelle)"
    if kind == "enregistre":
        import hashlib
        from registre.readers.cache import CacheReader
        table = json.loads(Path(lectures).read_text())
        modele = next(iter(table.values()), {}).get("modele", "?")
        modele = "Qwen3-VL-8B-Instruct (vLLM, A100)" if modele == "registre" else modele

        def pour(nom, chemin):
            x = table.get(hashlib.sha256(Path(chemin).read_bytes()).hexdigest())
            return CacheReader(x) if x else None
        return pour, (f"lectures RÉELLES du modèle {modele}, enregistrées sur GPU ({len(table)} photos) ; "
                      "une photo non enregistrée part en saisie manuelle")
    if kind == "vlm":
        from registre.cli import _reader
        r = _reader("vlm", load_config())
        return lambda nom, chemin=None: r, f"VLM local ({r.model})"
    from registre.readers.secondary import OracleReader
    verite = {}
    for d in DEMO_DIRS:
        for sc in Path(d).glob("*.json"):
            meta = json.loads(sc.read_text())
            if "gt" in meta:
                verite[sc.stem] = meta["gt"]
            elif meta.get("patient") and meta.get("page_type"):
                gt = json.loads(Path("data/gt", f"patient_{int(meta['patient']):02d}.json").read_text())
                verite[sc.stem] = gt.get(meta["page_type"], {})
    return (lambda nom, chemin=None: OracleReader(verite[Path(nom).stem]) if Path(nom).stem in verite else None,
            "SIMULATION (vérité terrain bruitée à 8 %) : montre le parcours, ne mesure pas la lecture")


def photos_demo() -> list[dict]:
    out = []
    for d in DEMO_DIRS:
        for p in sorted(Path(d).glob("*.png")) + sorted(Path(d).glob("*.jpg")):
            out.append({"chemin": str(p), "nom": p.name, "dossier": d})
    return out


# --------------------------------------------------------------------------- agrégats (bonus)
def tableau(serveur: Serveur) -> dict:
    rows = []
    for x in serveur.dossiers():
        try:
            rows.append(export_row(Dossier.model_validate(x["dossier"])))
        except Exception:
            continue

    def taux(col):
        v = [r[col] for r in rows if r.get(col) is not None]
        return {"testees": len(v), "positives": sum(v)} if v else {"testees": 0, "positives": 0}

    sys_ = [r["mean systolic bp"] for r in rows if r.get("mean systolic bp")]
    dia = [r["mean diastolic bp"] for r in rows if r.get("mean diastolic bp")]
    return {"dossiers": len(rows),
            "ta_moyenne": f"{statistics.mean(sys_):.0f}/{statistics.mean(dia):.0f}" if sys_ and dia else None,
            "ta_140_90": sum(1 for s, d in zip(sys_, dia) if s >= 140 or d >= 90),
            "vih": taux("hiv test result"), "syphilis": taux("syphilis test result"),
            "hepatite_b": taux("hepatitis b (Ag HBs)"),
            "hepatite_c": "absente du formulaire (Ag HBs = hépatite B)"}


# --------------------------------------------------------------------------- HTTP
class App:
    def __init__(self, racine: str, pin: str, lecteur: str, templates: str, sf: str,
                 lectures: str = "data/demo/lectures.json"):
        self.store = LocalStore(Path(racine, "telephone"), pin)
        self.reseau = Reseau(en_ligne=True)
        self.serveur = Serveur(Path(racine, "serveur", "serveur.db"))
        self.agent = Agent(self.store, self.reseau, templates, sf)
        lp, self.lecteur_desc = fabrique_lecteur(lecteur, templates, lectures)
        self.travailleur = Travailleur(self.store, self.reseau, self.serveur, lp, templates, self.agent.notifier)
        self.agent.demarrer()
        self.travailleur.demarrer()

    def etat(self, depuis: int) -> dict:
        recs = [r for r in self.store.tous() if r["etat"] != Etat.ANNULE.value]
        return {
            "messages": [m for m in self.agent.msgs if m["id"] > depuis],
            "en_ligne": self.reseau.en_ligne, "lecteur": self.lecteur_desc, "travail": self.travailleur.occupe,
            "file": [{"id": r["id"][:8], "code": r.get("code_patiente"), "photos": len(r["images"]),
                      "etat": r["etat"], "libelle": LIBELLES[Etat(r["etat"])],
                      "images": [i["image_id"] for i in r["images"]],
                      "historique": [f"{h.get('vers')} {h.get('raison', '')}" for h in r.get("historique", [])][-6:]}
                     for r in recs],
            "patientes": [{"id": p["id"][:8], "code": p["code"], "visites": len(p["visites"]), "profil": p.get("profil")}
                          for p in self.store.patientes()],
            "serveur": len(self.serveur.dossiers()),
        }


def handler(app: App):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _json(self, obj, code=200):
            data = json.dumps(obj, ensure_ascii=False, default=str).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _body(self) -> dict:
            n = int(self.headers.get("Content-Length", 0))
            return json.loads(self.rfile.read(n) or b"{}")

        def do_GET(self):
            u = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            if u.path in ("/", "/index.html"):
                data = (STATIC / "index.html").read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                return self.wfile.write(data)
            if u.path == "/api/etat":
                return self._json(app.etat(int(q.get("depuis", 0))))
            if u.path == "/api/demo":
                return self._json(photos_demo())
            if u.path == "/api/tableau":
                return self._json(tableau(app.serveur))
            if u.path.startswith("/api/image/"):
                return self._image(u.path.rsplit("/", 1)[1], q.get("role", ""), q.get("utilisateur", ""))
            self._json({"erreur": "introuvable"}, 404)

        def _image(self, image_id: str, role: str, utilisateur: str):
            info = app.store.info_image(image_id)
            if info is None:
                return self._json({"erreur": "image inconnue"}, 404)
            ok = role == "superviseur" or (role == "sage-femme" and utilisateur == info["sage_femme_id"])
            app.store.journaliser_acces(role, utilisateur, image_id, ok)
            if not ok:
                return self._json({"erreur": f"accès refusé au rôle « {role} » ({ROLES.get(role, 'rôle inconnu')})"},
                                  403)
            data = app.store.lire_image(image_id)
            self.send_response(200)
            self.send_header("Content-Type", mimetypes.guess_type("x.jpg")[0])
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            u, b = urlparse(self.path), self._body()
            if u.path == "/api/message":
                app.agent.bouton(b["bouton"]) if b.get("bouton") else app.agent.texte(b.get("texte", ""))
            elif u.path == "/api/photo":
                app.agent.photo(base64.b64decode(b["data"]), b.get("nom", "photo.jpg"))
            elif u.path == "/api/photo-demo":
                p = Path(b["chemin"]).resolve()
                if not any(p.is_relative_to(Path(d).resolve()) for d in DEMO_DIRS):
                    return self._json({"erreur": "chemin refusé"}, 400)
                app.agent.photo(p.read_bytes(), p.name)
            elif u.path == "/api/reseau":
                app.reseau.regler(bool(b.get("en_ligne")))
                with app.agent.lock:
                    app.agent._bot("📶 Réseau revenu : la file est traitée automatiquement." if app.reseau.en_ligne
                                   else "📴 Réseau coupé : je continue hors ligne, rien n'est perdu.")
            else:
                return self._json({"erreur": "introuvable"}, 404)
            self._json({"ok": True})
    return H


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--lecteur", choices=["simulation", "enregistre", "vlm", "aucun"], default="simulation")
    ap.add_argument("--lectures", default="data/demo/lectures.json", help="lectures enregistrées (--lecteur enregistre)")
    ap.add_argument("--racine", default="out/app", help="données du téléphone et du serveur simulés")
    ap.add_argument("--pin", default="2468", help="code PIN de l'appareil (dérive la clé de chiffrement)")
    ap.add_argument("--sage-femme", default="SF-01")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args(argv)
    cfg = load_config()
    app = App(a.racine, a.pin, a.lecteur, cfg.templates, a.sage_femme, a.lectures)
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), handler(app))
    print(f"Prototype sur http://localhost:{a.port}  (lecteur : {app.lecteur_desc})")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        app.travailleur.arreter()


if __name__ == "__main__":
    main()
