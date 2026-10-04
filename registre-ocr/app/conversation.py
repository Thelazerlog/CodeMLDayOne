"""Agent conversationnel type WhatsApp : ce que la sage-femme voit et les boutons qu'elle touche.

Parcours d'une visite (le registre papier reste l'outil de référence, l'agent s'y greffe) :
  1. code patiente écrit sur le registre -> 2. photos des pages (contrôle qualité sur le téléphone,
  doublons) -> 3. « J'ai terminé » : enregistrement chiffré, file « En attente de traitement IA » ->
  4. lecture IA dès que le réseau est là -> 5. vérification : Confirmer / Corriger / Reprendre la photo,
  questions de suivi, l'agent dit pourquoi il doute -> 6. patiente : [Patiente 1] [Patiente 2]
  [Aucune, créer] [Je ne sais pas] -> 7. renumérisation : ce qui a changé, la sage-femme choisit ->
  8. enregistrement, puis envoi au serveur au retour du réseau.
Saisie manuelle complète possible à tout moment (IA indisponible, page non reconnue).
"""
from __future__ import annotations

import datetime as dt
import itertools
import threading

import cv2
import numpy as np

from registre.labels import COLUMN_LABELS, PAGE_LABELS
from registre.questions import essentiel

from . import dossier as D
from .lifecycle import LIBELLES, Etat
from .linking import candidats, differences, etiquette, fusionner, norm_code, profil

B_NOUVEAU, B_FILE, B_MANUEL, B_RATTACHER = "📷 Nouveau registre", "📋 File d'attente", "✍️ Saisie manuelle", "🔗 Rattacher"
B_FINI, B_SANS_CODE = "✅ J'ai terminé", "Pas de code sur le registre"
B_PASSER, B_PLUS_TARD = "⏭ Passer", "Vérifier le reste plus tard"
B_CREER, B_NSP = "Aucune, créer", "Je ne sais pas"
B_PAGE_FINIE, B_VIDE, B_INCONNU = "⏭ Page terminée", "Laisser vide", "Inconnu"
B_ANNULER = "Annuler"
B_MENU = "↩️ Menu"
QUESTIONS_AVANT_PAUSE = 3   # au-delà, on propose de garder le reste pour plus tard (les doutes restent visibles)


class Agent:
    def __init__(self, store, reseau, templates: str = "templates", sage_femme_id: str = "SF-01",
                 controle_qualite: bool = True):
        self.store, self.reseau, self.templates = store, reseau, templates
        self.sf = sage_femme_id
        self.controle_qualite = controle_qualite
        self.msgs: list[dict] = []
        self._ids = itertools.count(1)
        self.mode, self.ctx = "idle", {}
        self.a_presenter: list[str] = []      # enregistrements lus par l'IA, à présenter dès que possible
        self.lock = threading.RLock()

    # ================================================================== messages
    def _bot(self, texte: str, boutons: list[str] | None = None, **extra) -> None:
        self.msgs.append({"id": next(self._ids), "de": "bot", "texte": texte, "boutons": boutons or [],
                          "heure": dt.datetime.now().strftime("%H:%M"), **extra})

    def _moi(self, texte: str, **extra) -> None:
        self.msgs.append({"id": next(self._ids), "de": "moi", "texte": texte, "boutons": [],
                          "heure": dt.datetime.now().strftime("%H:%M"), **extra})

    def _menu(self, texte: str = "Que voulez-vous faire ?") -> None:
        self._abandonner_session_vide()
        self.mode, self.ctx = "idle", {}
        boutons = [B_NOUVEAU, B_FILE, B_MANUEL]
        n = len(self._non_rattaches())
        if n:
            boutons.append(f"{B_RATTACHER} ({n})")
        self._bot(texte, boutons)

    def _abandonner_session_vide(self) -> None:
        """Session ouverte sans aucune photo : annulée en quittant (elle n'encombre pas la file)."""
        rid = self.ctx.get("rec") if self.mode == "photos" else None
        if rid:
            rec = self.store.get(rid)
            if rec["etat"] == Etat.CAPTURE.value and not rec["images"]:
                self.store.changer_etat(rec, Etat.ANNULE, "session vide abandonnée")

    def demarrer(self) -> None:
        with self.lock:
            self._bot("Bonjour 👋 Je numérise le registre papier. Le registre reste votre outil : "
                      "je lis les photos, je vous dis quand je doute, et vous gardez le dernier mot.\n"
                      "Tout est chiffré sur le téléphone et fonctionne sans réseau.")
            self._menu()

    # ================================================================== entrées
    def texte(self, t: str) -> None:
        with self.lock:
            t = (t or "").strip()
            if not t:
                return
            self._moi(t)
            low = t.lower()
            if low in ("menu", "aide", "annuler"):
                return self._menu()
            if low in ("file", "état", "etat"):
                return self._file()
            h = getattr(self, f"_texte_{self.mode}", None)
            if h:
                return h(t)
            self._bot("Je n'attends pas de texte ici. Utilisez les boutons, ou tapez « menu ».")
            self._repeter()

    def bouton(self, b: str) -> None:
        with self.lock:
            self._moi(b)
            if b == B_MENU:
                return self._menu()
            if b == B_NOUVEAU:
                self._abandonner_session_vide()
                return self._nouveau()
            if b == B_FILE:
                return self._file()
            if b == B_MANUEL:
                return self._nouveau(manuel=True)
            if b.startswith(B_MANUEL + " "):  # reprise d'un registre dont la lecture IA a échoué
                code = b[len(B_MANUEL) + 1:]
                rec = next((r for r in self.store.par_etat(Etat.REVISION_MANUELLE_REQUISE)
                            if (r.get("code_patiente") or "sans code") == code), None)
                return self._manuel_debut(rec) if rec else self._menu()
            if b.startswith(B_RATTACHER):
                rs = self._non_rattaches()
                return self._lier(rs[0]) if rs else self._menu("Aucun dossier en attente de rattachement.")
            h = getattr(self, f"_bouton_{self.mode}", None)
            if h:
                return h(b)
            self._menu()

    def photo(self, data: bytes, nom: str = "photo.jpg", echo: bool = True) -> None:
        with self.lock:
            if self.mode == "idle":
                self._moi("📷 " + nom, photo=True)
                self._bot("Photo gardée. Pour quelle patiente ? Indiquez le code écrit sur le registre "
                          "(vous pouvez continuer à envoyer les autres pages).", [B_SANS_CODE, B_MENU])
                self.mode, self.ctx = "code", {"photos_en_attente": [(data, nom)]}
                return
            if self.mode == "code":  # le code n'est pas encore donné : on garde les photos en attente
                self._moi("📷 " + nom, photo=True)
                self.ctx.setdefault("photos_en_attente", []).append((data, nom))
                n = len(self.ctx["photos_en_attente"])
                self._bot(f"Photo gardée ({n}). J'attends le code de la patiente.", [B_SANS_CODE, B_MENU])
                return
            if self.mode not in ("photos",):
                self._moi("📷 " + nom, photo=True)
                self._bot("Je ne peux pas recevoir de photo maintenant.")
                return self._repeter()
            rec = self.store.get(self.ctx["rec"])
            meta = self.store.ajouter_image(rec, data, nom)   # chiffrée et enregistrée AVANT tout le reste
            if echo:
                self._moi("📷 " + nom, photo=True, image_id=meta["image_id"])
            self._apres_photo(rec, meta, data)

    def notifier(self, rec: dict, quoi: str) -> None:
        """Appelé par le travailleur (autre fil) quand un enregistrement change d'état."""
        with self.lock:
            code = rec.get("code_patiente") or "sans code"
            if quoi == "traite":
                if rec["id"] not in self.a_presenter:
                    self.a_presenter.append(rec["id"])
                if self.mode == "idle":
                    self._presenter_suivant()
                else:
                    self._bot(f"🔔 Le registre {code} a été lu par l'IA. Je vous le présente dès que vous avez fini.")
            elif quoi == "manuel":
                self._bot(f"⚠️ La lecture IA du registre {code} a échoué plusieurs fois. Les photos sont gardées. "
                          "Voulez-vous saisir les valeurs à la main ?", [B_MANUEL + " " + code])
            elif quoi == "synchronise":
                self._bot(f"☁️ Registre {code} envoyé au serveur ✔️")

    # ================================================================== 1-3 capture
    def _nouveau(self, manuel: bool = False) -> None:
        self.mode, self.ctx = "code", {"manuel": manuel}
        self._bot("Quel est le code de la patiente écrit sur le registre (ex. K7Q2) ?\n"
                  "Je n'enregistre ni nom ni téléphone : seulement ce code.", [B_SANS_CODE, B_MENU])

    def _texte_code(self, t: str) -> None:
        code = norm_code(t)
        if not 3 <= len(code) <= 10:
            return self._bot("Le code doit faire 3 à 10 lettres ou chiffres. Réessayez.", [B_SANS_CODE, B_MENU])
        self._ouvrir_session(code)

    def _bouton_code(self, b: str) -> None:
        if b == B_SANS_CODE:
            return self._ouvrir_session(None)
        self._menu()

    def _ouvrir_session(self, code: str | None) -> None:
        rec = self.store.nouvel_enregistrement(self.sf, code)
        manuel, attente = self.ctx.get("manuel"), self.ctx.get("photos_en_attente", [])
        if manuel:
            return self._manuel_debut(rec)
        self.mode, self.ctx = "photos", {"rec": rec["id"]}
        if not attente:
            self._bot(f"Registre {code or '(sans code)'} ouvert. Envoyez les photos des pages remplies, une par une "
                      "(toutes les pages d'un même registre forment un seul dossier). Touchez « J'ai terminé » à la fin.",
                      [B_FINI, B_MENU])
        for data, nom in attente:
            self.photo(data, nom, echo=False)

    def _apres_photo(self, rec: dict, meta: dict, data: bytes) -> None:
        autre = self.store.image_deja_vue(meta["sha256"], rec["id"])
        if autre:
            rec.setdefault("doublons", []).append(meta["image_id"])
            self.store.save(rec)
            o = self.store.get(autre)
            self._bot(f"⚠️ Cette photo a déjà été envoyée (registre {o.get('code_patiente') or 'sans code'}, "
                      f"état : {LIBELLES[Etat(o['etat'])]}). Doublon possible : je la garde, vous déciderez à la fin.",
                      [B_FINI])
            return
        if self.controle_qualite:
            from registre.quality import assess
            img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            q = assess(img) if img is not None else None
            if q is not None and not q.accept:
                self.ctx["photo_douteuse"] = meta["image_id"]
                self.ctx.setdefault("douteuses", []).append((meta["image_id"], meta["nom"], " ".join(q.reasons)))
                self._bot("📸 Cette photo risque d'être mal lue : " + " ".join(q.reasons) +
                          "\nVoulez-vous la reprendre ?", ["Reprendre la photo", "Garder quand même"])
                return
        n = len(rec["images"])
        self._bot(f"✅ Page {n} reçue et chiffrée sur le téléphone. Autre page ?", [B_FINI])

    def _bouton_photos(self, b: str) -> None:
        rec = self.store.get(self.ctx["rec"])
        if rec["etat"] == Etat.DOUBLON_SUSPECT.value and b not in ("Ce n'est pas un doublon", B_ANNULER):
            return self._repeter()
        if b == "Reprendre la photo":
            iid = self.ctx.pop("photo_douteuse", None)
            self.ctx["douteuses"] = [d for d in self.ctx.get("douteuses", []) if d[0] != iid]
            rec["images"] = [i for i in rec["images"] if i["image_id"] != iid]
            self.store.save(rec)  # l'original reste dans le stockage chiffré, simplement non utilisé
            return self._bot("D'accord, envoyez la nouvelle photo.", [B_FINI])
        if b == "Garder quand même":
            iid = self.ctx.pop("photo_douteuse", None)
            self.ctx["douteuses"] = [d for d in self.ctx.get("douteuses", []) if d[0] != iid]
            rec.setdefault("photos_gardees", []).append(iid)
            self.store.save(rec)
            return self._bot(f"Gardée. {len(rec['images'])} page(s) au total. Autre page ?", [B_FINI])
        if b == B_FINI:
            dts = self.ctx.get("douteuses", [])
            if dts:  # photos douteuses envoyées en lot : la sage-femme décide avant la mise en file
                return self._bot(f"Avant d'enregistrer : {len(dts)} photo(s) risquent d'être mal lues.\n" +
                                 "\n".join(f"• {nom} : {why}" for _, nom, why in dts),
                                 ["Les garder et terminer", "Reprendre ces photos"])
            return self._terminer_capture(rec)
        if b == "Les garder et terminer":
            rec.setdefault("photos_gardees", []).extend(d[0] for d in self.ctx.pop("douteuses", []))
            self.ctx.pop("photo_douteuse", None)
            self.store.save(rec)
            return self._terminer_capture(rec)
        if b == "Reprendre ces photos":
            ids = {d[0] for d in self.ctx.pop("douteuses", [])}
            self.ctx.pop("photo_douteuse", None)
            rec["images"] = [i for i in rec["images"] if i["image_id"] not in ids]
            self.store.save(rec)
            return self._bot(f"D'accord : envoyez les nouvelles photos de ces {len(ids)} page(s), puis touchez "
                             "« J'ai terminé ».", [B_FINI, B_MENU])
        if b == "Ce n'est pas un doublon":
            self.store.changer_etat(rec, Etat.EN_ATTENTE_IA, "doublon écarté par la sage-femme")
            return self._apres_mise_en_file(rec)
        if b == B_ANNULER:
            self.store.changer_etat(rec, Etat.ANNULE, "doublon confirmé")
            return self._menu("Enregistrement annulé (les photos déjà envoyées restent dans l'autre dossier).")
        self._repeter()

    def _terminer_capture(self, rec: dict) -> None:
        if not rec["images"]:
            return self._bot("Je n'ai encore aucune photo. Envoyez au moins une page (bouton 📷), ou revenez au menu.",
                             [B_MENU])
        if self.ctx.get("reprise"):  # photo reprise pendant la vérification : relecture de tout le registre
            self.store.changer_etat(rec, Etat.EN_ATTENTE_IA, "photo reprise")
            return self._apres_mise_en_file(rec)
        doublons = set(rec.get("doublons", []))
        if doublons and all(i["image_id"] in doublons for i in rec["images"]):
            self.store.changer_etat(rec, Etat.DOUBLON_SUSPECT, "toutes les photos ont déjà été envoyées")
            return self._bot("Toutes ces photos ont déjà été envoyées : c'est sans doute un doublon.",
                             ["Ce n'est pas un doublon", B_ANNULER])
        self.store.changer_etat(rec, Etat.EN_ATTENTE_IA, f"{len(rec['images'])} photo(s)")
        self._apres_mise_en_file(rec)

    def _apres_mise_en_file(self, rec: dict) -> None:
        n = len(self.store.par_etat(Etat.EN_ATTENTE_IA, Etat.ECHEC_TRAITEMENT))
        if self.reseau.en_ligne:
            self._menu(f"📥 Registre {rec.get('code_patiente') or '(sans code)'} enregistré. Lecture IA en cours… "
                       "Je vous préviens dès que c'est prêt.")
        else:
            self._menu(f"📴 Pas de réseau. Le registre {rec.get('code_patiente') or '(sans code)'} est enregistré, "
                       f"chiffré, sur le téléphone : « {LIBELLES[Etat.EN_ATTENTE_IA]} » ({n} dans la file). "
                       "Il sera lu automatiquement au retour du réseau. Vous pouvez continuer à travailler.")

    # ================================================================== file d'attente
    def _file(self) -> None:
        recs = [r for r in self.store.tous() if r["etat"] != Etat.ANNULE.value]
        if not recs:
            return self._menu("La file est vide.")
        lignes = []
        for r in recs[-8:]:
            lignes.append(f"• {r.get('code_patiente') or '(sans code)'} — {len(r['images'])} photo(s) — "
                          f"{LIBELLES[Etat(r['etat'])]}")
            for h in r.get("historique", [])[-4:]:
                heure = dt.datetime.fromisoformat(h["le"]).astimezone().strftime("%H:%M") if h.get("le") else ""
                lignes.append(f"    {heure} {LIBELLES[Etat(h['vers'])]}" + (f" ({h['raison']})" if h.get("raison") else ""))
        self._menu("📋 Enregistrements sur ce téléphone :\n" + "\n".join(lignes) +
                   f"\nRéseau : {'en ligne 📶' if self.reseau.en_ligne else 'hors ligne 📴'}")

    def _non_rattaches(self) -> list[dict]:
        return [r for r in self.store.par_etat(Etat.VALIDE) if not self.a_presenter or r["id"] not in self.a_presenter]

    def _presenter_suivant(self) -> None:
        while self.a_presenter:
            rec = self.store.get(self.a_presenter.pop(0))
            if rec["etat"] in (Etat.A_REVISER.value, Etat.VALIDE.value):
                return self._presenter(rec)
        self._menu()

    def _repeter(self) -> None:
        last = next((m for m in reversed(self.msgs) if m["de"] == "bot" and m["boutons"]), None)
        if last:
            self._bot("(Rappel)", last["boutons"])

    # ================================================================== 5. vérification
    def _presenter(self, rec: dict) -> None:
        d = rec["dossier"]
        cnt = D.compter(d)
        pages = ", ".join(PAGE_LABELS.get(p, p) for p in d["pages"]) or "aucune page reconnue"
        qs = d.get("questions", [])
        self._bot(f"📄 Registre {rec.get('code_patiente') or '(sans code)'} lu : {pages}.\n"
                  f"{cnt.get('CONNU', 0)} valeurs sûres, {cnt.get('NON_FOURNI', 0)} cases vides, "
                  f"{cnt.get('A_REVISER', 0) + cnt.get('ILLISIBLE', 0)} à vérifier, "
                  f"{cnt.get('NON_APPLICABLE', 0)} sans objet.")
        if not qs or rec["etat"] == Etat.VALIDE.value:
            if rec["etat"] == Etat.A_REVISER.value:
                self.store.changer_etat(rec, Etat.VALIDE, "rien à vérifier")
            return self._lier(rec)
        self.mode, self.ctx = "revision", {"rec": rec["id"], "i": 0, "faites": 0, "passees": 0}
        self._bot(f"J'ai {len(qs)} question(s), les plus importantes d'abord.")
        self._question()

    def _q(self) -> tuple[dict, dict | None]:
        rec = self.store.get(self.ctx["rec"])
        qs = rec["dossier"].get("questions", [])
        i = self.ctx["i"]
        return rec, (qs[i] if i < len(qs) else None)

    def _question(self) -> None:
        rec, q = self._q()
        if q is None:
            return self._fin_revision(rec)
        n = len(rec["dossier"]["questions"])
        boutons = list(q["options"]) + [B_PASSER]
        if self.ctx["faites"] >= QUESTIONS_AVANT_PAUSE and n - self.ctx["i"] > 1:
            boutons.append(B_PLUS_TARD)
        extra = {}
        c = D.champ(rec["dossier"], q.get("page") or "", q.get("champ") or "")
        if c and c.get("confiance") is not None and q["type"] == "confirmer":
            extra["confiance"] = round(c["confiance"], 2)
        self._bot(f"❓ {self.ctx['i'] + 1}/{n} — {q['texte']}", boutons, **extra)

    def _suivante(self) -> None:
        self.ctx["i"] += 1
        self.ctx["faites"] += 1
        self._question()

    def _bouton_revision(self, b: str) -> None:
        rec, q = self._q()
        if q is None:
            return self._fin_revision(rec)
        d, page, cle = rec["dossier"], q.get("page"), q.get("champ")
        if b == B_PASSER:
            self.ctx["passees"] += 1
            return self._suivante()
        if b == B_PLUS_TARD:
            self.ctx["passees"] += len(d["questions"]) - self.ctx["i"]
            self.ctx["i"] = len(d["questions"])
            return self._fin_revision(rec)
        if b in ("Confirmer", "C'est exact", "Oui", "Continuer quand même", "Tout est correct"):
            for pg, k in self._refs(q):
                D.confirmer(d, pg, k, self.sf)
                D.memoriser(rec, pg, k)
            self.store.save(rec)
            return self._suivante()
        if b in ("Corriger", "Saisir la valeur", "Saisir à la main", "Corriger un champ"):
            refs = self._refs(q)
            if len(refs) > 1:
                self.ctx["choix"] = refs
                return self._bot("Quel champ voulez-vous corriger ?",
                                 [D.libelle(self.templates, pg, k) for pg, k in refs] + [B_ANNULER])
            if not refs:
                self._bot("Cette page n'a pas pu être lue : je passe en saisie manuelle pour ce registre.")
                return self._manuel_debut(rec)
            self.ctx["saisie"] = refs[0]
            return self._demander_valeur(*refs[0])
        if "choix" in self.ctx:
            refs = self.ctx.pop("choix")
            ref = next(((pg, k) for pg, k in refs if D.libelle(self.templates, pg, k) == b), None)
            if ref:
                self.ctx["saisie"] = ref
                return self._demander_valeur(*ref)
            return self._question()
        if b == B_VIDE:
            if page and cle:
                D.saisir(d, page, cle, None, self.sf, self.templates, statut="NON_FOURNI")
                D.memoriser(rec, page, cle)
                self.store.save(rec)
            return self._suivante()
        if b in ("Reprendre la photo", "Non, c'est une autre page"):
            self.mode, self.ctx = "photos", {"rec": rec["id"], "reprise": True}
            lab = PAGE_LABELS.get(page or "", "")
            return self._bot(f"Envoyez la nouvelle photo{(' de la page « ' + lab + ' »') if lab else ''}. "
                             "Ce que vous avez déjà vérifié est conservé.", [B_FINI])
        if b == "Réessayer":
            self.store.changer_etat(rec, Etat.EN_ATTENTE_IA, "nouvel essai demandé")
            return self._apres_mise_en_file(rec)
        if b == B_ANNULER:
            self.ctx.pop("saisie", None)
            return self._question()
        self._repeter()

    def _refs(self, q: dict) -> list[tuple[str, str]]:
        if q.get("champ") and q.get("page"):
            return [(q["page"], q["champ"])]
        refs = []
        for ref in (q.get("champ") or "").split(","):
            if "." in ref:
                pg, k = ref.split(".", 1)
                refs.append((pg, k))
        return refs

    def _demander_valeur(self, page: str, cle: str) -> None:
        ftype = D.type_champ(self.templates, page, cle)
        fmt = {"date": " (JJ/MM/AAAA)", "bp": " (ex. 110/70)", "int": " (nombre)",
               "quantity": " (nombre et unité)", "bool": " (oui / non)"}.get(ftype, "")
        self._bot(f"✏️ {PAGE_LABELS.get(page, page)} — {D.libelle(self.templates, page, cle)} : "
                  f"tapez la valeur écrite sur le registre{fmt}.", [B_VIDE, B_ANNULER])

    def _texte_revision(self, t: str) -> None:
        if "saisie" not in self.ctx:
            _, q = self._q()
            refs = self._refs(q) if q else []
            if len(refs) != 1:
                return self._bot("Utilisez les boutons, ou tapez « menu ».")
            self.ctx["saisie"] = refs[0]  # valeur tapée directement : vaut correction du champ en question
        page, cle = self.ctx["saisie"]
        rec = self.store.get(self.ctx["rec"])
        ok, err = D.saisir(rec["dossier"], page, cle, t, self.sf, self.templates)
        if not ok:
            return self._bot(f"Je n'arrive pas à comprendre « {t} » : {err}. Réessayez.", [B_VIDE, B_ANNULER])
        D.memoriser(rec, page, cle)
        self.store.save(rec)
        self.ctx.pop("saisie")
        self._bot(f"Noté : {D.champ(rec['dossier'], page, cle)['affichage']} ✔️")
        self._suivante()

    def _fin_revision(self, rec: dict) -> None:
        restants = self.ctx.get("passees", 0) + max(0, len(rec["dossier"].get("questions", [])) - self.ctx.get("i", 0))
        if restants > 0:
            self._bot(f"{restants} point(s) restent à vérifier : ils sont marqués « à vérifier » dans le dossier "
                      "(rien n'est présenté comme sûr) et n'entrent pas dans les statistiques.")
        self.store.changer_etat(rec, Etat.VALIDE, f"vérifié par {self.sf}")
        self._lier(rec)

    # ================================================================== 6. patiente
    def _lier(self, rec: dict) -> None:
        cands = candidats(rec.get("code_patiente"), rec.get("dossier"), self.store.patientes())
        self.mode, self.ctx = "lien", {"rec": rec["id"], "cands": [c["patiente"]["id"] for c in cands]}
        code = rec.get("code_patiente") or "(sans code)"
        if not cands:
            return self._bot(f"👤 Aucune patiente connue avec le code {code}. Je ne crée rien sans votre accord.",
                             [B_CREER, B_NSP])
        lignes = [f"Patiente {i} : {etiquette(c['patiente'])} ({', '.join(c['raisons'])})"
                  for i, c in enumerate(cands, 1)]
        self._bot(f"👤 À quelle patiente rattacher ce registre (code {code}) ?\n" + "\n".join(lignes),
                  [f"Patiente {i}" for i in range(1, len(cands) + 1)] + [B_CREER, B_NSP])

    def _bouton_lien(self, b: str) -> None:
        rec = self.store.get(self.ctx["rec"])
        if b == B_NSP:
            self._menu("D'accord. Le dossier reste vérifié sur le téléphone, sans patiente. "
                       f"Je vous le reproposerai (bouton « {B_RATTACHER} »).")
            return
        if b == B_CREER:
            p = self.store.nouvelle_patiente(rec.get("code_patiente") or "SANS-CODE", self.sf,
                                             {"profil": profil(rec["dossier"]), "dossier_courant": None})
            self._bot(f"Nouvelle patiente créée (identifiant interne {p['id'][:8]}…, tiré au hasard).")
            return self._rattacher(rec, p)
        if b.startswith("Patiente "):
            i = int(b.split()[1]) - 1
            if 0 <= i < len(self.ctx["cands"]):
                return self._rattacher(rec, self.store.patiente(self.ctx["cands"][i]))
        self._repeter()

    def _rattacher(self, rec: dict, p: dict) -> None:
        rec["patiente_id"] = p["id"]
        self.store.changer_etat(rec, Etat.PATIENTE_LIEE, f"patiente {p['id'][:8]}")
        ajouts, changements = differences(p.get("dossier_courant"), rec["dossier"])
        if p.get("dossier_courant"):
            self._bot(f"Ce registre a déjà été numérisé ({len(p['visites'])} fois). "
                      f"{len(ajouts)} nouvelle(s) information(s) ajoutée(s).")
        if changements:
            self.mode, self.ctx = "renum", {"rec": rec["id"], "p": p["id"], "i": 0, "garder": [],
                                            "chg": [(pt, k) for pt, k, _, _ in changements]}
            self._bot(f"{len(changements)} valeur(s) ont changé depuis la dernière numérisation. "
                      "Laquelle garder ?", )
            return self._renum_question()
        self._enregistrer(rec, p, set())

    def _renum_question(self) -> None:
        rec, p = self.store.get(self.ctx["rec"]), self.store.patiente(self.ctx["p"])
        i, chg = self.ctx["i"], self.ctx["chg"]
        if i >= len(chg):
            return self._enregistrer(rec, p, {tuple(x) for x in self.ctx["garder"]})
        pt, k = chg[i]
        old, new = D.champ(p["dossier_courant"], pt, k), D.champ(rec["dossier"], pt, k)
        self._bot(f"🔁 {i + 1}/{len(chg)} — {PAGE_LABELS.get(pt, pt)} — {D.libelle(self.templates, pt, k)} :\n"
                  f"avant « {old.get('affichage')} », maintenant « {new.get('affichage')} ».",
                  ["Garder la nouvelle", "Garder l'ancienne", "Tout mettre à jour"])

    def _bouton_renum(self, b: str) -> None:
        chg = self.ctx["chg"]
        if b == "Tout mettre à jour":
            self.ctx["garder"] += chg[self.ctx["i"]:]
            self.ctx["i"] = len(chg)
        elif b in ("Garder la nouvelle", "Garder l'ancienne"):
            if b == "Garder la nouvelle":
                self.ctx["garder"].append(chg[self.ctx["i"]])
            self.ctx["i"] += 1
        else:
            return self._repeter()
        self._renum_question()

    def _enregistrer(self, rec: dict, p: dict, garder: set) -> None:
        p["dossier_courant"] = fusionner(p.get("dossier_courant"), rec["dossier"], garder)
        p["profil"] = {**p.get("profil", {}), **{k: v for k, v in profil(p["dossier_courant"]).items() if v}}
        p["visites"].append({"enregistrement_id": rec["id"], "le": dt.datetime.now(dt.timezone.utc).isoformat(),
                             "pages": sorted(rec["dossier"].get("pages", {}))})
        self.store.save_patiente(p)
        self.store.changer_etat(rec, Etat.ENREGISTRE, "dossier de la patiente mis à jour")
        suite = ("Envoi au serveur en cours…" if self.reseau.en_ligne
                 else "📴 Pas de réseau : il sera envoyé automatiquement au retour de la connexion.")
        self._bot(f"💾 Dossier enregistré pour la patiente {etiquette(p)}. {suite}")
        if self.a_presenter:
            return self._presenter_suivant()
        self._menu()

    # ================================================================== saisie manuelle complète
    def _manuel_debut(self, rec: dict) -> None:
        if rec.get("dossier") is None:
            rec["dossier"] = {"dossier_id": rec["id"], "pages": {}, "alertes": [], "questions": []}
            self.store.save(rec)
        self.mode, self.ctx = "manuel_page", {"rec": rec["id"]}
        self._bot("✍️ Saisie manuelle (sans IA). Quelle page voulez-vous saisir ?",
                  [PAGE_LABELS[p] for p in PAGE_LABELS] + ["✅ Saisie terminée"])

    def _bouton_manuel_page(self, b: str) -> None:
        rec = self.store.get(self.ctx["rec"])
        if b == "✅ Saisie terminée":
            return self._manuel_fin(rec)
        page = next((k for k, v in PAGE_LABELS.items() if v == b), None)
        if page is None:
            return self._repeter()
        if page == "grossesse_actuelle":
            self.ctx["page"] = page
            self.mode = "manuel_colonne"
            return self._bot("Quelle visite ?", [COLUMN_LABELS[c] for c in COLUMN_LABELS] + ["Datation (DDR, DPA)"])
        self._manuel_champs(page, None)

    def _bouton_manuel_colonne(self, b: str) -> None:
        col = next((k for k, v in COLUMN_LABELS.items() if v == b), None)
        self._manuel_champs("grossesse_actuelle", col if col else "datation")

    def _manuel_champs(self, page: str, col: str | None) -> None:
        tous = D.champs_gabarit(self.templates, page)
        if col == "datation":
            cles = [k for k in ("ddr", "date_prevue_d_accouchement", "date_de_depassement_de_terme") if k in tous]
        elif col:
            cles = [k for k in tous if k.endswith("__" + col) and (essentiel(k) or k.startswith(("venue_le", "rendez")))]
        else:
            cles = [k for k, f in tous.items() if not f.get("sensitive") and "__" not in k and
                    (essentiel(k) or f.get("kind") != "checkbox")][:15]
        self.mode, self.ctx = "manuel", {**self.ctx, "page": page, "cles": cles, "i": 0}
        self._manuel_question()

    def _manuel_question(self) -> None:
        cles, i, page = self.ctx["cles"], self.ctx["i"], self.ctx["page"]
        if i >= len(cles):
            return self._manuel_debut(self.store.get(self.ctx["rec"]))
        self._demander_valeur(page, cles[i])
        self.msgs[-1]["boutons"] = [B_VIDE, B_INCONNU, B_PAGE_FINIE]

    def _texte_manuel(self, t: str) -> None:
        rec = self.store.get(self.ctx["rec"])
        page, cle = self.ctx["page"], self.ctx["cles"][self.ctx["i"]]
        ok, err = D.saisir(rec["dossier"], page, cle, t, self.sf, self.templates)
        if not ok:
            return self._bot(f"Je n'arrive pas à comprendre « {t} » : {err}. Réessayez.",
                             [B_VIDE, B_INCONNU, B_PAGE_FINIE])
        self.store.save(rec)
        self.ctx["i"] += 1
        self._manuel_question()

    def _bouton_manuel(self, b: str) -> None:
        rec = self.store.get(self.ctx["rec"])
        page, cle = self.ctx["page"], self.ctx["cles"][self.ctx["i"]]
        if b == B_PAGE_FINIE:
            return self._manuel_debut(rec)
        if b in (B_VIDE, B_INCONNU):
            D.saisir(rec["dossier"], page, cle, None, self.sf, self.templates,
                     statut="NON_FOURNI" if b == B_VIDE else "INCONNU")
            self.store.save(rec)
            self.ctx["i"] += 1
            return self._manuel_question()
        self._repeter()

    def _manuel_fin(self, rec: dict) -> None:
        if not rec["dossier"]["pages"]:
            return self._bot("Rien n'a été saisi.", [PAGE_LABELS[p] for p in PAGE_LABELS] + ["✅ Saisie terminée"])
        etat = Etat(rec["etat"])
        if etat in (Etat.CAPTURE, Etat.EN_ATTENTE_IA, Etat.ECHEC_TRAITEMENT):
            self.store.changer_etat(rec, Etat.REVISION_MANUELLE_REQUISE, "saisie manuelle choisie")
        if Etat(rec["etat"]) is not Etat.VALIDE:
            self.store.changer_etat(rec, Etat.VALIDE, f"saisie manuelle par {self.sf}")
        self._bot(f"Saisie terminée : {D.compter(rec['dossier']).get('CONNU', 0)} valeur(s).")
        self._lier(rec)
