"""Lecteur principal : modèle vision-langage LOCAL, via le protocole HTTP « chat/completions ».

Ce protocole est un format d'appel, pas un fournisseur : Ollama, LM Studio, mlx_vlm.server, vLLM
et llama.cpp l'exposent tous sur localhost. Rien ne sort de la machine. Le client refuse d'ailleurs
toute URL qui n'est pas locale / réseau privé, sauf autorisation explicite (`allow_remote=True`).

Stratégie :
- les recadrages (déjà sans identifiants) sont empilés en MOSAÏQUE numérotée, 1 image par appel ;
- le modèle répond par une liste compacte {"v": [n chaînes]} contrainte par un schéma (le serveur garantit
  exactement n réponses) : ~3x moins de jetons générés qu'avec des objets {n, etat, texte} ;
- si le serveur renvoie les logprobs, on en tire une confiance par champ ; sinon None
  (la fusion s'appuie alors sur l'accord avec un second lecteur, l'encre, le format, les règles).
"""
from __future__ import annotations

import base64
import ipaddress
import json
import re
import time
import urllib.error
import urllib.request
from urllib.parse import urlparse

import cv2
import numpy as np

from .base import ReadItem, Reading, mosaic

SYSTEM = (
    "Tu transcris des registres médicaux manuscrits (suivi de grossesse). Tu recopies exactement ce qui est "
    "écrit, sans corriger, compléter, traduire ni deviner. Quand tu n'es pas sûr, tu le dis avec l'état "
    "« illisible ». Tu ne transcris jamais de nom de personne, de numéro de téléphone, d'adresse ni de numéro "
    "d'identité."
)

ILLISIBLE = "<illisible>"

PROMPT = """L'image contient {n} recadrages numérotés (1 à {n}) d'un registre papier (écriture manuscrite,
français, parfois arabe ou anglais). Transcris EXACTEMENT le texte MANUSCRIT de chaque recadrage.

Réponds par une liste « v » de {n} chaînes, dans l'ordre des numéros :
- recadrage sans aucune écriture manuscrite -> ""
- écriture présente mais impossible à lire -> "<illisible>"
- sinon : le texte exact (un tiret -> "—" ; « ? », « NSP », « inconnu » recopiés tels quels ;
  chiffres arabes orientaux (٠-٩) recopiés tels quels)
Ignore le texte imprimé, les traits et les pointillés du formulaire. Ne corrige rien, ne devine rien.

Contenu attendu (aide à la lecture, ne pas inventer) :
{hints}

Réponds uniquement avec le JSON, sur une seule ligne."""


def cell_schema(n: int) -> dict:
    """Exactement n chaînes : la grammaire du serveur garantit le bon nombre de réponses."""
    return {"type": "object",
            "properties": {"v": {"type": "array", "items": {"type": "string"}, "minItems": n, "maxItems": n}},
            "required": ["v"]}


# Ancien format (objets {n, etat, texte}) : toujours accepté en lecture, plus demandé.
SCHEMA = cell_schema(8)


def _is_local(url: str) -> bool:
    host = urlparse(url).hostname or ""
    if host in {"localhost"} or host.endswith(".local"):
        return True
    try:
        ip = ipaddress.ip_address(host)
        return ip.is_loopback or ip.is_private
    except ValueError:
        return False


class VLMReader:
    def __init__(self, base_url: str = "http://localhost:11434/v1", model: str = "qwen3-vl:8b-instruct",
                 batch: int = 8, scale: float = 2.0, timeout: float = 180.0, allow_remote: bool = False,
                 use_schema: bool = True, want_logprobs: bool = True, name: str = "vlm",
                 backend: str = "auto", think: bool = False, keep_alive: str = "30m", parallel: int = 1):
        if not allow_remote and not _is_local(base_url):
            raise ValueError(f"URL non locale refusée ({base_url}) : les données ne doivent pas sortir. "
                             "Passez allow_remote=True uniquement pour un serveur que vous contrôlez.")
        if model.endswith("-cloud") and not allow_remote:
            raise ValueError(f"Le modèle « {model} » tourne dans le cloud d'Ollama : les images sortiraient de "
                             "la machine. Utilisez une variante locale (ex. qwen3-vl:8b-instruct).")
        self.base_url, self.model, self.batch, self.scale = base_url.rstrip("/"), model, batch, scale
        self.timeout, self.use_schema, self.want_logprobs, self.name = timeout, use_schema, want_logprobs, name
        # « ollama » : API native /api/chat, qui permet de COUPER le raisonnement (think=false).
        # Sans ça, un modèle « thinking » dépense ses jetons à réfléchir et le JSON arrive tronqué.
        if backend == "auto":
            backend = "ollama" if urlparse(self.base_url).port == 11434 else "openai"
        self.backend, self.think, self.keep_alive, self.parallel = backend, think, keep_alive, parallel
        self.root = re.sub(r"/v1$", "", self.base_url)
        self.last_debug: list[dict] = []

    # ------------------------------------------------------------------ HTTP
    def _post(self, payload: dict, path: str | None = None) -> dict:
        url = f"{self.root}/api/chat" if path == "ollama" else f"{self.base_url}/chat/completions"
        req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json", "Authorization": "Bearer local"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read())

    def available(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self.base_url}/models", timeout=5) as r:
                return r.status == 200
        except Exception:
            return False

    # ------------------------------------------------------------------ lecture
    def read(self, items: list[ReadItem]) -> dict[str, Reading]:
        batches = [items[i:i + self.batch] for i in range(0, len(items), self.batch)]
        out: dict[str, Reading] = {}
        if self.parallel > 1 and len(batches) > 1:  # serveur GPU (vLLM, Narval) : appels simultanés
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(self.parallel) as ex:
                for res in ex.map(self._read_batch, batches):
                    out.update(res)
        else:
            for b in batches:
                out.update(self._read_batch(b))
        return out

    def ask(self, prompt: str, image: np.ndarray, schema: dict | None, max_tokens: int,
            system: str = SYSTEM, timeout: float | None = None) -> tuple[dict, str, object, dict]:
        """Appel générique : une image + une consigne -> JSON (contraint par `schema` si possible)."""
        ok, png = cv2.imencode(".png", image)
        b64 = base64.b64encode(png.tobytes()).decode()
        t0 = time.time()
        saved, self.timeout = self.timeout, max(self.timeout, timeout or 0)
        try:
            if self.backend == "ollama":
                content, lp, dbg = self._call_ollama(prompt, b64, max_tokens, schema, system)
            else:
                content, lp, dbg = self._call_openai(prompt, b64, max_tokens, schema, system)
        finally:
            self.timeout = saved
        dbg.update({"latency_s": round(time.time() - t0, 2), "raw": content[:3000],
                    "image_px": list(image.shape[:2])})
        if dbg.get("raisonnement") and not self.think:
            dbg["alerte"] = ("Le modèle raisonne malgré think=false : c'est une variante « thinking ». "
                             "Utilisez la variante « -instruct ».")
        self.last_debug.append(dbg)
        return _parse_json(content), content, lp, dbg

    def _read_batch(self, items: list[ReadItem]) -> dict[str, Reading]:
        img = mosaic([it.crop for it in items], scale=self.scale)
        hints = "\n".join(f"{k}. {it.hint}" for k, it in enumerate(items, 1))
        n = len(items)
        parsed, content, lp, dbg = self.ask(PROMPT.format(n=n, hints=hints), img, cell_schema(n), 60 + 25 * n)
        dbg["n"] = n
        truncated = dbg.get("fin") == "length"
        values = _as_values(parsed, n)
        confs = _logprob_conf_values(content, lp, values)

        out: dict[str, Reading] = {}
        for k, it in enumerate(items):
            v = values[k] if k < len(values) else _MISSING
            if v is _MISSING:  # réponse absente ou tronquée : on ne l'invente pas, et on ne dit pas « illisible »
                why = ("Réponse du modèle tronquée (limite de jetons atteinte)." if truncated else
                       "Le modèle n'a pas répondu pour ce champ (JSON incomplet ou invalide).")
                out[it.key] = Reading(None, "erreur", 0.0, self.name, {"raison": why})
                continue
            c = confs[k] if k < len(confs) else None
            if v is None or not str(v).strip():
                out[it.key] = Reading(None, "vide", c, self.name)
            elif str(v).strip().lower() in (ILLISIBLE, "illisible", "<illegible>"):
                out[it.key] = Reading(None, "illisible", c, self.name)
            else:
                out[it.key] = Reading(str(v).strip(), "ecrit", c, self.name)
        return out

    def _call_ollama(self, prompt: str, b64: str, max_tokens: int, schema: dict | None, system: str):
        payload = {
            "model": self.model, "stream": False, "keep_alive": self.keep_alive,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": prompt, "images": [b64]}],
            "options": {"temperature": 0, "num_predict": max_tokens, "num_ctx": 8192},
            "think": self.think,
        }
        if self.use_schema and schema:
            payload["format"] = schema
        if self.want_logprobs:
            payload["logprobs"] = True
        try:
            resp = self._post(payload, "ollama")
        except urllib.error.HTTPError as e:
            if e.code not in (400, 422):
                raise
            # option non reconnue par cette version / ce modèle : on la retire et on réessaie
            for k in ("logprobs", "think"):
                payload.pop(k, None)
            self.want_logprobs = False
            resp = self._post(payload, "ollama")
        msg = resp.get("message", {})
        ns = 1e-9
        dbg = {"fin": resp.get("done_reason"), "jetons_prompt": resp.get("prompt_eval_count"),
               "jetons_reponse": resp.get("eval_count"),
               "chargement_s": round((resp.get("load_duration") or 0) * ns, 2),
               "lecture_image_s": round((resp.get("prompt_eval_duration") or 0) * ns, 2),
               "generation_s": round((resp.get("eval_duration") or 0) * ns, 2),
               "raisonnement": (msg.get("thinking") or "")[:500]}
        return msg.get("content") or "", resp.get("logprobs"), dbg

    def _call_openai(self, prompt: str, b64: str, max_tokens: int, schema: dict | None, system: str):
        payload = {
            "model": self.model, "temperature": 0, "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64}},
                ]},
            ],
        }
        if self.use_schema and schema:
            payload["response_format"] = {"type": "json_schema",
                                          "json_schema": {"name": "reponse", "schema": schema, "strict": True}}
        if self.want_logprobs:
            payload["logprobs"] = True
            payload["top_logprobs"] = 1
        try:
            resp = self._post(payload)
        except urllib.error.HTTPError as e:
            if e.code in (400, 422) and (self.use_schema or self.want_logprobs):
                # serveur qui ne gère pas json_schema / logprobs : on retire les options et on réessaie
                payload.pop("response_format", None)
                payload.pop("logprobs", None)
                payload.pop("top_logprobs", None)
                self.use_schema = self.want_logprobs = False
                resp = self._post(payload)
            else:
                raise
        ch = resp["choices"][0]
        dbg = {"fin": ch.get("finish_reason"), "jetons_reponse": (resp.get("usage") or {}).get("completion_tokens"),
               "raisonnement": (ch["message"].get("reasoning_content") or ch["message"].get("reasoning") or "")[:500]}
        return ch["message"].get("content") or "", ch.get("logprobs"), dbg


def _parse_json(content: str) -> dict:
    content = content.strip()
    content = re.sub(r"^```(?:json)?|```$", "", content, flags=re.M).strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", content, re.S)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
    return {"lectures": []}


_MISSING = object()


def _as_values(parsed: dict, n: int) -> list:
    """Liste de n valeurs (str | None | _MISSING) depuis le format compact {"v": [...]} ou l'ancien
    format {"lectures": [{n, etat, texte}]}."""
    if isinstance(parsed.get("v"), list):
        vals = list(parsed["v"])[:n]
        return vals + [_MISSING] * (n - len(vals))
    out = [_MISSING] * n
    for x in parsed.get("lectures", []):
        if not isinstance(x, dict) or not isinstance(x.get("n"), int) or not 1 <= x["n"] <= n:
            continue
        etat, txt = x.get("etat"), x.get("texte")
        txt = None if txt in ("", "null") else txt
        if etat == "illisible":
            out[x["n"] - 1] = ILLISIBLE
        else:  # « vide » avec un texte = contradiction : le texte l'emporte, l'encre tranchera
            out[x["n"] - 1] = txt
    return out


def _token_offsets(logprobs) -> list[tuple[int, int, float]]:
    toks = logprobs.get("content") if isinstance(logprobs, dict) else logprobs  # OpenAI : dict ; Ollama : liste
    out, pos = [], 0
    for t in toks or []:
        out.append((pos, pos + len(t["token"]), t["logprob"]))
        pos += len(t["token"])
    return out


def _logprob_conf_values(content: str, logprobs, values: list) -> list[float | None]:
    """Confiance par valeur = exp(moyenne des logprobs) des jetons de la chaîne JSON correspondante."""
    offsets = _token_offsets(logprobs)
    if not offsets:
        return [None] * len(values)
    m = re.search(r'"v"\s*:\s*\[', content)
    if not m:
        return [None] * len(values)
    spans = [mm.span() for mm in re.finditer(r'"(?:[^"\\]|\\.)*"', content[m.end():])]
    out = []
    for k in range(len(values)):
        if k >= len(spans):
            out.append(None)
            continue
        a, b = spans[k][0] + m.end(), spans[k][1] + m.end()
        lps = [lp for s, e, lp in offsets if s < b and e > a]
        out.append(float(np.exp(np.mean(lps))) if lps else None)
    return out
