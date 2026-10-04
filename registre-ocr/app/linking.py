"""Liaison patiente et renumérisation.

- La patiente est retrouvée par le CODE ALÉATOIRE que la sage-femme écrit sur le registre (ex. « K7Q2 »).
  Aucun nom, aucun identifiant direct : l'identifiant interne est un UUID tiré au hasard.
- On PROPOSE des correspondances, la sage-femme TRANCHE : [Patiente 1] [Patiente 2] [Aucune, créer]
  [Je ne sais pas]. Jamais de création automatique quand une correspondance est plausible.
- Correspondances plausibles : même code ; code à une lettre près (écriture ou lecture du code) ; code
  confondu par paires de caractères qui se ressemblent (0/O/Q/D, 1/I/L, 5/S, 2/Z, 8/B). Un profil cohérent
  (même DDR, même âge) renforce la suggestion, un profil incohérent l'affaiblit sans l'écarter.
- Renumérisation : un même registre est rephotographié à chaque visite. Les nouvelles informations
  s'ajoutent ; une valeur qui CHANGE n'est jamais écrasée sans l'accord de la sage-femme.
"""
from __future__ import annotations

import datetime as dt

CONFUSIONS = str.maketrans({"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "S": "5", "Z": "2", "B": "8"})


def norm_code(code: str | None) -> str:
    return "".join(ch for ch in (code or "").upper() if ch.isalnum())


def _distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _val(dossier: dict | None, page: str, cle: str):
    c = ((dossier or {}).get("pages", {}).get(page, {}).get("champs", {})).get(cle)
    if not c or c.get("statut") not in ("CONNU",) and not c.get("valide_par"):
        return None
    return c.get("valeur")


def profil(dossier: dict | None) -> dict:
    """Résumé NON identifiant, pour aider la sage-femme à reconnaître la patiente."""
    age = _val(dossier, "identification", "age")
    return {"age": age, "ddr": _val(dossier, "grossesse_actuelle", "ddr"),
            "gestite": _val(dossier, "identification", "gestation"),
            "parite": _val(dossier, "identification", "parite")}


def candidats(code: str | None, dossier: dict | None, patientes: list[dict], max_n: int = 3) -> list[dict]:
    c = norm_code(code)
    p_new = profil(dossier)
    out = []
    for p in patientes:
        pc = norm_code(p["code"])
        raisons, score = [], 0.0
        if c and pc == c:
            score, raisons = 1.0, ["même code"]
        elif c and pc.translate(CONFUSIONS) == c.translate(CONFUSIONS):
            score, raisons = 0.8, [f"code presque identique ({p['code']} / {code} : caractères qui se ressemblent)"]
        elif c and len(c) >= 3 and _distance(pc, c) == 1:
            score, raisons = 0.6, [f"code à une lettre près ({p['code']} / {code})"]
        else:
            continue
        prof = p.get("profil", {})
        if p_new.get("ddr") and prof.get("ddr"):
            ecart = abs((dt.date.fromisoformat(p_new["ddr"]) - dt.date.fromisoformat(prof["ddr"])).days)
            if ecart <= 7:
                score += 0.1
                raisons.append("même DDR")
            else:
                score -= 0.2
                raisons.append(f"DDR différente ({ecart} j d'écart)")
        if p_new.get("age") is not None and prof.get("age") is not None:
            if abs(float(p_new["age"]) - float(prof["age"])) <= 1:
                score += 0.05
                raisons.append("même âge")
            else:
                score -= 0.2
                raisons.append(f"âge différent ({prof['age']} / {p_new['age']})")
        out.append({"patiente": p, "score": round(score, 2), "raisons": raisons})
    out.sort(key=lambda x: -x["score"])
    return out[:max_n]


def etiquette(p: dict) -> str:
    prof = p.get("profil", {})
    bits = [f"code {p['code']}"]
    if prof.get("age") is not None:
        bits.append(f"{prof['age']:g} ans" if isinstance(prof["age"], (int, float)) else f"{prof['age']} ans")
    if prof.get("ddr"):
        bits.append("DDR " + dt.date.fromisoformat(prof["ddr"]).strftime("%d/%m/%Y"))
    bits.append(f"{len(p.get('visites', []))} visite(s)")
    return " · ".join(bits)


# --------------------------------------------------------------------------- renumérisation
def differences(ancien: dict | None, nouveau: dict) -> tuple[list[tuple], list[tuple]]:
    """Compare le dossier courant de la patiente et la nouvelle numérisation.
    Renvoie (ajouts, changements) : listes de (page, clé, ancien_champ, nouveau_champ).
    - ajout : rien de sûr avant, une valeur sûre maintenant (ex. nouvelle colonne de visite)
    - changement : deux valeurs sûres différentes -> la sage-femme choisit."""
    ajouts, changements = [], []
    for pt, page in nouveau.get("pages", {}).items():
        for k, c_new in page["champs"].items():
            if c_new.get("statut") != "CONNU" and not c_new.get("valide_par"):
                continue
            c_old = (ancien or {}).get("pages", {}).get(pt, {}).get("champs", {}).get(k)
            sur_avant = c_old and (c_old.get("statut") == "CONNU" or c_old.get("valide_par"))
            if not sur_avant:
                if c_new.get("statut") == "CONNU" and c_new.get("valeur") not in (None, False, ""):
                    ajouts.append((pt, k, c_old, c_new))
            elif c_old.get("valeur") != c_new.get("valeur"):
                changements.append((pt, k, c_old, c_new))
    return ajouts, changements


def fusionner(ancien: dict | None, nouveau: dict, garder_nouveau: set[tuple[str, str]]) -> dict:
    """Dossier courant = ancien + ajouts + changements acceptés. Les pages nouvelles entrent entières."""
    if not ancien:
        return nouveau
    out = {**ancien, "pages": {pt: {**p, "champs": dict(p["champs"])} for pt, p in ancien["pages"].items()}}
    ajouts, changements = differences(ancien, nouveau)
    for pt, page in nouveau.get("pages", {}).items():
        if pt not in out["pages"]:
            out["pages"][pt] = page
    for pt, k, _, c_new in ajouts:
        out["pages"][pt]["champs"][k] = c_new
    for pt, k, _, c_new in changements:
        if (pt, k) in garder_nouveau:
            out["pages"][pt]["champs"][k] = c_new
    return out
