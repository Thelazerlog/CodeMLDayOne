"""Jeu de données de fine-tuning pour le VLM (à entraîner sur Narval).

Les exemples ont EXACTEMENT la forme vue à l'inférence : mosaïque numérotée de recadrages + même
prompt + réponse JSON attendue. Les recadrages sont pris sur des photos dégradées redressées avec la
VRAIE homographie, plus un petit décalage aléatoire (simule l'erreur d'alignement réelle).

Découpage SANS FUITE : par patiente pour les 10 dossiers (1-7 entraînement, 8 validation, 9-10 test),
les pages synthétiques vont à l'entraînement. Le test final doit rester les vraies photos du Drive.

Sortie : data/finetune/{train,val,test}.jsonl + data/finetune/images/*.png
Format : messages « chat » avec une image (compatible ms-swift, LLaMA-Factory, TRL ; voir README).
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import cv2
import numpy as np

from .align import load_all
from .pipeline import _crop, _hint
from .readers.base import mosaic
from .readers.vlm import PROMPT, SYSTEM

SPLIT = {**{p: "train" for p in range(1, 8)}, 8: "val", 9: "test", 10: "test"}


def _answer(g: dict) -> dict:
    st = g["status"]
    if st == "CONNU":
        return {"etat": "lu", "texte": g["value"]}
    if st == "NON_FOURNI":
        return {"etat": "lu", "texte": g["raw"]} if g.get("raw") else {"etat": "vide", "texte": None}
    if st == "INCONNU":
        return {"etat": "lu", "texte": g.get("raw", "?")}
    return {"etat": "illisible", "texte": None}


def build(src_dirs: list[str], out_dir: str = "data/finetune", templates_dir: str = "templates",
          group: int = 8, jitter_px: float = 2.0, seed: int = 0, gt_dir: str = "data/gt") -> dict:
    rng = random.Random(seed)
    tpls = {t.page_type: t for t in load_all(templates_dir)}
    out = Path(out_dir)
    (out / "images").mkdir(parents=True, exist_ok=True)
    files = {k: open(out / f"{k}.jsonl", "w") for k in ("train", "val", "test")}
    counts = {k: 0 for k in files}
    for src in src_dirs:
        for sc in sorted(Path(src).glob("*.json")):
            meta = json.loads(sc.read_text())
            img_path = next((p for p in (sc.with_suffix(".jpg"), sc.with_suffix(".png")) if p.exists()), None)
            if img_path is None or meta.get("page_type") not in tpls:
                continue
            tpl = tpls[meta["page_type"]]
            gt = meta.get("gt") or json.loads(Path(gt_dir, f"patient_{int(meta['patient']):02d}.json")
                                              .read_text())[meta["page_type"]]
            split = "train" if meta.get("synthetic") else SPLIT[int(meta["patient"])]
            img = cv2.imread(str(img_path))
            # vraie homographie gabarit -> image (+ petite erreur simulée)
            M = np.vstack([np.array(meta["M_page_to_template"]), [0, 0, 1]])
            H = np.array(meta.get("H_page_to_photo", np.eye(3))) @ np.linalg.inv(M)
            J = np.eye(3)
            J[0, 2], J[1, 2] = rng.uniform(-jitter_px, jitter_px), rng.uniform(-jitter_px, jitter_px)
            W, Hh = tpl.size
            aligned = cv2.warpPerspective(img, np.linalg.inv(H @ J), (W, Hh), flags=cv2.INTER_CUBIC)
            fields = [f for f in tpl.data["fields"] if f["kind"] == "text" and not f.get("sensitive") and f["key"] in gt]
            rng.shuffle(fields)
            for i in range(0, len(fields), group):
                chunk = fields[i:i + group]
                crops = [_crop(aligned, f["bbox"]) for f in chunk]
                if any(c.size == 0 for c in crops):
                    continue
                name = f"{img_path.stem}_{i // group:03d}.png"
                cv2.imwrite(str(out / "images" / name), mosaic(crops))
                hints = "\n".join(f"{k}. {_hint(f)}" for k, f in enumerate(chunk, 1))
                answer = {"lectures": [{"n": k, **_answer(gt[f["key"]])} for k, f in enumerate(chunk, 1)]}
                ex = {"id": name[:-4], "images": [f"images/{name}"], "messages": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": "<image>\n" + PROMPT.format(n=len(chunk), hints=hints)},
                    {"role": "assistant", "content": json.dumps(answer, ensure_ascii=False)}]}
                files[split].write(json.dumps(ex, ensure_ascii=False) + "\n")
                counts[split] += 1
    for f in files.values():
        f.close()
    # déclaration pour LLaMA-Factory (format « sharegpt » avec images)
    tags = {"role_tag": "role", "content_tag": "content", "user_tag": "user", "assistant_tag": "assistant",
            "system_tag": "system"}
    info = {f"registre_{k}": {"file_name": f"{k}.jsonl", "formatting": "sharegpt",
                              "columns": {"messages": "messages", "images": "images"}, "tags": tags}
            for k in files}
    (out / "dataset_info.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(counts))
    return counts
