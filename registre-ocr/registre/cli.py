"""Ligne de commande.

    python -m registre.cli build-templates            # gabarits + vérité terrain depuis le PDF spécimen
    python -m registre.cli synth --n 400              # pages remplies synthétiques (autres polices, arabe...)
    python -m registre.cli augment --per-page 6       # photos de terrain simulées (géométrie connue)
    python -m registre.cli finetune-dataset           # recadrages + réponses JSON pour un fine-tuning (Narval)
    python -m registre.cli quality photo.jpg          # verdict ACCEPTER / REPRENDRE
    python -m registre.cli check-vlm                  # le serveur local répond-il ?
    python -m registre.cli process p1.jpg p2.jpg ... --code K7Q2 --sage-femme SF-012
    python -m registre.cli evaluate data/augmented --reader vlm|oracle|aucun
    python -m registre.cli calibrate-quality          # seuils de flou à partir des augmentations
    python -m registre.cli record data/demo            # enregistre les vraies lectures du VLM (rejouables sans GPU)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def _reader(kind: str, cfg):
    if kind == "vlm":
        from .readers.vlm import VLMReader
        v = cfg.vlm
        r = VLMReader(v.base_url, v.model, batch=v.batch, scale=v.scale, timeout=v.timeout, allow_remote=v.allow_remote,
                      backend=v.backend, think=v.think, parallel=v.parallel)
        if not r.available():
            print(f"⚠ serveur VLM injoignable sur {v.base_url} : les champs texte partiront en saisie manuelle.",
                  file=sys.stderr)
        return r
    return None


def _dump_debug(reader, out: str) -> None:
    """Réponses brutes du modèle (latence, JSON renvoyé) : à lire pour comprendre une mauvaise lecture.
    Ne contient que des valeurs de zones déjà masquées : aucun identifiant."""
    dbg = getattr(reader, "last_debug", None)
    if dbg:
        Path(out).mkdir(parents=True, exist_ok=True)
        Path(out, "vlm_debug.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in dbg))
        lat = [x["latency_s"] for x in dbg]
        print(f"VLM : {len(lat)} appels, {sum(lat) / len(lat):.1f} s en moyenne -> {out}/vlm_debug.jsonl")


def main(argv=None):
    from . import config as config_mod
    cfg = config_mod.load()
    ap = argparse.ArgumentParser(prog="registre")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("build-templates")
    a = sub.add_parser("augment")
    a.add_argument("--per-page", type=int, default=3)
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--out", default="data/augmented")
    a.add_argument("--patients", type=int, nargs="*")
    a.add_argument("--src", default="data/clean", help="data/clean (10 dossiers) ou data/synth (pages synthétiques)")
    a.add_argument("--pattern", default="*.png", help="ex. « 1-*.jpg » pour augmenter les vraies photos")
    sy = sub.add_parser("synth")
    sy.add_argument("--n", type=int, default=80)
    sy.add_argument("--seed", type=int, default=0)
    sy.add_argument("--out", default="data/synth")
    ft = sub.add_parser("finetune-dataset")
    ft.add_argument("--src", nargs="+", default=["data/augmented"])
    ft.add_argument("--out", default="data/finetune")
    q = sub.add_parser("quality")
    q.add_argument("images", nargs="+")
    sub.add_parser("check-vlm")
    p = sub.add_parser("process")
    p.add_argument("images", nargs="+")
    p.add_argument("--code", default=None, help="code aléatoire écrit par la sage-femme sur le registre")
    p.add_argument("--sage-femme", default="SF-demo")
    p.add_argument("--reader", choices=["vlm", "aucun"], default="vlm")
    p.add_argument("--tesseract", action="store_true", help="second lecteur sur les nombres")
    p.add_argument("--out", default=cfg.out)
    e = sub.add_parser("evaluate")
    e.add_argument("images_dir")
    e.add_argument("--reader", choices=["vlm", "oracle", "aucun"], default="vlm")
    e.add_argument("--tesseract", action="store_true")
    e.add_argument("--limit", type=int)
    e.add_argument("--glob", default="*.json", help="sous-ensemble, ex. « patient_01_grossesse* »")
    e.add_argument("--out", default="out/eval")
    sub.add_parser("calibrate-quality")
    rc = sub.add_parser("record", help="enregistre les lectures du VLM pour les rejouer sans GPU (démo, jury)")
    rc.add_argument("images_dir")
    rc.add_argument("--out", default=None, help="par défaut <images_dir>/lectures.json")
    rc.add_argument("--glob", default="*")
    gs = sub.add_parser("gt-skeleton", help="fichier de vérité terrain à remplir pour une page faite à la main")
    gs.add_argument("page_type")
    gs.add_argument("image", help="chemin de la photo (le JSON est créé à côté)")
    sc = sub.add_parser("sidecars", help="associe des photos réelles à leur vérité terrain (CSV image,patient,page_type)")
    sc.add_argument("csv")
    sc.add_argument("--dir", default="data/drive")
    args = ap.parse_args(argv)

    if args.cmd == "build-templates":
        from .build_templates import build
        build("data/raw/dossiers_specimen_10_patientes.pdf", cfg.templates, "data/gt", "data/clean")
    elif args.cmd == "augment":
        from .augment import generate
        print(generate(args.src, args.out, per_page=args.per_page, seed=args.seed, patients=args.patients,
                       pattern=args.pattern),
              "images générées")
    elif args.cmd == "synth":
        from .synth import generate as synth
        print(synth(args.n, cfg.templates, "fonts", args.out, seed=args.seed), "pages synthétiques")
    elif args.cmd == "finetune-dataset":
        from .finetune_dataset import build
        build(args.src, args.out, cfg.templates)
    elif args.cmd == "quality":
        import cv2
        from .quality import assess
        for f in args.images:
            r = assess(cv2.imread(f))
            print(f"{f}: {'ACCEPTER' if r.accept else 'REPRENDRE'}  score={r.score:.2f}  "
                  f"flou≈{r.metrics['blur_sigma_template']:.2f}px")
            for reason in r.reasons:
                print("   -", reason)
    elif args.cmd == "check-vlm":
        from .readers.vlm import VLMReader
        r = VLMReader(cfg.vlm.base_url, cfg.vlm.model, allow_remote=cfg.vlm.allow_remote)
        print("OK" if r.available() else "INJOIGNABLE", cfg.vlm.base_url, cfg.vlm.model)
        try:  # Ollama : le modèle « réfléchit »-il avant de répondre ?
            import urllib.request
            root = cfg.vlm.base_url.rstrip("/").removesuffix("/v1")
            req = urllib.request.Request(f"{root}/api/show", data=json.dumps({"model": cfg.vlm.model}).encode(),
                                         headers={"Content-Type": "application/json"})
            caps = json.loads(urllib.request.urlopen(req, timeout=10).read()).get("capabilities", [])
            print("capacités :", ", ".join(caps))
            if "vision" not in caps:
                print("⚠ ce modèle ne lit pas les images.")
            if "thinking" in caps:
                print("⚠ modèle à raisonnement : lent et réponses tronquées. Préférez la variante « -instruct » "
                      "(ex. qwen3-vl:8b-instruct).")
        except Exception:
            pass
    elif args.cmd == "process":
        from .pipeline import process_session
        from .readers.secondary import TesseractReader
        reader = _reader(args.reader, cfg)
        t0 = time.time()
        d, results = process_session(args.images, reader,
                                     TesseractReader() if args.tesseract else None, cfg.templates,
                                     args.sage_femme, args.code, args.out)
        _dump_debug(reader, args.out)
        print(f"Durée : {time.time() - t0:.1f} s pour {len(args.images)} photo(s)")
        print(json.dumps(d.resume(), ensure_ascii=False, indent=1))
        print(f"\n{len(d.questions)} question(s) pour la sage-femme :")
        for q in d.questions[:30]:
            print(f"  [{q['priorite']}] {q['type']:15s} {q['texte'][:150]}")
        print(f"\nDossier écrit dans {args.out}/dossier_{d.dossier_id}.json")
    elif args.cmd == "evaluate":
        from .evaluate import oracle_factory, run, to_markdown
        from .readers.secondary import TesseractReader
        if args.reader == "oracle":
            factory = oracle_factory()
            print("⚠ lecteur SIMULÉ : teste la plomberie, ne mesure pas la lecture.", file=sys.stderr)
        else:
            r = _reader(args.reader, cfg)
            factory = lambda gt: r  # noqa: E731
        rep = run(args.images_dir, factory, TesseractReader() if args.tesseract else None, cfg.templates,
                  limit=args.limit, out=args.out, pattern=args.glob)
        if args.reader == "vlm":
            _dump_debug(r, args.out)
        print(to_markdown(rep))
    elif args.cmd == "gt-skeleton":
        tpl = json.loads(Path(cfg.templates, f"{args.page_type}.json").read_text())
        gt = {}
        for f in tpl["fields"]:
            if f.get("sensitive"):
                continue
            gt[f["key"]] = ({"value": False, "status": "CONNU"} if f["kind"] == "checkbox"
                            else {"value": None, "status": "NON_FOURNI"})
        out = Path(args.image).with_suffix(".json")
        out.write_text(json.dumps({"patient": None, "page_type": args.page_type, "severity": "manuel",
                                   "gt": gt}, ensure_ascii=False, indent=1))
        print(f"{out} : {len(gt)} champs à compléter (cases : true/false ; texte : value + status CONNU).")
    elif args.cmd == "sidecars":
        import csv
        n = 0
        for row in csv.DictReader(open(args.csv, encoding="utf-8")):
            img = Path(args.dir, row["image"])
            if not img.exists():
                print("introuvable :", img, file=sys.stderr)
                continue
            img.with_suffix(".json").write_text(json.dumps(
                {"patient": int(row["patient"]), "page_type": row["page_type"], "severity": row.get("note", "reel")}))
            n += 1
        print(n, "fichiers d'association écrits")
    elif args.cmd == "record":
        import hashlib
        import cv2
        import numpy as np
        from .align import load_all
        from .pipeline import process_image
        from .readers.cache import RecordingReader
        vlm, templates = _reader("vlm", cfg), load_all(cfg.templates)
        out = Path(args.out or Path(args.images_dir, "lectures.json"))
        table = json.loads(out.read_text()) if out.exists() else {}
        files = sorted(p for p in Path(args.images_dir).glob(args.glob) if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        for i, p in enumerate(files, 1):
            raw = p.read_bytes()
            sha = hashlib.sha256(raw).hexdigest()
            if sha in table:
                continue
            rec = RecordingReader(vlm)
            res = process_image(cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR), templates, rec,
                                image_bytes=raw)
            table[sha] = {"image": p.name, "modele": cfg.vlm.model, **rec.dump()}
            out.write_text(json.dumps(table, ensure_ascii=False, default=str))
            print(f"[{i}/{len(files)}] {p.name}: page={res.page.page_type if res.page else 'non lue'} "
                  f"zones={len(rec.lectures)} appels_libres={len(rec.appels)}", flush=True)
        print(f"{len(table)} photo(s) enregistrée(s) -> {out}")
    elif args.cmd == "calibrate-quality":
        from .calibrate import calibrate_quality
        calibrate_quality("data/augmented")


if __name__ == "__main__":
    main()
