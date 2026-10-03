"""Construit, à partir du PDF spécimen :

1. un gabarit par type de page  -> templates/<type>.json + <type>_blank.png + <type>_overlay.png
   - repères (anchors) : libellés imprimés stables, servent à l'alignement des photos
   - champs : cases à cocher et zones de texte (cellules de tableau, lignes « Libellé : ____ »)
   - zones à masquer : identifiants directs (nom, CIN, téléphone, adresse, mari)
2. la vérité terrain des 10 patientes      -> data/gt/patient_XX.json (sans aucun identifiant)
3. les pages propres rendues en PNG        -> data/clean/patient_XX_<type>.png

Les clés de champs sont générées automatiquement (ligne__colonne ou libellé).
À RELIRE par un humain sur les images *_overlay.png (voir README, « Où intervenir »).
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import pymupdf

from .pdf_layout import PAGE_TYPES, PageLayout, Span, read_document
from .textutil import slug

DPI = 150
S = DPI / 72.0  # points -> pixels

# Libellés dont la valeur est un identifiant direct : jamais lue, jamais stockée.
SENSITIVE_LABELS = {
    "nom_prenom_de_la_parturiente", "cin", "adresse", "telephone", "nom_du_mari", "patiente",
    "profession_2",  # profession du mari (2e « Profession : » de la page)
}
# Renommages explicites des clés générées (à compléter après relecture des overlays).
KEY_RENAMES = {
    "*": {"c1_visite_1": "t1_v1", "c2_visite_2": "t1_v2", "c3_visite_3": "t1_v3",
          "c4_visite_1_2": "t2_v1", "c5_visite_2_2": "t2_v2", "c6_visite_3_2": "t2_v3",
          "7eme_mois": "t3_m7", "8eme_mois": "t3_m8", "9eme_mois": "t3_m9"},
    "grossesse_actuelle": {"rh": "rh_neg", "rh_2": "rh_pos"},
    "identification": {"le": "date_vacc_rubeole", "le_2": "date_vacc_hepatite_b",
                       "1": "vat_1", "2": "vat_2", "3": "vat_3", "4": "vat_4", "5": "vat_5"},
    "pp_precoce_mere": {"t": "temperature", "libre": "pf_motif_refus"},
    "pp_tardif_mere": {"t": "temperature", "libre": "pf_motif_refus"},
    "pp_precoce_nne": {"autres_a_preciser": "signes_graves_autres", "autres_a_preciser_2": "traumatismes_autres"},
    "pp_tardif_nne": {"autres_a_preciser": "signes_graves_autres", "autres_a_preciser_2": "traumatismes_autres"},
}
# Glyphes absents de certaines polices manuscrites (� dans la couche texte, rien à l'image).
GT_REPAIRS = {"Dr\ufffda": "Drâa", "\ufffdC": "°C"}


def _rename(key: str, ptype: str) -> str:
    for scope in ("*", ptype):
        for old, new in KEY_RENAMES.get(scope, {}).items():
            if key == old:
                return new
            if key.endswith("__" + old):
                key = key[: -len(old)] + new
    return key


def _repair_gt(txt: str, vocab: set[str]) -> str:
    """Restaure le texte voulu quand la police n'avait pas le glyphe (é, —, °)."""
    if "\ufffd" not in txt:
        return txt
    if txt.strip() == "\ufffd":
        return "—"
    import re
    pat = re.compile("^" + re.escape(txt).replace("\ufffd", ".") + "$")
    hits = [v for v in vocab if pat.match(v)]
    if len(hits) == 1:
        return hits[0]
    for a, b in GT_REPAIRS.items():
        txt = txt.replace(a, b)
    return txt.replace("\ufffd", "é")


# Libellés imprimés qui changent d'une patiente à l'autre : ni repère, ni champ.
VARIABLE_LABEL_PREFIXES = ("Patiente fictive", "MÈRE —")
SENSITIVE_PRINTED_PREFIXES = ("MÈRE —",)  # en-tête qui contient le nom imprimé


# --------------------------------------------------------------------------- géométrie
def _similarity_to_ref(lay: PageLayout, ref: PageLayout) -> np.ndarray:
    """Transformée (2x3) des coordonnées de `lay` vers celles de `ref`, estimée sur les libellés communs."""
    def uniq(labels):
        c = Counter(l.text for l in labels)
        return {l.text: l for l in labels if c[l.text] == 1}

    a, b = uniq(lay.labels), uniq(ref.labels)
    common = [t for t in a if t in b and not t.startswith(VARIABLE_LABEL_PREFIXES)]
    src = np.float32([[a[t].bbox[0], a[t].bbox[3]] for t in common] + [[a[t].bbox[2], a[t].bbox[1]] for t in common])
    dst = np.float32([[b[t].bbox[0], b[t].bbox[3]] for t in common] + [[b[t].bbox[2], b[t].bbox[1]] for t in common])
    M, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=2.0)
    return M


def _tf_box(M: np.ndarray, b) -> tuple:
    pts = np.float32([[b[0], b[1]], [b[2], b[3]]]).reshape(-1, 1, 2)
    p = cv2.transform(pts, M).reshape(-1, 2)
    return (float(min(p[:, 0])), float(min(p[:, 1])), float(max(p[:, 0])), float(max(p[:, 1])))


def _center_in(b, z, pad=0.0) -> bool:
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    return z[0] - pad <= cx <= z[2] + pad and z[1] - pad <= cy <= z[3] + pad


# --------------------------------------------------------------------------- tableaux
def _line_components(hl, vl, tol=2.0):
    """Regroupe règles horizontales/verticales qui se touchent -> un composant = un tableau/cadre."""
    lines = [("h", l) for l in hl] + [("v", l) for l in vl]
    n = len(lines)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            a, b = lines[i][1], lines[j][1]
            if a[0] - tol <= b[2] and b[0] - tol <= a[2] and a[1] - tol <= b[3] and b[1] - tol <= a[3]:
                parent[find(i)] = find(j)
    comps = defaultdict(list)
    for i in range(n):
        comps[find(i)].append(lines[i])
    return list(comps.values())


def _uniq_sorted(vals, tol=3.0):
    out = []
    for v in sorted(vals):
        if not out or v - out[-1] > tol:
            out.append(v)
    return out


def _table_cells(ref: PageLayout):
    """Retourne une liste de cellules de données {key,row,col,bbox,table} déduites des règles."""
    fields = []
    for t_idx, comp in enumerate(_line_components(ref.hlines, ref.vlines)):
        hs = [l for k, l in comp if k == "h"]
        vs = [l for k, l in comp if k == "v"]
        if len(vs) < 3 or len(hs) < 3:
            continue  # simple cadre, pas un tableau
        xs = _uniq_sorted([(v[0] + v[2]) / 2 for v in vs])
        ys = _uniq_sorted([(h[1] + h[3]) / 2 for h in hs])
        tb = (xs[0], ys[0], xs[-1], ys[-1])
        labels = [l for l in ref.labels if _center_in(l.bbox, tb)]

        def cell_of(sp: Span):
            ci = max(i for i in range(len(xs) - 1) if xs[i] <= sp.bbox[0] + 1) if sp.bbox[0] + 1 >= xs[0] else 0
            ri = max((i for i in range(len(ys) - 1) if ys[i] <= sp.cy), default=0)
            return ri, ci

        grid = {}
        for l in labels:
            grid.setdefault(cell_of(l), []).append(l)
        nrows, ncols = len(ys) - 1, len(xs) - 1
        rows_with_label0 = [r for r in range(nrows) if (r, 0) in grid and not grid[(r, 0)][0].bold]
        has_row_labels = len(rows_with_label0) >= 2

        # en-tête de colonne : libellé le plus bas au-dessus, dans la même colonne
        def col_header(r, c):
            for rr in range(r - 1, -1, -1):
                if (rr, c) in grid:
                    return grid[(rr, c)][0]
            return None

        col_names: dict[int, str] = {}
        seen = Counter()
        for c in range(1, ncols) if has_row_labels else range(ncols):
            h = col_header(nrows, c)
            if h is None:
                continue
            base = slug(h.text, 20)
            seen[base] += 1
            col_names[c] = base if seen[base] == 1 else f"{base}_{seen[base]}"
        # désambiguïser « Visite 1 » (T1) / « Visite 1 » (T2) : suffixe = rang de colonne
        dup = Counter(slug(grid[(r, c)][0].text, 20) for (r, c) in grid if c in col_names)
        for r in range(nrows):
            if has_row_labels:
                if r not in rows_with_label0:
                    continue
                row_label = grid[(r, 0)][0]
                if row_label.bold:
                    continue
                cols = range(1, ncols)
            else:
                if any((r, c) in grid for c in range(ncols)):
                    continue  # ligne d'en-tête
                row_label = None
                cols = range(ncols)
            for c in cols:
                if (r, c) in grid or c not in col_names:
                    continue
                if col_header(r, c) is None:
                    continue
                cname = col_names[c]
                if any(cname.startswith(k) and v > 1 for k, v in dup.items()):
                    cname = f"c{c}_{cname}"
                key = (slug(row_label.text, 24) + "__" + cname) if row_label else cname
                fields.append({
                    "key": key, "kind": "text", "table": f"t{t_idx}",
                    "row": row_label.text if row_label else None,
                    "col": col_header(r, c).text,
                    "bbox_pt": (xs[c] + 1, ys[r] + 1, xs[c + 1] - 1, ys[r + 1] - 1),
                })
    return fields


# --------------------------------------------------------------------------- lignes « Libellé : ____ »
def _underline_fields(ref: PageLayout, taken):
    fields = []
    for h in ref.hlines:
        x0, y, x1 = h[0], (h[1] + h[3]) / 2, h[2]
        if x1 - x0 < 15 or x1 - x0 > 420:
            continue
        cands = [l for l in ref.labels
                 if x0 - 30 <= l.bbox[2] <= x0 + 6 and abs(l.bbox[3] - y) < 7]
        if not cands:
            continue
        lab = max(cands, key=lambda l: l.bbox[2])
        z = (x0, y - 16, x1, y + 3)
        if any(_center_in(z, t) for t in taken):
            continue
        fields.append({"key": slug(lab.text, 40), "kind": "text", "label": lab.text, "bbox_pt": z})
    return fields


# --------------------------------------------------------------------------- cases à cocher
def _checkbox_fields(ref: PageLayout):
    fields = []
    order = sorted(ref.boxes, key=lambda b: (round(b[1] / 6), b[0]))
    ctx_labels = sorted([l for l in ref.labels if l.text.rstrip().endswith(":") or l.bold],
                        key=lambda l: (l.bbox[1], l.bbox[0]))
    for b in order:
        cy = (b[1] + b[3]) / 2
        same = [l for l in ref.labels if abs(l.cy - cy) < 5 and not l.text.startswith(VARIABLE_LABEL_PREFIXES)]
        right = [l for l in same if 0 <= l.bbox[0] - b[2] < 25]
        left = [l for l in same if 0 <= b[0] - l.bbox[2] < 25]
        far_left = [l for l in same if l.bbox[2] <= b[0]]
        if right:
            lab = min(right, key=lambda l: l.bbox[0])
        elif left:
            lab = max(left, key=lambda l: l.bbox[2])
        elif far_left:
            lab = max(far_left, key=lambda l: l.bbox[2])
        else:
            lab = None
        # contexte : uniquement si le libellé est ambigu sur la page (« Normal », « Autres »...)
        ctx_lab = None
        same_left = [l for l in same if l.bbox[2] <= b[0] and l.text.rstrip().endswith(":") and l is not lab]
        if same_left:
            ctx_lab = max(same_left, key=lambda l: l.bbox[2])
        else:
            above = [l for l in ctx_labels if 0 < b[1] - l.bbox[3] < 60 and l.bbox[0] <= b[0] + 10]
            if above:
                ctx_lab = max(above, key=lambda l: l.bbox[3])
        fields.append({"label": lab.text if lab else None,
                       "context": ctx_lab.text if ctx_lab else None,
                       "kind": "checkbox", "bbox_pt": tuple(b)})
    label_count = Counter(f["label"] for f in fields)
    for f in fields:
        name = slug(f["label"], 24) if f["label"] else "case"
        if label_count[f["label"]] > 1 and f["context"]:
            f["key"] = f"{slug(f['context'], 18)}__{name}"
        else:
            f["key"] = name
    return fields


# --------------------------------------------------------------------------- assemblage
def _dedupe(fields):
    seen = Counter()
    for f in sorted(fields, key=lambda f: (f["bbox_pt"][1], f["bbox_pt"][0])):
        seen[f["key"]] += 1
        if seen[f["key"]] > 1:
            f["key"] = f"{f['key']}_{seen[f['key']]}"
    return fields


def _infer_type(values: list[str]) -> str:
    import re
    vals = [v for v in values if v and v not in {"—", "-"}]
    if not vals:
        return "text"
    pats = [
        ("date", r"^\d{1,2}/\d{1,2}/\d{2,4}$"),
        ("bp", r"^\d{2,3}\s*/\s*\d{2,3}$"),
        ("int", r"^\d{1,4}$"),
        ("quantity", r"^\d+([.,]\d+)?\s*[a-zA-Z°/%µ]*[a-zA-Z/]*$"),
    ]
    for name, p in pats:
        if sum(bool(re.match(p, v)) for v in vals) >= 0.8 * len(vals):
            return name
    return "text"


def build(pdf_path: str, out_templates: str, out_gt: str, out_clean: str, ref_patient: int = 1) -> None:
    layouts = read_document(pdf_path)
    doc = pymupdf.open(pdf_path)
    tdir, gdir, cdir = Path(out_templates), Path(out_gt), Path(out_clean)
    for d in (tdir, gdir, cdir):
        d.mkdir(parents=True, exist_ok=True)

    by_type = defaultdict(list)
    for lay in layouts:
        by_type[lay.page_type].append(lay)

    gt = defaultdict(dict)  # patient -> page_type -> key -> {...}
    report = {}
    for ptype in PAGE_TYPES:
        pages = by_type[ptype]
        ref = next(p for p in pages if p.patient == ref_patient)
        Ms = {p.patient: _similarity_to_ref(p, ref) for p in pages}

        # 1) champs
        table_fields = _table_cells(ref)
        taken = [f["bbox_pt"] for f in table_fields]
        kv_fields = _underline_fields(ref, taken)
        cb_fields = _checkbox_fields(ref)
        fields = table_fields + kv_fields + cb_fields

        # 2) valeurs orphelines (vues chez une patiente, hors de toute zone) -> zone de secours
        text_zones = [f for f in fields if f["kind"] == "text"]
        orphans = []
        for p in pages:
            for v in p.values:
                vb = _tf_box(Ms[p.patient], v.bbox)
                if not any(_center_in(vb, f["bbox_pt"], pad=2) for f in text_zones):
                    orphans.append((vb, v.text))
        for vb, txt in orphans:
            if any(_center_in(vb, f["bbox_pt"], pad=2) for f in text_zones):
                continue
            cy = (vb[1] + vb[3]) / 2
            left = [l for l in ref.labels if abs(l.cy - cy) < 7 and l.bbox[2] <= vb[0] + 2]
            lab = max(left, key=lambda l: l.bbox[2]) if left else None
            z = (lab.bbox[2] + 1 if lab else vb[0] - 5, vb[1] - 4, max(vb[2] + 60, (lab.bbox[2] if lab else vb[0]) + 120), vb[3] + 4)
            f = {"key": slug(lab.text, 40) if lab else "libre", "kind": "text", "label": lab.text if lab else None,
                 "bbox_pt": z, "origin": "orphelin"}
            fields.append(f)
            text_zones.append(f)
        fields = _dedupe(fields)
        for f in fields:
            f["key"] = _rename(f["key"], ptype)

        # 3) sensibles
        redact_pt = []
        for f in fields:
            f["sensitive"] = f["key"] in SENSITIVE_LABELS
            if f["sensitive"]:
                redact_pt.append(f["bbox_pt"])
        for l in ref.labels:
            if l.text.startswith(SENSITIVE_PRINTED_PREFIXES):
                redact_pt.append((l.bbox[0] + 30, l.bbox[1] - 2, l.bbox[2] + 120, l.bbox[3] + 2))

        # 4) repères : libellés uniques, présents chez toutes les patientes, non variables
        cnt = Counter(l.text for l in ref.labels)
        presence = Counter()
        for p in pages:
            for t in {l.text for l in p.labels}:
                presence[t] += 1
        anchors = []
        for l in ref.labels:
            if cnt[l.text] != 1 or presence[l.text] < len(pages) or l.text.startswith(VARIABLE_LABEL_PREFIXES):
                continue
            if len(l.text) < 2 or (l.bbox[2] - l.bbox[0]) < 12:
                continue
            anchors.append({"id": slug(l.text, 30), "text": l.text, "bold": l.bold, "bbox_pt": l.bbox})

        # 5) vérité terrain
        vocab = {v.text for lay in layouts for v in lay.values if "\ufffd" not in v.text}
        for p in pages:
            M = Ms[p.patient]
            vals = [(_tf_box(M, v.bbox), v.text) for v in p.values]
            ticks = [_tf_box(M, t) for t in p.ticks]
            rec = {}
            for f in fields:
                if f["sensitive"]:
                    continue  # jamais d'identifiant dans la vérité terrain non plus
                z = f["bbox_pt"]
                if f["kind"] == "checkbox":
                    checked = any(_center_in(t, z, pad=3) for t in ticks)
                    rec[f["key"]] = {"value": checked, "status": "CONNU"}
                else:
                    inside = sorted([(b, t) for b, t in vals if _center_in(b, z, pad=2)], key=lambda x: (round(x[0][1] / 8), x[0][0]))
                    txt = _repair_gt(" ".join(t for _, t in inside).strip(), vocab)
                    if not txt:
                        rec[f["key"]] = {"value": None, "status": "NON_FOURNI"}
                    elif txt in {"—", "-", "–"}:
                        rec[f["key"]] = {"value": None, "status": "NON_FOURNI", "raw": txt}
                    else:
                        rec[f["key"]] = {"value": txt, "status": "CONNU"}
            gt[p.patient][ptype] = rec

        for f in fields:
            if f["kind"] == "text":
                vals = [gt[p.patient][ptype].get(f["key"], {}).get("value") or "" for p in pages]
                if f.get("table") and "__" in f["key"]:
                    # cellule de tableau : même type pour toute la ligne (toutes colonnes, toutes patientes)
                    row = f["key"].split("__")[0]
                    vals = [v.get("value") or "" for p in pages for k, v in gt[p.patient][ptype].items()
                            if k.split("__")[0] == row]
                f["type"] = _infer_type(vals)
            else:
                f["type"] = "bool"

        # 6) export gabarit (pixels à 150 dpi)
        def px(b):
            return [round(v * S, 1) for v in b]

        W, H = round(ref.size[0] * S), round(ref.size[1] * S)
        tpl = {
            "page_type": ptype, "dpi": DPI, "size_px": [W, H],
            "title": ref.labels[0].text,
            "anchors": [{**a, "bbox": px(a["bbox_pt"])} for a in anchors],
            "fields": [{**{k: v for k, v in f.items() if k != "bbox_pt"}, "bbox": px(f["bbox_pt"])} for f in fields],
            "redact": [px(b) for b in redact_pt],
        }
        for a in tpl["anchors"]:
            a.pop("bbox_pt", None)
        (tdir / f"{ptype}.json").write_text(json.dumps(tpl, ensure_ascii=False, indent=1))

        # 7) images : gabarit vierge, overlay de relecture, pages propres
        blank = _render_blank(doc, ref, W, H)
        cv2.imwrite(str(tdir / f"{ptype}_blank.png"), blank)
        cv2.imwrite(str(tdir / f"{ptype}_overlay.png"), _overlay(blank, tpl))
        for p in pages:
            pix = doc[p.index].get_pixmap(dpi=DPI)
            img = np.frombuffer(pix.samples, np.uint8).reshape(pix.h, pix.w, pix.n)[:, :, :3][:, :, ::-1]
            cv2.imwrite(str(cdir / f"patient_{p.patient:02d}_{ptype}.png"), img)
            Mpx = np.array(Ms[p.patient], dtype=float).copy()
            Mpx[:, 2] *= S  # translation pt -> px (partie linéaire inchangée)
            (cdir / f"patient_{p.patient:02d}_{ptype}.json").write_text(json.dumps(
                {"patient": p.patient, "page_type": ptype, "M_page_to_template": Mpx.tolist()}))

        report[ptype] = {"anchors": len(anchors), "checkbox": len(cb_fields),
                         "text": len([f for f in fields if f["kind"] == "text"]),
                         "sensitive": sum(f["sensitive"] for f in fields),
                         "orphan_zones": sum(1 for f in fields if f.get("origin") == "orphelin")}

    for patient, pages in gt.items():
        (gdir / f"patient_{patient:02d}.json").write_text(json.dumps(pages, ensure_ascii=False, indent=1))
    print(json.dumps(report, indent=1, ensure_ascii=False))


def _render_blank(doc, ref: PageLayout, W: int, H: int) -> np.ndarray:
    """Page de référence sans écriture manuscrite ni coches = gabarit vierge."""
    tmp = pymupdf.open()
    tmp.insert_pdf(doc, from_page=ref.index, to_page=ref.index)
    page = tmp[0]
    for v in ref.values:
        page.add_redact_annot(pymupdf.Rect(v.bbox), fill=False)
    for l in ref.labels:  # en-tête variable (nom imprimé, n° de patiente)
        if l.text.startswith(VARIABLE_LABEL_PREFIXES):
            page.add_redact_annot(pymupdf.Rect(l.bbox), fill=False)
    page.apply_redactions(images=pymupdf.PDF_REDACT_IMAGE_NONE, graphics=pymupdf.PDF_REDACT_LINE_ART_NONE)
    pix = page.get_pixmap(dpi=DPI)
    img = np.frombuffer(pix.samples, np.uint8).reshape(pix.h, pix.w, pix.n)[:, :, :3][:, :, ::-1].copy()
    bg = tuple(int(c * 255) for c in ref.background[::-1])
    for t in ref.ticks:  # effacer les coches puis redessiner le carré
        x0, y0, x1, y1 = [int(round(v * S)) for v in t]
        img[max(0, y0 - 2):y1 + 3, max(0, x0 - 2):x1 + 3] = bg
    for b in ref.boxes:
        x0, y0, x1, y1 = [int(round(v * S)) for v in b]
        cv2.rectangle(img, (x0, y0), (x1, y1), (26, 20, 31), 1)
    return cv2.resize(img, (W, H)) if img.shape[:2] != (H, W) else img


def _overlay(blank: np.ndarray, tpl: dict) -> np.ndarray:
    img = blank.copy()
    for a in tpl["anchors"]:
        x0, y0, x1, y1 = map(int, a["bbox"])
        cv2.rectangle(img, (x0, y0), (x1, y1), (0, 160, 0), 1)
    for f in tpl["fields"]:
        x0, y0, x1, y1 = map(int, f["bbox"])
        color = (0, 0, 220) if f["sensitive"] else ((200, 90, 0) if f["kind"] == "checkbox" else (0, 140, 255))
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2 if f["sensitive"] else 1)
        cv2.putText(img, f["key"][:28], (x0, max(8, y0 - 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.28, color, 1, cv2.LINE_AA)
    for r in tpl["redact"]:
        x0, y0, x1, y1 = map(int, r)
        cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 220), 2)
    return img


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", default="data/raw/dossiers_specimen_10_patientes.pdf")
    ap.add_argument("--templates", default="templates")
    ap.add_argument("--gt", default="data/gt")
    ap.add_argument("--clean", default="data/clean")
    a = ap.parse_args()
    build(a.pdf, a.templates, a.gt, a.clean)
