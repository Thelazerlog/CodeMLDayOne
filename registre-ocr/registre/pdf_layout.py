"""Lecture de la couche vectorielle du PDF spécimen.

Le PDF contient une couche texte et des dessins vectoriels :
- libellés imprimés   : police Helvetica / Helvetica-Bold
- valeurs manuscrites : autres polices (Caveat, Gaegu, ReenieBeanie...), glyphe par glyphe
- cases à cocher      : carrés de ~8 pt tracés en 4 segments
- coches              : segments bleus superposés aux cases
- règles de tableau   : segments horizontaux / verticaux fins

On s'en sert UNIQUEMENT pour fabriquer les gabarits et la vérité terrain.
Le pipeline de production, lui, ne lit que des photos.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pymupdf

PRINT_FONTS = ("Helvetica",)
PAGE_TYPES = [
    "couverture",
    "identification",
    "grossesse_actuelle",
    "accouchement",
    "pp_precoce_mere",
    "pp_precoce_nne",
    "pp_tardif_mere",
    "pp_tardif_nne",
]
PAGES_PER_PATIENT = len(PAGE_TYPES)


@dataclass
class Span:
    text: str
    bbox: tuple[float, float, float, float]  # x0, y0, x1, y1 en points
    printed: bool
    bold: bool = False

    @property
    def cx(self) -> float:
        return (self.bbox[0] + self.bbox[2]) / 2

    @property
    def cy(self) -> float:
        return (self.bbox[1] + self.bbox[3]) / 2


@dataclass
class PageLayout:
    index: int
    page_type: str
    patient: int  # 1..10
    size: tuple[float, float]
    labels: list[Span] = field(default_factory=list)  # texte imprimé
    values: list[Span] = field(default_factory=list)  # texte manuscrit regroupé en valeurs
    boxes: list[tuple] = field(default_factory=list)  # cases à cocher
    ticks: list[tuple] = field(default_factory=list)  # coches (bbox)
    hlines: list[tuple] = field(default_factory=list)
    vlines: list[tuple] = field(default_factory=list)
    background: tuple = (1, 1, 1)


def _values_from_trace(page) -> list[Span]:
    """Valeurs manuscrites = suites de glyphes non-Helvetica à numéros de séquence consécutifs,
    recoupées quand l'écriture saute (autre cellule, autre ligne)."""
    runs: list[list] = []
    last_seq = None
    for t in page.get_texttrace():
        if t["font"].startswith(PRINT_FONTS):
            continue
        if last_seq is None or t["seqno"] != last_seq + 1:
            runs.append([])
        last_seq = t["seqno"]
        for ch in t["chars"]:
            runs[-1].append((chr(ch[0]), ch[3]))
    out = []
    for run in runs:
        for piece in _split_geometric(run):
            text = "".join(c for c, _ in piece).strip()
            boxes = [b for c, b in piece if c.strip()]
            if not text or not boxes:
                continue
            bbox = (min(b[0] for b in boxes), min(b[1] for b in boxes),
                    max(b[2] for b in boxes), max(b[3] for b in boxes))
            out.append(Span(" ".join(text.split()), bbox, printed=False))
    return out


def _split_geometric(run: list, max_gap: float = 8.0, max_dy: float = 6.0) -> list[list]:
    pieces: list[list] = [[]]
    prev = None  # bbox du dernier glyphe visible
    for c, b in run:
        if not c.strip():
            pieces[-1].append((c, b))
            continue
        if prev is not None:
            jump_back = b[0] < prev[2] - 3
            dy = abs((b[1] + b[3]) / 2 - (prev[1] + prev[3]) / 2)
            if jump_back or dy > max_dy or b[0] - prev[2] > max_gap:
                pieces.append([])
        pieces[-1].append((c, b))
        prev = b
    return pieces


def _is_box(d) -> bool:
    """Case à cocher = petit carré fait uniquement de segments horizontaux/verticaux (une croix n'en est pas)."""
    r = d["rect"]
    if not (6 < r.width < 12 and 6 < r.height < 12 and abs(r.width - r.height) < 1.5) or _is_blue(d):
        return False
    if any(it[0] == "re" for it in d["items"]):
        return True
    segs = [it for it in d["items"] if it[0] == "l"]
    return len(segs) >= 4 and all(abs(a.x - b.x) < 0.5 or abs(a.y - b.y) < 0.5 for _, a, b in segs)


def _is_blue(d) -> bool:
    c = d.get("color") or (0, 0, 0)
    return c[2] > 0.4 and c[0] < 0.2


def _merge_boxes(rects: list[tuple], pad: float = 1.0) -> list[tuple]:
    rects = [list(r) for r in rects]
    merged = True
    while merged:
        merged = False
        out: list[list] = []
        for r in rects:
            for o in out:
                if r[0] <= o[2] + pad and o[0] <= r[2] + pad and r[1] <= o[3] + pad and o[1] <= r[3] + pad:
                    o[:] = [min(o[0], r[0]), min(o[1], r[1]), max(o[2], r[2]), max(o[3], r[3])]
                    merged = True
                    break
            else:
                out.append(r)
        rects = out
    return [tuple(r) for r in rects]


def read_page(doc: pymupdf.Document, index: int) -> PageLayout:
    page = doc[index]
    lay = PageLayout(
        index=index,
        page_type=PAGE_TYPES[index % PAGES_PER_PATIENT],
        patient=index // PAGES_PER_PATIENT + 1,
        size=(page.rect.width, page.rect.height),
    )
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            for s in line["spans"]:
                if s["text"].strip() and s["font"].startswith(PRINT_FONTS):
                    lay.labels.append(Span(s["text"].strip(), tuple(s["bbox"]), printed=True,
                                           bold="Bold" in s["font"]))
    lay.values = _values_from_trace(page)

    drawings = page.get_drawings()
    # 1er passage : les cases (carrés ~8 pt)
    box_rects = [d["rect"] for d in drawings if _is_box(d)]

    def _is_tick(d) -> bool:
        """Coche = petit tracé posé sur une case, quelle que soit la couleur d'encre."""
        r = d["rect"]
        if r.width > 20 or r.height > 20 or any(r == b for b in box_rects):
            return False  # trop grand, ou c'est une case elle-même
        for b in box_rects:
            same = max(abs(r.x0 - b.x0), abs(r.y0 - b.y0), abs(r.x1 - b.x1), abs(r.y1 - b.y1)) < 0.6
            if not same and r.x0 < b.x1 + 1.5 and b.x0 < r.x1 + 1.5 and r.y0 < b.y1 + 1.5 and b.y0 < r.y1 + 1.5:
                return True
        return False

    for i, dr in enumerate(drawings):
        r = dr["rect"]
        color = tuple(round(c, 2) for c in (dr.get("color") or (0, 0, 0)))
        if i == 0 and dr.get("fill") is not None and r.width > 500:
            lay.background = tuple(dr["fill"])
            continue
        if _is_blue(dr) or _is_tick(dr):
            lay.ticks.append((r.x0, r.y0, r.x1, r.y1))
            continue
        if _is_box(dr):
            lay.boxes.append((r.x0, r.y0, r.x1, r.y1))
            continue
        if dr.get("fill") is not None and dr.get("color") is None:
            continue  # aplats (bandeaux de section)
        # règles : on décompose chaque tracé en segments (les tableaux sont légèrement inclinés)
        for it in dr["items"]:
            if it[0] == "l":
                (x0, y0), (x1, y1) = (it[1].x, it[1].y), (it[2].x, it[2].y)
                segs = [(x0, y0, x1, y1)]
            elif it[0] == "re":
                q = it[1]
                segs = [(q.x0, q.y0, q.x1, q.y0), (q.x0, q.y1, q.x1, q.y1),
                        (q.x0, q.y0, q.x0, q.y1), (q.x1, q.y0, q.x1, q.y1)]
            else:
                continue
            for x0, y0, x1, y1 in segs:
                dx, dy = abs(x1 - x0), abs(y1 - y0)
                if dx > 8 and dy <= max(3.0, 0.03 * dx):
                    lay.hlines.append((min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)))
                elif dy > 8 and dx <= max(3.0, 0.03 * dy):
                    lay.vlines.append((min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)))
    lay.ticks = _merge_boxes(lay.ticks)
    return lay


def read_document(path: str) -> list[PageLayout]:
    doc = pymupdf.open(path)
    return [read_page(doc, i) for i in range(len(doc))]
