# -*- coding: utf-8 -*-
"""
docs/METODOLOJI.md → docs/METODOLOJI.pdf (Türkçe karakter destekli).
Basit markdown alt kümesi: # / ## başlık, **kalın**, | tablo |, - / 1. liste, ``` kod.
Çalıştırma: python scripts/md_to_pdf.py
"""
import re
from pathlib import Path
from fpdf import FPDF

ROOT = Path(__file__).resolve().parent.parent
MD   = ROOT / "docs" / "METODOLOJI.md"
PDF  = ROOT / "docs" / "METODOLOJI.pdf"
FONT_DIR = Path("C:/Windows/Fonts")


_TR = str.maketrans({
    "─": "-", "│": "|", "┬": "+", "┐": "+", "└": "+", "┘": "+", "├": "+",
    "┤": "+", "┴": "+", "┼": "+", "▼": "v", "→": "->", "≤": "<=", "≥": ">=",
    "×": "x", "•": "-", "≈": "~",
})


def clean(s: str) -> str:
    """Arial'ın render edemediği özel glyph'leri ASCII'ye çevir (Türkçe harfler kalır)."""
    return s.translate(_TR)


def strip_bold(s: str) -> str:
    return clean(s.replace("**", ""))


class Doc(FPDF):
    def header(self): pass
    def footer(self):
        self.set_y(-12)
        self.set_font("dejavu", "", 8)
        self.set_text_color(150)
        self.cell(0, 8, f"AYZEK — Metodoloji Notu · sayfa {self.page_no()}", align="C")
        self.set_text_color(0)


pdf = Doc()
pdf.set_auto_page_break(auto=True, margin=15)
pdf.add_font("dejavu", "",  str(FONT_DIR / "arial.ttf"))
pdf.add_font("dejavu", "B", str(FONT_DIR / "arialbd.ttf"))
pdf.add_page()

lines = MD.read_text(encoding="utf-8").splitlines()
i = 0
while i < len(lines):
    ln = lines[i]
    # Tablo bloğu
    if ln.strip().startswith("|"):
        rows = []
        while i < len(lines) and lines[i].strip().startswith("|"):
            cells = [strip_bold(c.strip()) for c in lines[i].strip().strip("|").split("|")]
            if not all(set(c) <= set("-: ") for c in cells):   # ayraç satırını atla
                rows.append(cells)
            i += 1
        if rows:
            ncol = max(len(r) for r in rows)
            w = pdf.epw / ncol
            for ri, r in enumerate(rows):
                pdf.set_font("dejavu", "B" if ri == 0 else "", 9)
                pdf.set_x(pdf.l_margin)
                for c in (r + [""] * (ncol - len(r))):
                    pdf.cell(w, 7, c[:38], border=1)
                pdf.ln()
        pdf.ln(2)
        continue

    def write(txt, h, size, bold=False):
        pdf.set_font("dejavu", "B" if bold else "", size)
        pdf.set_x(pdf.l_margin)
        try:
            pdf.multi_cell(pdf.epw, h, txt, new_x="LMARGIN", new_y="NEXT")
        except Exception:
            pass   # nadir render sorununda o satırı atla, PDF üretimi sürsün

    if ln.strip().startswith("```"):     # kod bloğu (mimari şema)
        i += 1
        pdf.set_text_color(90)
        while i < len(lines) and not lines[i].strip().startswith("```"):
            write(clean(lines[i]), 4, 8)
            i += 1
        pdf.set_text_color(0); i += 1; pdf.ln(2); continue
    if ln.startswith("# "):
        pdf.ln(1); write(strip_bold(ln[2:]), 8, 15, bold=True); pdf.ln(1)
    elif ln.startswith("## "):
        pdf.ln(1); write(strip_bold(ln[3:]), 7, 12, bold=True); pdf.ln(0.5)
    elif ln.strip() == "":
        pdf.ln(2)
    elif re.match(r"^\s*[-*0-9]+[.)]?\s+", ln):
        write("  - " + strip_bold(re.sub(r"^\s*[-*0-9]+[.)]?\s+", "", ln)), 5.5, 10)
    else:
        write(strip_bold(ln), 5.5, 10)
    i += 1

pdf.output(str(PDF))
print("YAZILDI:", PDF)
