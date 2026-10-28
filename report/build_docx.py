"""
Builds chapter4_results_analysis.docx from chapter4_results_analysis.md (both in this folder).

    python report/build_docx.py

Supports the Markdown used in the chapter: headings, paragraphs, bullet and numbered lists,
tables, images ![caption](path), block quotes and inline **bold**, *italic* and `code`.
"""
import os
import re

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt
from PIL import Image

# Paths are relative to this folder, so the script works from any working directory
HERE = os.path.dirname(os.path.abspath(__file__))
SOURCE = os.path.join(HERE, "chapter4_results_analysis.md")
TARGET = os.path.join(HERE, "chapter4_results_analysis.docx")
FONT = "Times New Roman"
INLINE = re.compile(r"(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)")


def add_runs(paragraph, text, bold=False, size=None):
    for part in INLINE.split(text):
        if not part:
            continue
        if part.startswith("**"):
            run = paragraph.add_run(part[2:-2])
            run.bold = True
        elif part.startswith("*"):
            run = paragraph.add_run(part[1:-1])
            run.italic = True
            run.bold = bold
        elif part.startswith("`"):
            run = paragraph.add_run(part[1:-1])
            run.font.name = "Consolas"
            run.bold = bold
        else:
            run = paragraph.add_run(part)
            run.bold = bold
        if run.font.name is None:
            run.font.name = FONT
        if size:
            run.font.size = size


def add_heading(doc, text, level):
    heading = doc.add_heading(level=level)
    add_runs(heading, text, bold=True, size=Pt({1: 16, 2: 14}.get(level, 12)))
    for run in heading.runs:
        run.font.color.rgb = None  # black instead of the theme colour


def add_table(doc, rows):
    cells = [[c.strip() for c in row.strip().strip("|").split("|")] for row in rows]
    header, body = cells[0], [r for r in cells[1:] if not set("".join(r)) <= set(":- ")]
    table = doc.add_table(rows=1 + len(body), cols=len(header))
    table.style = "Table Grid"
    for i, row in enumerate([header] + body):
        for j, text in enumerate(row):
            paragraph = table.cell(i, j).paragraphs[0]
            add_runs(paragraph, text, bold=(i == 0), size=Pt(10))
    doc.add_paragraph()


def create_report():
    doc = Document()
    style = doc.styles["Normal"]
    style.font.name = FONT
    style.font.size = Pt(12)

    with open(SOURCE, encoding="utf-8") as f:
        lines = f.read().splitlines()

    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        if line.startswith("|"):
            block = []
            while i < len(lines) and lines[i].startswith("|"):
                block.append(lines[i])
                i += 1
            add_table(doc, block)
            continue
        i += 1
        if not line.strip() or line.strip() == "---":
            continue
        heading = re.match(r"(#+) (.*)", line)
        bullet = re.match(r"\s*\*\s+(.*)", line)
        numbered = re.match(r"\s*(\d+\.)\s+(.*)", line)
        image = re.match(r"!\[(.*)\]\((.*)\)", line)
        if image:
            # 6 inches wide, unless that would make a tall screenshot overflow the page
            path = os.path.join(HERE, image.group(2))
            with Image.open(path) as picture:
                aspect = picture.height / picture.width
            doc.add_picture(path, width=Inches(min(6.0, 7.0 / aspect)))
            doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
            caption = doc.add_paragraph()
            caption.alignment = WD_ALIGN_PARAGRAPH.CENTER
            add_runs(caption, f"*{image.group(1)}*", size=Pt(10))
        elif heading:
            add_heading(doc, heading.group(2), len(heading.group(1)))
        elif bullet:
            add_runs(doc.add_paragraph(style="List Bullet"), bullet.group(1))
        elif numbered:
            # Word's "List Number" style keeps counting across separate lists, so keep the Markdown numbers
            paragraph = doc.add_paragraph()
            paragraph.paragraph_format.left_indent = Pt(18)
            paragraph.paragraph_format.first_line_indent = Pt(-18)
            add_runs(paragraph, f"{numbered.group(1)}\t{numbered.group(2)}")
        elif line.startswith(">"):
            paragraph = doc.add_paragraph()
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            add_runs(paragraph, line.lstrip("> "))
        else:
            paragraph = doc.add_paragraph()
            paragraph.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            add_runs(paragraph, line)

    doc.save(TARGET)
    print(f"Document saved as {TARGET}")


if __name__ == "__main__":
    create_report()
