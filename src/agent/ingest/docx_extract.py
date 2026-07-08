"""Extract structured content from corpus .docx files.

Documents are read body-element by body-element, in order, and grouped into
sections at Heading 1/2 boundaries. Tables are converted to markdown blocks and
kept whole (large tables are split by rows with the header repeated) — several
corpus doc types (ECR, BOM) are 100% tables and would be invisible otherwise.

Known corpus shapes this must handle:
- ~24 files are completely empty            -> extract_sections returns []
- ~22 substantive files have no headings    -> one section with heading_path ()
- the trace matrix is ~2.9M chars of tables -> table row-splitting keeps blocks bounded
- 123 files use Word list numbering (numPr) -> list items get a "- " prefix, since
  Word stores numbers/bullets outside the text

Deliberately skipped (verified against the corpus, 2026-07-08):
- headers/footers (130 files): doc-ID/title banners, duplicate filename metadata
- content controls (2 files): Table of Contents fields only — noise next to sections
- equation math symbols (6 files, ~1% of their text); no text boxes, nested tables,
  tracked changes, footnotes, or hyperlinks exist in the corpus

Run as a module to inspect a single file:

    uv run python -m agent.ingest.docx_extract "data/corpus/ECR-577 ....docx"

or over a directory for corpus-wide stats:

    uv run python -m agent.ingest.docx_extract data/corpus
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph

# A table block larger than this is split by rows (header repeated per piece).
MAX_TABLE_BLOCK_CHARS = 4_000

_HEADING_RE = re.compile(r"heading\s*(\d)", re.IGNORECASE)


@dataclass
class Block:
    kind: str  # "text" | "table"
    text: str


@dataclass
class Section:
    heading_path: tuple[str, ...]  # ("METHODS", "Experimental Procedure"); () = no headings
    blocks: list[Block] = field(default_factory=list)

    @property
    def char_count(self) -> int:
        return sum(len(b.text) for b in self.blocks)


def _heading_level(paragraph: Paragraph) -> int | None:
    """1/2/... for heading-styled paragraphs, else None."""
    style_name = (paragraph.style.name if paragraph.style else "") or ""
    m = _HEADING_RE.search(style_name)
    return int(m.group(1)) if m else None


def _is_list_item(paragraph: Paragraph) -> bool:
    """True for numbered/bulleted paragraphs (numbering lives in numPr, not text)."""
    pPr = paragraph._p.pPr
    return pPr is not None and pPr.numPr is not None


def _cell_text(cell) -> str:
    text = " ".join(cell.text.split())
    return text.replace("|", "\\|")


def table_to_markdown_blocks(
    table: Table, max_chars: int = MAX_TABLE_BLOCK_CHARS
) -> list[str]:
    """Convert a table to one or more markdown blocks.

    Merged cells surface as repeated text (python-docx behavior); acceptable for
    search. Tables exceeding max_chars are split by rows with the header row
    repeated so every piece stays interpretable on its own.
    """
    rows = [[_cell_text(c) for c in row.cells] for row in table.rows]
    rows = [r for r in rows if any(r)]
    if not rows:
        return []

    header, body = rows[0], rows[1:]
    head_md = f"| {' | '.join(header)} |\n| {' | '.join('---' for _ in header)} |"
    if not body:
        return [head_md]

    blocks: list[str] = []
    current: list[str] = []
    current_len = len(head_md)
    for row in body:
        line = f"| {' | '.join(row)} |"
        if current and current_len + len(line) > max_chars:
            blocks.append("\n".join([head_md, *current]))
            current, current_len = [], len(head_md)
        current.append(line)
        current_len += len(line) + 1
    if current:
        blocks.append("\n".join([head_md, *current]))
    return blocks


def extract_sections(path: Path) -> list[Section]:
    """Extract a document as ordered sections of text/table blocks.

    Returns [] for empty documents. Documents without heading styles yield a
    single section with heading_path=() — the caller decides how to chunk those.
    """
    doc = Document(str(path))

    sections: list[Section] = [Section(heading_path=())]
    path_parts: list[str] = []  # current heading path, one entry per level

    for item in doc.iter_inner_content():
        if isinstance(item, Paragraph):
            text = " ".join(item.text.split())
            level = _heading_level(item)
            if level is not None:
                if not text:
                    continue
                path_parts = path_parts[: level - 1] + [text]
                sections.append(Section(heading_path=tuple(path_parts)))
            elif text:
                if _is_list_item(item):
                    text = f"- {text}"
                sections[-1].blocks.append(Block("text", text))
        elif isinstance(item, Table):
            for md in table_to_markdown_blocks(item):
                sections[-1].blocks.append(Block("table", md))

    sections = [s for s in sections if s.blocks]
    return sections


def _dump_file(path: Path) -> None:
    sections = extract_sections(path)
    if not sections:
        print(f"{path.name}: EMPTY")
        return
    total = sum(s.char_count for s in sections)
    print(f"{path.name}: {len(sections)} sections, {total:,} chars\n")
    for s in sections:
        heading = " > ".join(s.heading_path) or "(no heading)"
        kinds = ", ".join(
            f"{sum(1 for b in s.blocks if b.kind == k)} {k}"
            for k in ("text", "table")
            if any(b.kind == k for b in s.blocks)
        )
        print(f"  § {heading}  [{kinds}, {s.char_count:,} chars]")
        preview = s.blocks[0].text[:150].replace("\n", " ⏎ ")
        print(f"      {preview}")


def _corpus_stats(corpus_dir: Path) -> None:
    empty, no_headings, table_blocks, text_blocks, total_chars = [], [], 0, 0, 0
    files = sorted(corpus_dir.glob("*.docx"))
    for f in files:
        sections = extract_sections(f)
        if not sections:
            empty.append(f.name)
            continue
        if all(s.heading_path == () for s in sections):
            no_headings.append(f.name)
        for s in sections:
            total_chars += s.char_count
            table_blocks += sum(1 for b in s.blocks if b.kind == "table")
            text_blocks += sum(1 for b in s.blocks if b.kind == "text")

    print(f"files:            {len(files)}")
    print(f"empty:            {len(empty)}")
    print(f"no headings:      {len(no_headings)}")
    print(f"text blocks:      {text_blocks:,}")
    print(f"table blocks:     {table_blocks:,}")
    print(f"total chars:      {total_chars:,}")


def main() -> int:
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/corpus")
    if target.is_dir():
        _corpus_stats(target)
    elif target.is_file():
        _dump_file(target)
    else:
        print(f"not found: {target}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
