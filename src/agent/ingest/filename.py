"""Parse corpus filenames into document metadata.

Corpus filenames follow the convention:

    {DOC-ID} - {Title}_{Rev}[-signed|-Obsolete][(1)].docx

e.g. "VVPR-P01-151 - Half Value Layer Verification Protocol  Report_B-signed.docx"

The convention is treated as trustworthy but sloppy: separator dashes/spaces vary,
status suffixes appear as "-signed", "- Signed", "_Obsolete", etc., and the revision
letter may be absent entirely (treated as an unversioned, current document).

Run as a module to verify the parser against a corpus directory:

    uv run python -m agent.ingest.filename data/corpus
"""

from __future__ import annotations

import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

# Doc IDs: "VVPR-P01-151", "VVPR-SWV-011", "BOM-055", "3P-P01-32", plus the one
# non-numeric ID "IFU-MX1". Prefix may start with a digit ("3P").
_DOC_ID_RE = re.compile(r"^(IFU-MX1|[A-Z0-9]{2,4}(?:-[A-Z0-9]+)*?-\d+)")

# Revision + optional status at the tail. Real variants: "_B", "_B-signed",
# "_B - Signed", "_C-Signed", "_E_Obsolete", "_F-Obsolete". The status token is
# captured permissively: an unknown token (e.g. "-Draft") keeps the revision and
# produces a warning rather than silently corrupting title + revision.
_REV_RE = re.compile(r"_\s*([A-Z])\s*(?:[-_]\s*([A-Za-z]+))?\s*$")

_KNOWN_STATUSES = {"signed", "obsolete"}

_DUPLICATE_RE = re.compile(r"\(\d+\)\s*$")

_SW_VERSION_RE = re.compile(r"v\d+\.\d+(?:\.\d+)?(?:-\w+)?")


@dataclass(frozen=True)
class DocMeta:
    doc_id: str
    type_prefix: str
    title: str  # per-revision: titles can change between revisions
    revision: str | None  # None = unversioned, treated as current
    is_signed: bool
    is_obsolete: bool
    is_duplicate: bool  # a "(1)" copy of another file
    sw_version: str | None  # software version mentioned in the title, e.g. "v3.4.0"
    filename: str
    warning: str | None = None  # non-fatal parse anomaly, surfaced in the report


class FilenameParseError(ValueError):
    pass


def parse_filename(name: str) -> DocMeta:
    """Parse a corpus filename (with or without .docx) into DocMeta."""
    stem = name.removesuffix(".docx").strip()

    is_duplicate = bool(_DUPLICATE_RE.search(stem))
    stem_clean = _DUPLICATE_RE.sub("", stem).strip()

    id_match = _DOC_ID_RE.match(stem_clean)
    if not id_match:
        raise FilenameParseError(f"no doc ID at start of {name!r}")
    doc_id = id_match.group(1)

    rest = stem_clean[id_match.end() :]

    rev_match = _REV_RE.search(rest)
    revision: str | None = None
    status = ""
    warning: str | None = None
    if rev_match:
        revision = rev_match.group(1)
        status = (rev_match.group(2) or "").lower()
        if status and status not in _KNOWN_STATUSES:
            warning = f"unknown status suffix {rev_match.group(2)!r}"
            status = ""
        rest = rest[: rev_match.start()]

    # Title: strip the " - " separator (sometimes malformed or missing) and tidy up.
    title = rest.strip().lstrip("-").strip()
    title = re.sub(r"\s{2,}", " ", title)

    sw_match = _SW_VERSION_RE.search(title)

    return DocMeta(
        doc_id=doc_id,
        type_prefix=doc_id.split("-")[0],
        title=title,
        revision=revision,
        is_signed=status == "signed",
        is_obsolete=status == "obsolete",
        is_duplicate=is_duplicate,
        sw_version=sw_match.group(0) if sw_match else None,
        filename=name,
        warning=warning,
    )


def parse_corpus(corpus_dir: Path) -> tuple[list[DocMeta], list[tuple[str, str]]]:
    """Parse every .docx filename in a directory.

    Returns (parsed, failures) where failures are (filename, error) pairs.
    """
    parsed: list[DocMeta] = []
    failures: list[tuple[str, str]] = []
    for path in sorted(corpus_dir.glob("*.docx")):
        if path.name.startswith("~$"):  # Word lock files
            continue
        try:
            parsed.append(parse_filename(path.name))
        except FilenameParseError as e:
            failures.append((path.name, str(e)))
    return parsed, failures


def _report(parsed: list[DocMeta], failures: list[tuple[str, str]]) -> None:
    chains: dict[str, list[DocMeta]] = defaultdict(list)
    for meta in parsed:
        chains[meta.doc_id].append(meta)

    prefix_counts = Counter(m.type_prefix for m in parsed)
    multi_rev = {k: v for k, v in chains.items() if len(v) > 1}

    print(f"files parsed:      {len(parsed)}")
    print(f"parse failures:    {len(failures)}")
    print(f"unique doc IDs:    {len(chains)}")
    print(f"revision chains:   {len(multi_rev)} doc IDs with >1 file")
    print(f"signed:            {sum(m.is_signed for m in parsed)}")
    print(f"obsolete:          {sum(m.is_obsolete for m in parsed)}")
    print(f"duplicates '(1)':  {sum(m.is_duplicate for m in parsed)}")
    print(f"missing revision:  {sum(m.revision is None for m in parsed)}")
    print(f"with sw version:   {sum(m.sw_version is not None for m in parsed)}")
    print(f"warnings:          {sum(m.warning is not None for m in parsed)}")
    print(f"type prefixes:     {dict(prefix_counts.most_common())}")

    warned = [m for m in parsed if m.warning]
    if warned:
        print("\nwarnings:")
        for m in warned:
            print(f"  {m.filename}: {m.warning}")

    print("\nrevision chains:")
    for doc_id, metas in sorted(multi_rev.items()):
        revs = ", ".join(
            f"{m.revision or '?'}{'(obs)' if m.is_obsolete else ''}"
            f"{'(dup)' if m.is_duplicate else ''}"
            for m in sorted(metas, key=lambda m: m.revision or "")
        )
        print(f"  {doc_id}: {revs}")

    if failures:
        print("\nFAILURES:")
        for name, err in failures:
            print(f"  {name}: {err}")


def main() -> int:
    corpus_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/corpus")
    if not corpus_dir.is_dir():
        print(f"not a directory: {corpus_dir}", file=sys.stderr)
        return 2
    parsed, failures = parse_corpus(corpus_dir)
    _report(parsed, failures)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
