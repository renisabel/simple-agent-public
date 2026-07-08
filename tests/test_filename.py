"""Table-driven tests for filename parsing.

Cases marked "corpus:" are real filenames from the MedAI corpus, chosen because each
exercises a quirk that exists in the wild. Synthetic cases cover convention drift the
corpus doesn't contain yet (unknown status words, higher duplicate numbers, no rev).
"""

import pytest

from agent.ingest.filename import DocMeta, FilenameParseError, parse_filename


def case(name, **expected):
    return pytest.param(name, expected, id=name[:60])


CASES = [
    # corpus: plain rev, standard separator
    case(
        "BOM-055 - MX1 Top-level assembly_G.docx",
        doc_id="BOM-055", type_prefix="BOM", revision="G",
        title="MX1 Top-level assembly",
    ),
    # corpus: digit-first type prefix
    case(
        "3P-P01-33 - Intertek IEC 60601-1 MX1 Rev F Dielectric and Leakage Summary Report_A.docx",
        doc_id="3P-P01-33", type_prefix="3P", revision="A",
    ),
    # corpus: the one non-numeric doc ID
    case(
        "IFU-MX1 - MX1 Instructions for Use_D.docx",
        doc_id="IFU-MX1", type_prefix="IFU", revision="D",
        title="MX1 Instructions for Use",
    ),
    # corpus: "space - space Signed" status variant
    case(
        "VVPR-P01-081 - Usability Summative Evaluation Protocol and Report_B - Signed.docx",
        revision="B", is_signed=True,
    ),
    # corpus: underscore status variant
    case(
        "MEMO-P01-630 - MX1 Software Requirement Specifications_E_Obsolete.docx",
        revision="E", is_obsolete=True,
    ),
    # corpus: capitalized -Signed variant
    case(
        "MEMO-P01-640 - MX1 OTS SOUP Report_C-Signed.docx",
        revision="C", is_signed=True,
    ),
    # corpus: duplicate "(1)" file
    case(
        "VVPR-SWV-027 - Galden Verification Firmware Verification and Validation Protocol and Report_B(1).docx",
        doc_id="VVPR-SWV-027", revision="B", is_duplicate=True,
    ),
    # corpus: malformed separator (no space after ID's dash)
    case(
        "VVPR-P01-152- Pediatric Filtration Verification Protocol and Report_B-signed.docx",
        doc_id="VVPR-P01-152", revision="B", is_signed=True,
        title="Pediatric Filtration Verification Protocol and Report",
    ),
    # corpus: missing separator dash entirely
    case(
        "VVPR-P01-172 Attenuation Equivalent Detector Verification Protocol  Report_B-Signed.docx",
        doc_id="VVPR-P01-172", revision="B",
        title="Attenuation Equivalent Detector Verification Protocol Report",
    ),
    # corpus: software version in title; rev letter order != chronology
    case(
        "BOM-079 - MX1 Portable X-ray System MAI MX1 Software System v4.0.0_M.docx",
        revision="M", sw_version="v4.0.0",
    ),
    # corpus: pre-release software version
    case(
        "VVPR-P01-271- MX1 MedAI Diagnostic Tool WS-002 and WS-006 v3.1.2-alpha Verification Protocol and Report_B.docx",
        sw_version="v3.1.2-alpha", revision="B",
    ),
    # synthetic: no revision at all -> unversioned, treated as current
    case(
        "MEMO-P01-999 - Some Untracked Note.docx",
        doc_id="MEMO-P01-999", revision=None,
        title="Some Untracked Note",
    ),
    # synthetic: unknown status token keeps the revision, warns, doesn't corrupt title
    case(
        "MEMO-P01-998 - Future Convention_B-Draft.docx",
        revision="B", is_signed=False, is_obsolete=False,
        title="Future Convention",
        warning="unknown status suffix 'Draft'",
    ),
    # synthetic: higher duplicate numbers
    case(
        "MEMO-P01-997 - Copied Twice_A(2).docx",
        revision="A", is_duplicate=True,
    ),
    # synthetic: title containing an underscore must not eat the revision
    case(
        "MEMO-P01-996 - Config_Management Notes_C.docx",
        revision="C", title="Config_Management Notes",
    ),
]


@pytest.mark.parametrize("name,expected", CASES)
def test_parse(name: str, expected: dict):
    meta = parse_filename(name)
    for field, want in expected.items():
        assert getattr(meta, field) == want, field


def test_defaults_are_clean():
    meta = parse_filename("BOM-055 - MX1 Top-level assembly_G.docx")
    assert meta == DocMeta(
        doc_id="BOM-055", type_prefix="BOM", title="MX1 Top-level assembly",
        revision="G", is_signed=False, is_obsolete=False, is_duplicate=False,
        sw_version=None, filename="BOM-055 - MX1 Top-level assembly_G.docx",
    )


def test_unparseable_raises():
    with pytest.raises(FilenameParseError):
        parse_filename("random notes final v2.docx")


def test_whole_corpus_parses():
    """Every real corpus file must parse without failure or warning."""
    from pathlib import Path

    from agent.ingest.filename import parse_corpus

    corpus = Path(__file__).parent.parent / "data" / "corpus"
    if not corpus.is_dir():
        pytest.skip("corpus not present")
    parsed, failures = parse_corpus(corpus)
    assert failures == []
    assert len(parsed) == 189
    assert [m.filename for m in parsed if m.warning] == []
