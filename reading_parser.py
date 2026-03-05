"""
Parse NAPLAN reading questionnaire PDFs into sections and multiple-choice questions.
Uses PyMuPDF (fitz) for text extraction.
"""
from __future__ import annotations

import re
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class Question:
    number: str  # "1" or "P1"
    text: str
    options: list[str]  # 4 options for MC; may be empty for other question types


@dataclass
class Section:
    title: str  # "Read Splat! on page 2..."
    questions: list[Question] = field(default_factory=list)


def extract_year_from_reading_filename(filename: str) -> int | None:
    """Extract year from reading PDF filename."""
    m = re.search(r"NAPLAN_(\d{4})", filename, re.I)
    if m:
        return int(m.group(1))
    m = re.search(r"naplan-(\d{4})|e\d-naplan-(\d{4})", filename, re.I)
    if m:
        return int(m.group(1) or m.group(2))
    return None


def is_magazine(filename: str) -> bool:
    return "magazine" in filename.lower()


def is_questionnaire(filename: str) -> bool:
    """True if this is the reading questions PDF (not magazine)."""
    lower = filename.lower()
    if "magazine" in lower:
        return False
    return "reading" in lower and (lower.endswith(".pdf") and "year" in lower or "Reading_Year" in filename)


def parse_reading_pdf(pdf_path: Path) -> list[Section]:
    """Parse a NAPLAN reading questionnaire PDF into sections and questions."""
    try:
        import fitz
    except ImportError:
        return []

    if not pdf_path.is_file() or pdf_path.suffix.lower() != ".pdf":
        return []

    doc = fitz.open(pdf_path)
    full_text = "\n".join(page.get_text() for page in doc)
    doc.close()

    lines = [ln.strip() for ln in full_text.split("\n") if ln.strip()]
    # Drop header/footer lines only (do NOT drop standalone digits - they are question numbers)
    skip_patterns = [
        r"^year\s*\d+$",
        r"^©\s*ACARA",
        r"^YEAR\s+\d+\s+READING",
        r"Time available",
        r"Use 2B or HB",
        r"pencil only",
        r"literacy and numeracy",
        r"^READING$",
        r"^national assessment",
        r"Shade one",
        r"^bubble\.?$",
    ]
    cleaned = []
    for ln in lines:
        if any(re.search(p, ln, re.I) for p in skip_patterns):
            continue
        cleaned.append(ln)

    text = "\n".join(cleaned)
    sections = []

    # Section header: "Read X on page N of the magazine and answer questions A to B."
    section_re = re.compile(
        r"Read\s+(.+?)\s+on\s+page\s+\d+\s+of\s+the\s+magazine\s+and\s+answer\s+(?:questions\s+)?([A-Za-z0-9]+)\s+to\s+([A-Za-z0-9]+)",
        re.DOTALL | re.I,
    )

    # Find section starts (may be split across lines)
    text_normalized = re.sub(r"\n+", " ", text)
    section_matches = list(section_re.finditer(text_normalized))

    for idx, match in enumerate(section_matches):
        section_title = match.group(0).strip()
        if section_title.lower().startswith("read "):
            # Trim "questions X to Y" for display if long
            pass
        start = match.end()
        end = section_matches[idx + 1].start() if idx + 1 < len(section_matches) else len(text_normalized)
        block = text_normalized[start:end].strip()
        questions = _parse_question_block(block)
        sections.append(Section(title=section_title, questions=questions))

    # If no sections found, treat whole text as one block of questions (fallback)
    if not sections and cleaned:
        block = "\n".join(cleaned)
        # Remove any leading "Read ..." line
        block = re.sub(r"^Read\s+.+?\.\s*", "", block, flags=re.DOTALL | re.I)
        questions = _parse_question_block(block)
        if questions:
            sections.append(Section(title="Reading questions", questions=questions))

    return sections


def _parse_question_block(block: str) -> list[Question]:
    """Parse a block of text into Question items (number, text, 4 options)."""
    questions = []
    # Split by question number: " 1 " or " 2 " or " P1 " or " 25 " at word boundary
    parts = re.split(r"\s+(\d+|P\d+)\s+", block)
    # parts[0] may be empty or junk; then [1]=num, [2]=content, [3]=num, [4]=content, ...
    i = 1
    while i + 1 < len(parts):
        num = parts[i].strip()
        content = parts[i + 1].strip()
        i += 2
        if not num or not content:
            continue
        # Content is "Question text? option1. option2. option3. option4." or similar
        # Options often end with period and are on separate lines in original; here they may be space-joined
        q_text, options = _split_question_and_options(content)
        questions.append(Question(number=num, text=q_text or content, options=options))
    return questions


def _split_question_and_options(content: str) -> tuple[str, list[str]]:
    """Heuristic: question ends before 4 short option-like phrases (often sentence fragments)."""
    # Split on period or question-mark followed by space (sentence boundaries)
    bits = re.split(r"[.?]\s+", content)
    bits = [b.strip() for b in bits if b.strip()]
    if len(bits) >= 5:
        option_candidates = bits[-4:]
        question_bits = bits[:-4]
        question_text = ". ".join(question_bits).strip()
        if question_text and all(len(o) < 120 for o in option_candidates):
            options = [(o.rstrip(".") + ".") for o in option_candidates]
            return question_text, options
    if len(bits) == 4:
        # Question + 3 options, or 1 question phrase + 3 options (e.g. "She finds it" merged with q)
        option_candidates = bits[-3:]
        if all(len(o) < 120 for o in option_candidates):
            question_text = bits[0].strip()
            options = [(o.rstrip(".") + ".") for o in option_candidates]
            return question_text, options
    # Newline split (when block has newlines)
    lines = [ln.strip() for ln in content.split("\n") if ln.strip()]
    if len(lines) >= 5:
        question_text = " ".join(lines[:-4]).strip()
        options = lines[-4:]
        if all(len(o) < 150 for o in options):
            return question_text, options
    # Fallback: split on ". " and take last 4 short fragments as options
    parts = re.split(r"\.\s+", content)
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) >= 4:
        opts = [(p.rstrip(".") + ".") for p in parts[-4:] if len(p) < 120]
        if len(opts) == 4:
            return ". ".join(parts[:-4]).strip(), opts
    return content, []
