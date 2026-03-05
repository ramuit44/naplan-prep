"""
Adhrit Naplan App - NAPLAN practice for students (Writing, Reading, Language Conventions, Numeracy).
Reading: parse questionnaire PDFs on the fly when user navigates to a year's questions.
Uses DashScope Qwen3-VL when DASHSCOPE_API_KEY is set for heterogeneous question types.
VL-extracted sections are cached as <pdf_stem>.sections.json next to the PDF for reuse.
"""
import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, render_template, send_file, request, abort

load_dotenv()

app = Flask(__name__)
BASE_DIR = Path(__file__).resolve().parent
WRITING_DIR = BASE_DIR / "writing"
READING_DIR = BASE_DIR / "reading"


def extract_year_from_filename(filename: str) -> int | None:
    """Extract year from NAPLAN PDF filename. Returns None if not found."""
    # Pattern 1: 200819_NAPLAN_2008_... or 201119_NAPLAN_2011_...
    m = re.search(r"NAPLAN_(\d{4})", filename, re.I)
    if m:
        return int(m.group(1))
    # Pattern 2: naplan-2012-... or naplan-2016-...
    m = re.search(r"naplan-(\d{4})", filename, re.I)
    if m:
        return int(m.group(1))
    return None


def get_writing_assignments_by_year():
    """Scan writing folder and return {year: [list of {name, filename}]} sorted by year desc."""
    if not WRITING_DIR.is_dir():
        return {}
    by_year = {}
    for f in WRITING_DIR.iterdir():
        if f.suffix.lower() != ".pdf":
            continue
        year = extract_year_from_filename(f.name)
        if year is None:
            continue
        # Human-friendly title: use filename without extension, cleaned up
        name = f.stem.replace("-", " ").replace("_", " ").strip()
        if name.upper().startswith("NAPLAN"):
            name = name[6:].strip()
        entry = {"name": name, "filename": f.name}
        by_year.setdefault(year, []).append(entry)
    # Sort years descending (newest first), sort assignments per year by name
    for year in by_year:
        by_year[year].sort(key=lambda x: x["name"])
    return dict(sorted(by_year.items(), key=lambda x: -x[0]))


def get_reading_by_year():
    """Scan reading folder; return {year: {magazine: filename, questionnaire: filename}}.
    No pre-parsing: questionnaire is parsed on the fly when user opens that year's questions.
    """
    if not READING_DIR.is_dir():
        return {}
    from reading_parser import (
        extract_year_from_reading_filename,
        is_magazine,
        is_questionnaire,
    )
    by_year = {}
    for f in READING_DIR.iterdir():
        if f.suffix.lower() != ".pdf":
            continue
        year = extract_year_from_reading_filename(f.name)
        if year is None:
            continue
        if year not in by_year:
            by_year[year] = {}
        if is_magazine(f.name):
            by_year[year]["magazine"] = f.name
        elif is_questionnaire(f.name):
            by_year[year]["questionnaire"] = f.name
    return dict(sorted(by_year.items(), key=lambda x: -x[0]))


def _reading_sections_cache_path(pdf_path: Path) -> Path:
    """Path for cached sections JSON: same dir as PDF, <stem>.sections.json."""
    return pdf_path.parent / f"{pdf_path.stem}.sections.json"


def _load_reading_sections_cache(cache_path: Path) -> list | None:
    """Load sections from cache file. Returns None if missing or invalid."""
    if not cache_path.is_file():
        return None
    try:
        with open(cache_path, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list) or not data:
            return None
        # Validate shape: list of {title, questions: [{number, text, options, question_type}]}
        for sec in data:
            if not isinstance(sec, dict) or "title" not in sec or "questions" not in sec:
                return None
        return data
    except (json.JSONDecodeError, OSError):
        return None


def _save_reading_sections_cache(cache_path: Path, sections: list) -> None:
    """Save sections (template format) to JSON cache."""
    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(sections, f, ensure_ascii=False, indent=2)
    except OSError:
        pass


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/writing")
def writing():
    by_year = get_writing_assignments_by_year()
    return render_template("writing.html", by_year=by_year)


@app.route("/reading")
def reading():
    by_year = get_reading_by_year()
    return render_template("reading.html", by_year=by_year)


@app.route("/reading/<int:year>/questions")
def reading_questions(year):
    """Parse questionnaire PDF on the fly and render test page.
    If a .sections.json cache exists for that PDF, use it (no LLM call). Otherwise use
    DashScope Qwen3-VL (when DASHSCOPE_API_KEY is set) or text-based parser, and save
    VL results to cache for next time.
    """
    by_year = get_reading_by_year()
    if year not in by_year or "questionnaire" not in by_year[year]:
        abort(404)
    filename = by_year[year]["questionnaire"]
    pdf_path = READING_DIR / filename
    magazine_filename = by_year[year].get("magazine")
    cache_path = _reading_sections_cache_path(pdf_path)
    sections = _load_reading_sections_cache(cache_path)
    used_vl = False
    if not sections and os.environ.get("DASHSCOPE_API_KEY"):
        try:
            from vl_extract_dashscope import extract_questions_from_pdf, vl_sections_to_template_format
            vl_sections = extract_questions_from_pdf(pdf_path)
            sections = vl_sections_to_template_format(vl_sections)
            used_vl = True
        except Exception:
            pass
    if not sections:
        from reading_parser import parse_reading_pdf
        raw_sections = parse_reading_pdf(pdf_path)
        sections = [
            {
                "title": s.title,
                "questions": [
                    {"number": q.number, "text": q.text, "options": q.options or [], "question_type": "multiple_choice"}
                    for q in s.questions
                ],
            }
            for s in raw_sections
        ]
    if used_vl and sections:
        _save_reading_sections_cache(cache_path, sections)
    return render_template(
        "reading_questions.html",
        year=year,
        sections=sections,
        questionnaire_filename=filename,
        magazine_filename=magazine_filename,
    )


@app.route("/view/reading/<path:filename>")
def view_reading_pdf(filename):
    """Serve a reading PDF (magazine or questionnaire) for in-browser viewing."""
    path = READING_DIR / filename
    if not path.is_file() or path.suffix.lower() != ".pdf":
        abort(404)
    try:
        path.resolve().relative_to(READING_DIR.resolve())
    except ValueError:
        abort(404)
    return send_file(
        path,
        mimetype="application/pdf",
        as_attachment=False,
        download_name=path.name,
    )

@app.route("/language-conventions")
def language_conventions():
    return render_template("section_placeholder.html", section="Language Conventions", icon="✏️")

@app.route("/numeracy")
def numeracy():
    return render_template("section_placeholder.html", section="Numeracy", icon="🔢")


@app.route("/view/writing/<path:filename>")
def view_writing_pdf(filename):
    """Serve a writing PDF for in-browser viewing (inline, no download)."""
    path = WRITING_DIR / filename
    if not path.is_file() or path.suffix.lower() != ".pdf":
        abort(404)
    # Ensure path is inside WRITING_DIR (no path traversal)
    try:
        path.resolve().relative_to(WRITING_DIR.resolve())
    except ValueError:
        abort(404)
    return send_file(
        path,
        mimetype="application/pdf",
        as_attachment=False,
        download_name=path.name,
    )


@app.route("/pdf-viewer")
def pdf_viewer():
    """Page that embeds a PDF in an iframe (open in page)."""
    url = request.args.get("url")
    title = request.args.get("title", "Assignment")
    if not url:
        abort(400)
    return render_template("pdf_viewer.html", pdf_url=url, title=title)


if __name__ == "__main__":
    app.run(debug=True, port=5006)
