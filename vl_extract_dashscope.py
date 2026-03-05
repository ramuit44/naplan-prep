"""
Extract questions from NAPLAN questionnaire PDFs using Alibaba DashScope + Qwen3-VL.
PDF pages are rendered to images and sent to the vision model on the fly; the model
returns structured questions (multiple choice, fill in the blank, write number in box).
"""
from __future__ import annotations

import base64
import json
import os
import re
from pathlib import Path
from dataclasses import dataclass, field


# DashScope model: Qwen3-VL 235B A22B Instruct (or -thinking variant)
DASHSCOPE_VL_MODEL = os.environ.get("DASHSCOPE_VL_MODEL", "qwen3-vl-235b-a22b-instruct")

# Prompt for the VL model to return structured JSON per page
EXTRACT_PROMPT = """This image is one page from a NAPLAN reading test questionnaire.
Extract every question on this page in order.

For each question return:
- type: one of "multiple_choice" (choose one of several options), "fill_blank" (write a short answer or fill in blank), "number_in_box" (write a number in a box).
- number: the question number as shown (e.g. "1", "2", "P1").
- text: the full question text.
- options: only for type "multiple_choice", a list of the choice strings (e.g. ["She finds it.", "She buys it.", ...]). Omit for fill_blank and number_in_box.

Return valid JSON only, no markdown or explanation. Format:
{"questions": [{"type": "...", "number": "...", "text": "...", "options": [...]}, ...]}

If there are no questions on this page, return: {"questions": []}"""


@dataclass
class VLQuestion:
    type: str  # multiple_choice | fill_blank | number_in_box
    number: str
    text: str
    options: list[str] = field(default_factory=list)


@dataclass
class VLSection:
    title: str  # e.g. "Page 2"
    questions: list[VLQuestion] = field(default_factory=list)


def _pdf_pages_to_images(pdf_path: Path, dpi: int = 150) -> list[bytes]:
    """Render each PDF page to PNG bytes. Returns list of PNG image bytes."""
    try:
        import fitz
    except ImportError:
        return []
    if not pdf_path.is_file() or pdf_path.suffix.lower() != ".pdf":
        return []
    doc = fitz.open(pdf_path)
    images = []
    try:
        for page in doc:
            mat = fitz.Matrix(dpi / 72, dpi / 72)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            images.append(pix.tobytes("png"))
    finally:
        doc.close()
    return images


def _call_dashscope_vl(image_base64: str, prompt: str, api_key: str | None = None) -> str:
    """Send one image + prompt to DashScope MultiModalConversation. Returns assistant text."""
    try:
        import dashscope
        from dashscope import MultiModalConversation
    except ImportError:
        raise RuntimeError("Install dashscope: pip install dashscope")
    api_key = api_key or os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        raise ValueError("Set DASHSCOPE_API_KEY or pass api_key")
    base_url = os.environ.get("DASHSCOPE_BASE_URL", "").rstrip("/")
    if base_url:
        dashscope.base_http_api_url = base_url if base_url.endswith("/api/v1") else f"{base_url}/api/v1"
    data_url = f"data:image/png;base64,{image_base64}"
    messages = [
        {
            "role": "user",
            "content": [
                {"image": data_url},
                {"text": prompt},
            ],
        }
    ]
    response = MultiModalConversation.call(
        api_key=api_key,
        model=DASHSCOPE_VL_MODEL,
        messages=messages,
        stream=False,
    )
    if getattr(response, "status_code", getattr(response, "code", 200)) != 200:
        msg = getattr(response, "message", "") or str(response)
        raise RuntimeError(f"DashScope error: {msg}")
    # response.output.choices[0].message.content can be list of dicts with "text" key
    content = response.output.choices[0].message.content
    if isinstance(content, list) and content:
        text = content[0].get("text", "") if isinstance(content[0], dict) else str(content[0])
    else:
        text = str(content) if content else ""
    return text


def _parse_questions_json(raw: str) -> list[VLQuestion]:
    """Parse model output to list of VLQuestion. Tolerates markdown code block."""
    raw = raw.strip()
    # Strip optional markdown code block
    if "```json" in raw:
        raw = re.sub(r"^.*?```json\s*", "", raw, flags=re.DOTALL)
    if "```" in raw:
        raw = re.sub(r"\s*```.*", "", raw, flags=re.DOTALL)
    raw = raw.strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []
    questions = data.get("questions") or data.get("question") or []
    if not isinstance(questions, list):
        return []
    out = []
    for q in questions:
        if not isinstance(q, dict):
            continue
        qtype = (q.get("type") or "multiple_choice").strip().lower()
        if "multiple_choice" in qtype or "multiple choice" in qtype:
            qtype = "multiple_choice"
        elif "fill" in qtype or "blank" in qtype:
            qtype = "fill_blank"
        elif "number" in qtype or "box" in qtype:
            qtype = "number_in_box"
        number = str(q.get("number") or q.get("num") or "")
        text = str(q.get("text") or q.get("question") or "").strip()
        options = q.get("options") or q.get("choices") or []
        if isinstance(options, list):
            options = [str(o).strip() for o in options]
        else:
            options = []
        out.append(VLQuestion(type=qtype, number=number, text=text, options=options))
    return out


def extract_questions_from_pdf(pdf_path: Path, api_key: str | None = None) -> list[VLSection]:
    """
    Render PDF to images, send each page to DashScope Qwen3-VL, parse JSON and return
    sections (one per page) with typed questions. No caching; runs on the fly.
    """
    images = _pdf_pages_to_images(pdf_path)
    if not images:
        return []
    api_key = api_key or os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        raise ValueError("DASHSCOPE_API_KEY is required for VL extraction")
    sections = []
    for i, png_bytes in enumerate(images):
        page_num = i + 1
        b64 = base64.b64encode(png_bytes).decode("utf-8")
        try:
            raw = _call_dashscope_vl(b64, EXTRACT_PROMPT, api_key=api_key)
        except Exception as e:
            raw = json.dumps({"questions": [{"type": "multiple_choice", "number": "?", "text": f"(Error reading page: {e})", "options": []}]})
        questions = _parse_questions_json(raw)
        if questions or page_num <= 2:  # Include first pages even if no questions (cover/instructions)
            sections.append(VLSection(title=f"Page {page_num}", questions=questions))
    return sections


def vl_sections_to_template_format(sections: list[VLSection]) -> list[dict]:
    """Convert VLSection list to structure the reading_questions template expects (with question_type)."""
    out = []
    for sec in sections:
        out.append({
            "title": sec.title,
            "questions": [
                {
                    "number": q.number,
                    "text": q.text,
                    "options": q.options,
                    "question_type": q.type,
                }
                for q in sec.questions
            ],
        })
    return out
