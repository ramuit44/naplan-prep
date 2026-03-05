# Adhrit Naplan App

A simple web app for students to access NAPLAN year-wise questions and practise. Built with Python (Flask).

## Sections

- **Writing** – Year-wise writing assignments (PDFs from the `writing` folder). Click to open in the same page (no download).
- **Reading** – Year-wise magazine (PDF viewer) + **Questions** (questionnaire built on the fly). If `DASHSCOPE_API_KEY` is set, each questionnaire PDF page is sent to Alibaba DashScope **Qwen3-VL-235B-A22B-Instruct** to extract questions; the app then renders multiple choice, fill-in-the-blank, and number-in-box question types. Otherwise a text-based parser is used.
- **Language Conventions** – Placeholder (add content later).
- **Numeracy** – Placeholder (add content later).

## Setup

```bash
cd /Users/sri/work/projects/Naplan
python3 -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Open **http://127.0.0.1:5000** in your browser.

## Writing content

Put NAPLAN writing PDFs in the `writing` folder. The app groups them by year using the filename (e.g. `NAPLAN_2012` or `naplan-2012-...`). PDFs open in-page in the viewer; they are not downloaded.

## Reading: DashScope (Qwen3-VL) extraction

To use vision-based extraction for reading questionnaires (recommended for heterogeneous question types):

1. Get an [Alibaba DashScope API key](https://help.aliyun.com/zh/model-studio/) and set it:
   ```bash
   export DASHSCOPE_API_KEY="sk-..."
   ```
2. Optional: override the VL model (default: `qwen3-vl-235b-a22b-instruct`):
   ```bash
   export DASHSCOPE_VL_MODEL="qwen3-vl-235b-a22b-instruct"
   ```
3. Optional (outside China): set the base URL for your region:
   ```bash
   export DASHSCOPE_BASE_URL="https://dashscope-intl.aliyuncs.com"   # Singapore
   # or
   export DASHSCOPE_BASE_URL="https://dashscope-us.aliyuncs.com"    # US
   ```

When a student opens **Reading → [Year] → Questions**, the app renders each PDF page to an image, sends it to Qwen3-VL, and builds the questionnaire from the model’s JSON (multiple choice, fill in the blank, write number in box). The app looks for a cached **`<questionnaire_stem>.sections.json`** next to the PDF (e.g. `reading/naplan-2012-final-test---reading-year-3.sections.json`). If found, the questionnaire is loaded from cache and no LLM call is made; if not, VL runs and the result is saved to that JSON for next time.

## Adding more sections

When you have content for Language Conventions or Numeracy, add folders and update `app.py` to scan and serve them similarly to Writing/Reading.
# naplan-prep
