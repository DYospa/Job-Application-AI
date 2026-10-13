# JobAI

An AI-assisted job application tool. JobAI learns who you are from your resume, finds relevant jobs, scores how well you match each one, drafts tailored application materials, and tracks everything, while keeping **you** in control of what actually gets submitted.

## Vision

Applying to jobs is repetitive: read a posting, compare it to your background, tweak your resume, rewrite a cover letter, fill out the same form again. JobAI automates the repetitive parts and leaves the decisions to you.

```
Resume ──► Candidate Profile ──► Match against jobs ──► Tailor materials ──► You review ──► Apply ──► Track
```

## Design principles

- **Human in the loop.** JobAI prepares and fills applications; you approve every submission. No blind mass-applying.
- **Truthful by construction.** Generated answers and resume tweaks are grounded in the structured profile parsed from your real resume. The system should never invent experience.
- **Layered pipeline.** Messy input (PDFs, job pages) is cleaned up early. Later stages only ever depend on structured data, never on raw text.
- **Respect the platforms.** Prefer official APIs, company career pages, and assisted form-filling over aggressive scraping or automation that violates a site's terms.

## Roadmap

| Phase | Goal | Output |
|-------|------|--------|
| **1** | Resume → Candidate Profile | Parse a PDF resume into structured data and store it |
| **2** | Job Listings → Job Parser | Turn job descriptions into the same kind of structured shape |
| **3** | Matching | Match score, missing skills, and an explanation per job |
| **4** | Job Discovery | Collect relevant jobs from approved sources |
| **5** | Application Preparation | Tailored resume, cover letter, and answers to application questions |
| **6** | Review and Apply | Approve in a dashboard, then assisted form-filling |
| **7** | Tracking | Status per application: applied, interview, rejected, offer |

## Current status

**Phase 1 is in progress.**

Working:
- Email and phone extraction
- Name extraction, including names split across lines
- Skills, education, projects, and certifications from a real two-column resume
- Candidate profile dataclass and builder
- Database schema and save/load helpers
- Minimal Flask upload endpoint
- Unit tests for the original simple-format parser

In progress:
- Merging the new position-aware PDF parser into `ai/resume_parser.py` (it currently lives in a prototype script)
- Experience parsing: splitting company, title, location, and dates reliably when job titles contain commas
- Updating unit tests to cover multi-column resumes and bullet-style sections

## How resume parsing works

Resumes are harder to parse than they look. Many templates use two columns, and PDF text extraction often returns the content in the wrong order, sometimes placing a section's body before its own heading. JobAI handles this by:

1. Reading every text fragment along with its actual (x, y) position on the page.
2. Detecting columns by looking for the widest horizontal gap between fragments.
3. Rebuilding reading order one column at a time, top to bottom, across all pages.
4. Detecting section headings by keyword (summary, skills, education, experience, projects, certifications, and so on) rather than from a fixed list, and merging headings that wrap onto two lines.
5. Parsing each section with a format-aware extractor that handles bulleted and plain lists.

The result is a plain dict that `candidate_profile.py` turns into a validated `CandidateProfile`.

## Project structure

```
JobAI/
├── ai/
│   ├── resume_parser.py        PDF → raw structured dict
│   └── candidate_profile.py    dict → validated CandidateProfile
├── database/
│   ├── database.py             engine and session setup
│   ├── models.py               SQLAlchemy models
│   └── candidate.py            save/load CandidateProfile
├── tests/
│   ├── test_resume_parser.py
│   └── test_candidate_profile.py
├── app.py                      Flask app (POST /upload-resume)
├── config.py
├── requirements.txt
└── README.md
```

## Tech stack

- **Python 3.11**
- **pypdf** for PDF text extraction
- **SQLAlchemy and SQLite** for storage (swappable for Postgres via `DATABASE_URL`)
- **Flask** for the API layer
- **pytest** for tests
- Planned: an LLM API for matching and tailoring, Playwright for assisted form-filling, a React or HTML dashboard

## Setup

```bash
conda create -n jobai python=3.11
conda activate jobai
cd JobAI
pip install -r requirements.txt
pytest
python app.py
```

Try the pipeline on your own resume:

```python
from ai.resume_parser import parse_resume
from ai.candidate_profile import build_candidate_profile
import json

resume = parse_resume("your_resume.pdf")
profile = build_candidate_profile(resume)
print(json.dumps(profile.to_dict(), indent=2))
```

Or through the API:

```bash
curl -F "resume=@your_resume.pdf" http://127.0.0.1:5000/upload-resume
```

## Known limitations

- Section detection depends on resumes having recognizable headings.
- Parsing is heuristic. Unusual layouts (tables, sidebars with icons, graphics-heavy templates) may need new rules.
- Scanned or image-only PDFs need OCR, which is not implemented.
- Only the first education entry is extracted so far.
- An LLM-based extractor is a natural upgrade later for higher accuracy, at the cost of an API call per resume.

## Next up

1. Finish and merge the Phase 1 parser, then test it on several resumes with different layouts.
2. Phase 2: parse job descriptions into structured requirements.
3. Phase 3: connect the candidate profile to the matcher.
