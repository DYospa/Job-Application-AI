[README.md](https://github.com/user-attachments/files/32820403/README.md)
# JobAI — Phase 1: Resume → Candidate Profile

Turns a PDF resume into a structured `CandidateProfile` and stores it in a database.

## Pipeline

```
resume.pdf
    │
    ▼
ai/resume_parser.py   → raw dict (name, email, skills, education, experience, ...)
    │
    ▼
ai/candidate_profile.py → CandidateProfile (validated, deduped)
    │
    ▼
database/candidate.py  → saved to SQLite via SQLAlchemy models
```

## Setup

```bash
conda create -n jobai python=3.11
conda activate jobai
cd JobAI
pip install -r requirements.txt
```

## Run the tests

```bash
pytest
```

## Try it end-to-end (no web server needed)

```python
from ai.resume_parser import parse_resume
from ai.candidate_profile import build_candidate_profile
from database.database import init_db
from database.candidate import save_candidate_profile

init_db()  # creates jobai.db with the right tables, first time only

resume = parse_resume("path/to/your_resume.pdf")
profile = build_candidate_profile(resume)
candidate_id = save_candidate_profile(profile, resume_path="path/to/your_resume.pdf")

print(profile.to_dict())
print("Saved as candidate", candidate_id)
```

## Or run it through the Flask app

```bash
python app.py
```

Then, from another terminal:

```bash
curl -F "resume=@your_resume.pdf" http://127.0.0.1:5000/upload-resume
```

## Known limitations (fine for Phase 1, worth knowing about)

- **Section detection** relies on the resume having clear headings
  ("Skills", "Education", "Experience", "Projects", "Certifications") each
  on their own line. Resumes that bury these in a sidebar, table, or
  unusual layout will parse poorly. `pypdf`'s text extraction doesn't
  preserve visual layout, only reading order — so two-column resumes in
  particular can come out with lines interleaved oddly.
- **Job title/company splitting** uses a heuristic (short line, comma or
  dash separated, doesn't end in a period) to tell a new job entry apart
  from a bullet point underneath it. It'll misfire on resumes with
  unusual formatting. If this becomes a recurring problem, Phase 3+ is
  the natural place to swap in an LLM-based extractor for higher accuracy
  at the cost of an API call per resume.
- **Education parsing** currently only extracts the first education entry
  found. Multiple degrees will need a small extension to
  `extract_education` (loop over entries instead of taking the first).
- No blank lines survive PDF text extraction in many cases — the
  experience parser is written to not depend on them, but keep this in
  mind if you extend other section parsers later.

## Phase 1 checklist

- [x] `conda activate jobai` + `pip install -r requirements.txt` works
- [x] `parse_resume("resume.pdf")` returns a structured dict
- [x] `build_candidate_profile(resume)` returns a `CandidateProfile`
- [x] `pytest` passes
- [x] Database schema exists and can save/load a candidate
- [x] `python app.py` starts Flask and serves `/upload-resume`

Once this is solid on a few real resumes (yours plus a couple of
different formats/layouts), you're ready for Phase 2: parsing job
descriptions into the same kind of structured shape.
