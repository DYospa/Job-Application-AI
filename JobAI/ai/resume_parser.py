"""
Resume parser: PDF -> structured dict of resume fields.

Fixes vs. the original version:
- Section extraction now stops at the NEXT heading it finds, instead of
  reading all the way to the end of the document. The old code did
  `text.split("Skills")[1]` and then only skipped lines that literally
  started with "Experience" or "Education" -- since most content lines
  don't start with those exact words, every section ended up containing
  everything after it (Skills swallowed Projects, Certifications, etc.).
- Switched PyPDF2 (deprecated, unmaintained) -> pypdf (its successor,
  same project, active maintenance, near-identical API).
- Skills/experience/education parsing handles bullets, commas, and
  simple "Degree, School, Year" style lines.
"""

import re
from pypdf import PdfReader
import fitz

# Headings we recognize. Order doesn't matter here; we detect whichever
# ones actually appear in the document and use them as section boundaries.
KNOWN_HEADINGS = [
    "Summary",
    "Objective",
    "Skills",
    "Technical Skills",
    "Education",
    "Experience",
    "Work Experience",
    "Professional Experience",
    "Projects",
    "Certifications",
    "Certification",
]


def extract_resume_text(pdf_path):
    """
    Extract text from a resume while preserving
    the reading order of one- and two-column layouts.
    """

    document = fitz.open(pdf_path)

    all_pages = []

    for page in document:
        page_text = extract_page_text(page)
        all_pages.append(page_text)

    document.close()

    return "\n".join(all_pages)

def extract_page_text(page):
    """
    Detect whether the page is single-column
    or two-column and extract accordingly.
    """

    words = page.get_text("words")

    if not words:
        return ""

    # Get page dimensions
    page_width = page.rect.width

    # Find words on left and right sides
    left_column = []
    right_column = []

    midpoint = page_width / 2

    for word in words:

        x0 = word[0]
        x1 = word[2]
        text = word[4]

        # Determine which column the word belongs to
        if x1 <= midpoint:
            left_column.append(word)

        elif x0 >= midpoint:
            right_column.append(word)

    # Determine whether this is actually a two-column page
    if has_two_columns(left_column, right_column):

        left_text = words_to_text(left_column)
        right_text = words_to_text(right_column)

        return left_text + "\n" + right_text

    else:
        return page.get_text("text")
    
def words_to_text(words):
    """
    Convert coordinate-based words back into
    properly ordered text.
    """

    # Sort by vertical position first,
    # then horizontal position
    words = sorted(
        words,
        key=lambda word: (word[1], word[0])
    )

    lines = []
    current_line = []
    current_y = None

    for word in words:

        x0 = word[0]
        y0 = word[1]
        text = word[4]

        # Start first line
        if current_y is None:
            current_y = y0
            current_line.append((x0, text))
            continue

        # Same line
        if abs(y0 - current_y) < 5:
            current_line.append((x0, text))

        # New line
        else:
            current_line.sort(key=lambda item: item[0])

            line_text = " ".join(
                item[1] for item in current_line
            )

            lines.append(line_text)

            current_line = [(x0, text)]
            current_y = y0

    # Add final line
    if current_line:
        current_line.sort(key=lambda item: item[0])

        line_text = " ".join(
            item[1] for item in current_line
        )

        lines.append(line_text)

    return "\n".join(lines)

def _find_heading_matches(text, headings=KNOWN_HEADINGS):
    """
    Find every place a known heading appears on its own line.
    Returns a list of (start_index, end_index, heading_name), sorted by
    position in the document.
    """
    matches = []
    for heading in headings:
        # Heading must occupy its own line (optionally with trailing ':').
        # This avoids matching the word "Skills" if it shows up inside a
        # sentence elsewhere in the resume.
        pattern = re.compile(
            r"^[ \t]*" + re.escape(heading) + r"[ \t]*:?[ \t]*$",
            re.IGNORECASE | re.MULTILINE,
        )
        for m in pattern.finditer(text):
            matches.append((m.start(), m.end(), heading))

    matches.sort(key=lambda x: x[0])
    return matches


def _extract_section(text, target_heading, all_matches=None):
    """
    Return the text between `target_heading` and whichever recognized
    heading comes next (or end of document if it's the last section).
    """
    if all_matches is None:
        all_matches = _find_heading_matches(text)

    for i, (start, end, name) in enumerate(all_matches):
        if name.lower() != target_heading.lower():
            continue
        section_start = end
        section_end = all_matches[i + 1][0] if i + 1 < len(all_matches) else len(text)
        return text[section_start:section_end].strip()

    return ""


def _split_list_items(section_text):
    """
    Split a section's text into individual items, handling both
    line-separated and comma/bullet-separated formats.
    """
    if not section_text:
        return []

    # Normalize common bullet characters to newlines first.
    normalized = re.sub(r"[•▪●·\-]\s+", "\n", section_text)

    items = []
    for line in normalized.splitlines():
        line = line.strip().strip(",")
        if not line:
            continue
        # If a line is itself comma-separated (e.g. "Python, SQL, R"),
        # split it further.
        if "," in line and len(line) < 200:
            items.extend(part.strip() for part in line.split(",") if part.strip())
        else:
            items.append(line)

    return items


def extract_name(text):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[0] if lines else ""


def extract_email(text):
    match = re.search(r"[\w.\-]+@[\w.\-]+\.\w+", text)
    return match.group(0) if match else ""


def extract_phone(text):
    match = re.search(r"(\+?\d{1,2}[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}", text)
    return match.group(0) if match else ""


def extract_skills(text, all_matches=None):
    section = _extract_section(text, "Skills", all_matches) or _extract_section(
        text, "Technical Skills", all_matches
    )
    return _split_list_items(section)


def extract_education(text, all_matches=None):
    section = _extract_section(text, "Education", all_matches)
    if not section:
        return {}

    lines = [l.strip() for l in section.splitlines() if l.strip()]
    if not lines:
        return {}

    # Work on the first education entry (most resumes list most-recent first).
    entry_text = lines[0]
    # If degree/school/year are split across the first couple of lines
    # instead of one comma-separated line, join them for parsing.
    if len(lines) > 1 and "," not in entry_text:
        entry_text = ", ".join(lines[:2])

    education = {}

    year_match = re.search(r"(19|20)\d{2}", entry_text)
    if year_match:
        education["graduation"] = year_match.group(0)
        entry_text = entry_text.replace(year_match.group(0), "")

    parts = [p.strip(" ,") for p in re.split(r",|\u2013|-", entry_text) if p.strip(" ,")]

    degree_keywords = ("B.S.", "B.A.", "M.S.", "M.A.", "Bachelor", "Master", "Ph.D", "Associate", "Degree")
    school_keywords = ("University", "College", "Institute", "School")

    for part in parts:
        if any(k.lower() in part.lower() for k in degree_keywords) and "degree" not in education:
            education["degree"] = part
        elif any(k.lower() in part.lower() for k in school_keywords) and "school" not in education:
            education["school"] = part

    # Fallback: if we couldn't classify, keep whatever we have.
    if "degree" not in education and parts:
        education["degree"] = parts[0]
    if "school" not in education and len(parts) > 1:
        education["school"] = parts[1]

    return education


def _looks_like_job_header(line):
    """
    Heuristic for "this line introduces a new job" (e.g. "Title, Company"
    or "Title - Company") vs. a bullet/description line underneath it.

    PDF text extraction frequently drops the blank lines that visually
    separated entries in the original layout, so we can't rely on blank
    lines to tell jobs apart -- we have to guess from line shape instead.
    """
    if not line or len(line) > 100 or line.endswith("."):
        return False
    parts = [p.strip() for p in re.split(r",|\u2013|--| - | at ", line) if p.strip()]
    if len(parts) != 2:
        return False
    return line[0].isupper()


def extract_experience(text, all_matches=None):
    section = (
        _extract_section(text, "Experience", all_matches)
        or _extract_section(text, "Work Experience", all_matches)
        or _extract_section(text, "Professional Experience", all_matches)
    )
    if not section:
        return []

    experience = []
    for line in section.splitlines():
        line = line.strip()
        if not line:
            continue
        if _looks_like_job_header(line):
            title, company = [p.strip() for p in re.split(r",|\u2013|--| - | at ", line)]
            experience.append({"title": title, "company": company, "description": []})
        elif experience:
            # A bullet/description line under the most recent job.
            experience[-1]["description"].append(line)

    return experience


def extract_projects(text, all_matches=None):
    section = _extract_section(text, "Projects", all_matches)
    items = _split_list_items(section)
    return [{"title": item} for item in items]


def extract_certifications(text, all_matches=None):
    section = _extract_section(text, "Certifications", all_matches) or _extract_section(
        text, "Certification", all_matches
    )
    return _split_list_items(section)


def parse_resume(path):
    text = extract_resume_text(path)
    all_matches = _find_heading_matches(text)

    return {
        "name": extract_name(text),
        "email": extract_email(text),
        "phone": extract_phone(text),
        "skills": extract_skills(text, all_matches),
        "education": extract_education(text, all_matches),
        "experience": extract_experience(text, all_matches),
        "projects": extract_projects(text, all_matches),
        "certifications": extract_certifications(text, all_matches),
        "raw_text": text,
    }
