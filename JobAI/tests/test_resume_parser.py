"""
Tests target the string-processing functions directly with a fixed
sample of "already extracted PDF text", rather than round-tripping
through an actual PDF file. That keeps these tests fast and independent
of any particular resume file, and isolates parsing-logic bugs from
PDF-extraction quirks (worth testing separately -- see
test_extract_resume_text_from_real_pdf below for that).
"""

import os
import pytest

from ai.resume_parser import (
    extract_name,
    extract_email,
    extract_phone,
    extract_skills,
    extract_education,
    extract_experience,
    extract_projects,
    extract_certifications,
    extract_resume_text,
    parse_resume,
)

SAMPLE_TEXT = """Doron Yospa
doronyospa@gmail.com | (555) 123-4567

Skills
Python, R, SQL, C, Bash
Tableau, Machine Learning, Deep Learning
Artificial Intelligence, Data Science, Database Management

Education
B.S. Data Science, Georgia State University, 2026

Experience
Claims Specialist, State Farm
Handled customer claims and data entry.
Bartender, Puttshack
Managed bar operations and customer service.

Projects
Deep Learning with Medical Imaging
Job Scheduling Assistant
Traffic Sign Recognition Using Deep Learning

Certifications
Google Professional Data Analytics Certificate
"""


def test_extract_name():
    assert extract_name(SAMPLE_TEXT) == "Doron Yospa"


def test_extract_email():
    assert extract_email(SAMPLE_TEXT) == "doronyospa@gmail.com"


def test_extract_phone():
    assert extract_phone(SAMPLE_TEXT) == "(555) 123-4567"


def test_extract_skills_does_not_leak_into_other_sections():
    skills = extract_skills(SAMPLE_TEXT)
    assert "Python" in skills
    assert "Database Management" in skills
    # Regression test for the original bug: Skills used to swallow
    # everything after it, including Education/Experience/Projects.
    assert "Georgia State University" not in skills
    assert "Claims Specialist" not in skills
    assert "Deep Learning with Medical Imaging" not in skills


def test_extract_education():
    education = extract_education(SAMPLE_TEXT)
    assert education["school"] == "Georgia State University"
    assert education["degree"] == "B.S. Data Science"
    assert education["graduation"] == "2026"


def test_extract_experience_finds_both_jobs():
    experience = extract_experience(SAMPLE_TEXT)
    assert len(experience) == 2
    assert experience[0]["title"] == "Claims Specialist"
    assert experience[0]["company"] == "State Farm"
    assert experience[1]["title"] == "Bartender"
    assert experience[1]["company"] == "Puttshack"


def test_extract_projects():
    projects = extract_projects(SAMPLE_TEXT)
    titles = [p["title"] for p in projects]
    assert "Deep Learning with Medical Imaging" in titles
    assert len(projects) == 3


def test_extract_certifications():
    certs = extract_certifications(SAMPLE_TEXT)
    assert certs == ["Google Professional Data Analytics Certificate"]
    # Regression test: Certifications used to run off the end of the
    # doc since it's the last section -- fine here, but should still
    # not include content from an earlier section.
    assert "Python" not in certs


@pytest.mark.skipif(
    not os.path.exists(os.path.join(os.path.dirname(__file__), "sample_resume.pdf")),
    reason="No sample PDF present; run tests/generate_sample_pdf.py first if you want this test.",
)
def test_extract_resume_text_from_real_pdf():
    """Optional end-to-end check against an actual PDF file, if present."""
    path = os.path.join(os.path.dirname(__file__), "sample_resume.pdf")
    text = extract_resume_text(path)
    assert "Doron Yospa" in text

    profile = parse_resume(path)
    assert profile["name"] == "Doron Yospa"
    assert "Python" in profile["skills"]
