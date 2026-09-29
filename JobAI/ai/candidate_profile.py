"""
Turns the raw dict produced by resume_parser.parse_resume() into a
structured CandidateProfile object.

This is a separate step from parsing on purpose: resume_parser deals with
messy PDF text, candidate_profile deals with clean, validated data. Phase
3 (matching) and beyond should only ever depend on CandidateProfile, never
on the raw parser output.
"""

from dataclasses import dataclass, field, asdict


@dataclass
class CandidateProfile:
    name: str = ""
    email: str = ""
    phone: str = ""
    skills: list = field(default_factory=list)
    education: dict = field(default_factory=dict)
    experience: list = field(default_factory=list)
    projects: list = field(default_factory=list)
    certifications: list = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


def _dedupe_preserve_order(items):
    seen = set()
    result = []
    for item in items:
        key = item.lower().strip() if isinstance(item, str) else item
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def build_candidate_profile(resume: dict) -> CandidateProfile:
    """
    Build a validated CandidateProfile from a raw resume dict (the output
    of resume_parser.parse_resume). Missing fields default sensibly
    instead of raising, since real-world resumes will rarely produce a
    perfectly complete extraction.
    """
    if not resume:
        return CandidateProfile()

    skills = _dedupe_preserve_order(
        [s.strip() for s in resume.get("skills", []) if s and s.strip()]
    )
    certifications = _dedupe_preserve_order(
        [c.strip() for c in resume.get("certifications", []) if c and c.strip()]
    )

    return CandidateProfile(
        name=resume.get("name", "").strip(),
        email=resume.get("email", "").strip(),
        phone=resume.get("phone", "").strip(),
        skills=skills,
        education=resume.get("education", {}) or {},
        experience=resume.get("experience", []) or [],
        projects=resume.get("projects", []) or [],
        certifications=certifications,
    )
