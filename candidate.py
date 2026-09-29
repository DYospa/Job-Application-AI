"""
Bridges ai.candidate_profile.CandidateProfile <-> database rows.

Keeps the ORM details out of the AI layer, and keeps the AI layer's data
shape out of the database layer, so either can change independently.
"""

from database.database import get_session
from database.models import Candidate, Skill, Education, Experience, Project, Certification
from ai.candidate_profile import CandidateProfile


def save_candidate_profile(profile: CandidateProfile, resume_path: str = "") -> int:
    """Persist a CandidateProfile to the database. Returns the new candidate's id."""
    session = get_session()
    try:
        candidate = Candidate(
            name=profile.name,
            email=profile.email,
            phone=profile.phone,
            resume_path=resume_path,
        )

        for skill_name in profile.skills:
            candidate.skills.append(Skill(name=skill_name))

        if profile.education:
            candidate.education_entries.append(
                Education(
                    school=profile.education.get("school", ""),
                    degree=profile.education.get("degree", ""),
                    graduation=profile.education.get("graduation", ""),
                )
            )

        for job in profile.experience:
            description = job.get("description", [])
            if isinstance(description, list):
                description = "\n".join(description)
            candidate.experience_entries.append(
                Experience(
                    title=job.get("title", ""),
                    company=job.get("company", ""),
                    description=description,
                )
            )

        for project in profile.projects:
            candidate.projects.append(Project(title=project.get("title", "")))

        for cert_name in profile.certifications:
            candidate.certifications.append(Certification(name=cert_name))

        session.add(candidate)
        session.commit()
        return candidate.id
    finally:
        session.close()


def get_candidate_profile(candidate_id: int) -> CandidateProfile:
    """Load a CandidateProfile back out of the database."""
    session = get_session()
    try:
        candidate = session.get(Candidate, candidate_id)
        if candidate is None:
            return None

        education = {}
        if candidate.education_entries:
            edu = candidate.education_entries[0]
            education = {"school": edu.school, "degree": edu.degree, "graduation": edu.graduation}

        return CandidateProfile(
            name=candidate.name,
            email=candidate.email,
            phone=candidate.phone,
            skills=[s.name for s in candidate.skills],
            education=education,
            experience=[
                {
                    "title": e.title,
                    "company": e.company,
                    "description": e.description.split("\n") if e.description else [],
                }
                for e in candidate.experience_entries
            ],
            projects=[{"title": p.title} for p in candidate.projects],
            certifications=[c.name for c in candidate.certifications],
        )
    finally:
        session.close()
