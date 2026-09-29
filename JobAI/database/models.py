"""
ORM models for Phase 1: storing a parsed candidate profile.

Schema:

    Candidate
      ├── Skill          (many)
      ├── Education      (one-to-many, though usually just one entry)
      ├── Experience      (many)
      │     └── description stored as newline-joined text
      ├── Project        (many)
      └── Certification  (many)

Kept intentionally normalized (separate tables per list field) rather than
dumping JSON blobs into columns, since Phase 3 (matching) will want to
query/filter on individual skills, and JSON columns make that painful in
plain SQLite.
"""

from datetime import datetime, timezone
from sqlalchemy import Column, Integer, String, ForeignKey, DateTime, Text
from sqlalchemy.orm import relationship

from database.database import Base


class Candidate(Base):
    __tablename__ = "candidates"

    id = Column(Integer, primary_key=True)
    name = Column(String(255), default="")
    email = Column(String(255), default="")
    phone = Column(String(50), default="")
    resume_path = Column(String(500), default="")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    skills = relationship("Skill", back_populates="candidate", cascade="all, delete-orphan")
    education_entries = relationship("Education", back_populates="candidate", cascade="all, delete-orphan")
    experience_entries = relationship("Experience", back_populates="candidate", cascade="all, delete-orphan")
    projects = relationship("Project", back_populates="candidate", cascade="all, delete-orphan")
    certifications = relationship("Certification", back_populates="candidate", cascade="all, delete-orphan")


class Skill(Base):
    __tablename__ = "skills"

    id = Column(Integer, primary_key=True)
    candidate_id = Column(Integer, ForeignKey("candidates.id"), nullable=False)
    name = Column(String(255), nullable=False)

    candidate = relationship("Candidate", back_populates="skills")


class Education(Base):
    __tablename__ = "education"

    id = Column(Integer, primary_key=True)
    candidate_id = Column(Integer, ForeignKey("candidates.id"), nullable=False)
    school = Column(String(255), default="")
    degree = Column(String(255), default="")
    graduation = Column(String(20), default="")

    candidate = relationship("Candidate", back_populates="education_entries")


class Experience(Base):
    __tablename__ = "experience"

    id = Column(Integer, primary_key=True)
    candidate_id = Column(Integer, ForeignKey("candidates.id"), nullable=False)
    title = Column(String(255), default="")
    company = Column(String(255), default="")
    description = Column(Text, default="")

    candidate = relationship("Candidate", back_populates="experience_entries")


class Project(Base):
    __tablename__ = "projects"

    id = Column(Integer, primary_key=True)
    candidate_id = Column(Integer, ForeignKey("candidates.id"), nullable=False)
    title = Column(String(255), default="")

    candidate = relationship("Candidate", back_populates="projects")


class Certification(Base):
    __tablename__ = "certifications"

    id = Column(Integer, primary_key=True)
    candidate_id = Column(Integer, ForeignKey("candidates.id"), nullable=False)
    name = Column(String(255), default="")

    candidate = relationship("Candidate", back_populates="certifications")
