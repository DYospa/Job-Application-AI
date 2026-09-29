"""
Database engine + session setup.

Uses SQLite for now (zero setup, fine for local dev). Swap DATABASE_URL
for a Postgres URI later without changing any other code -- that's the
whole point of going through SQLAlchemy instead of raw sqlite3 calls.
"""

import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///jobai.db")

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


def init_db():
    """Create all tables. Safe to call repeatedly -- no-ops on existing tables."""
    from database import models  # noqa: F401  (ensures models are registered on Base)
    Base.metadata.create_all(bind=engine)


def get_session():
    """Yield a session; caller is responsible for closing it (or use as a context)."""
    return SessionLocal()
