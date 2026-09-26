from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from .config import settings

connect_args = {}
if settings.database_url.startswith("sqlite"):
    # Needed for SQLite when used from multiple threads (FastAPI's default
    # threadpool for sync endpoints).
    connect_args = {"check_same_thread": False}

engine = create_engine(
    settings.database_url,
    echo=settings.echo_sql,
    future=True,
    connect_args=connect_args,
)

SessionLocal = sessionmaker(
    bind=engine, autoflush=False, autocommit=False, future=True
)


class Base(DeclarativeBase):
    pass


def get_db():
    """FastAPI dependency: yields a request-scoped SQLAlchemy session."""
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()
