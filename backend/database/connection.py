"""
Database Connection Manager using SQLAlchemy.

Manages connection engine, session factory, and table creation for PostgreSQL.
Falls back safely or logs helpful connection error details if PostgreSQL is unreachable.
"""

from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker, Session

from backend.config import settings
from backend.logging_config import get_logger

log = get_logger(__name__)

Base = declarative_base()

_engine = None
_SessionLocal = None


def get_engine():
    global _engine
    if _engine is None:
        try:
            # Check if URI uses postgresql:// scheme
            uri = settings.POSTGRES_URI
            if uri.startswith("postgres://"):
                uri = uri.replace("postgres://", "postgresql://", 1)

            _engine = create_engine(
                uri,
                pool_pre_ping=True,
                pool_size=5,
                max_overflow=10,
                echo=False,
            )
            log.info("[database] Created SQLAlchemy engine for %s", uri.split("@")[-1] if "@" in uri else uri)
        except Exception:
            log.exception("[database] Failed to create SQLAlchemy engine")
            raise
    return _engine


def get_session_factory():
    global _SessionLocal
    if _SessionLocal is None:
        engine = get_engine()
        _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return _SessionLocal


def get_db_session() -> Generator[Session, None, None]:
    """Dependency helper to get a transactional database session."""
    factory = get_session_factory()
    session: Session = factory()
    try:
        yield session
    finally:
        session.close()


def init_db():
    """Create all tables defined in Base models if they don't exist yet."""
    try:
        engine = get_engine()
        Base.metadata.create_all(bind=engine)
        log.info("[database] Database tables initialized successfully.")
    except Exception as exc:
        log.warning("[database] Could not initialize database tables: %s", exc)
