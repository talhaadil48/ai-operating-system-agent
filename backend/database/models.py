"""
SQLAlchemy Models for Long-Term Memory (LTM).
"""

import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, Text, DateTime
from backend.database.connection import Base


def utc_now():
    return datetime.now(timezone.utc)


class LongTermMemoryRecord(Base):
    __tablename__ = "long_term_memories"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(255), nullable=False, default="default", index=True)
    category = Column(String(50), nullable=False, default="preference")
    fact = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), default=utc_now)
    updated_at = Column(DateTime(timezone=True), default=utc_now, onupdate=utc_now)

    def to_dict(self):
        return {
            "id": self.id,
            "user_id": self.user_id,
            "category": self.category,
            "fact": self.fact,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
