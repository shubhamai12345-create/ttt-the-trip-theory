"""
TTT – The Trip Theory
Database layer: SQLite via SQLAlchemy
Tables: users, itineraries, preferences, coins_ledger
"""

from sqlalchemy import (
    create_engine, Column, Integer, String, Float,
    Text, DateTime, ForeignKey, Boolean
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from datetime import datetime
import os

# ─── Database path (same folder as app.py) ───────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH  = os.path.join(BASE_DIR, "ttt.db")
DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},  # needed for SQLite + FastAPI
    echo=False,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# ─── Models ──────────────────────────────────────────────────────────────────

class User(Base):
    __tablename__ = "users"

    id                = Column(Integer, primary_key=True, index=True)
    email             = Column(String(255), unique=True, index=True, nullable=False)
    full_name         = Column(String(255), nullable=True)
    hashed_password   = Column(String(255), nullable=False)
    coins_balance     = Column(Integer, default=500)          # 500 welcome coins
    travel_persona    = Column(String(100), nullable=True)    # Explorer, Luxury, etc.
    instagram_handle  = Column(String(100), nullable=True)
    is_active         = Column(Boolean, default=True)
    created_at        = Column(DateTime, default=datetime.utcnow)
    updated_at        = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    itineraries = relationship("Itinerary",   back_populates="user", cascade="all, delete")
    preferences = relationship("Preference",  back_populates="user", cascade="all, delete", uselist=False)
    coins_ledger= relationship("CoinsLedger", back_populates="user", cascade="all, delete")


class Itinerary(Base):
    __tablename__ = "itineraries"

    id          = Column(Integer, primary_key=True, index=True)
    user_id     = Column(Integer, ForeignKey("users.id"), nullable=False)
    title       = Column(String(255), nullable=False)
    destination = Column(String(255), nullable=False)
    start_date  = Column(String(20),  nullable=True)
    end_date    = Column(String(20),  nullable=True)
    content     = Column(Text,        nullable=True)   # full itinerary JSON/text
    status      = Column(String(50),  default="draft") # draft | confirmed | completed
    created_at  = Column(DateTime, default=datetime.utcnow)
    updated_at  = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user = relationship("User", back_populates="itineraries")


class Preference(Base):
    __tablename__ = "preferences"

    id               = Column(Integer, primary_key=True, index=True)
    user_id          = Column(Integer, ForeignKey("users.id"), unique=True, nullable=False)
    budget_range     = Column(String(50),  nullable=True)   # budget | mid | luxury | ultra
    travel_style     = Column(String(100), nullable=True)   # adventure | cultural | relaxation
    pace             = Column(String(50),  nullable=True)   # slow | moderate | fast
    accommodation    = Column(String(100), nullable=True)   # homestay | hotel | resort | villa
    updated_at       = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user = relationship("User", back_populates="preferences")


class CoinsLedger(Base):
    __tablename__ = "coins_ledger"

    id         = Column(Integer, primary_key=True, index=True)
    user_id    = Column(Integer, ForeignKey("users.id"), nullable=False)
    amount     = Column(Integer, nullable=False)    # positive = credit, negative = debit
    reason     = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="coins_ledger")


# ─── Init DB ─────────────────────────────────────────────────────────────────

def init_db():
    """Create all tables if they don't exist."""
    Base.metadata.create_all(bind=engine)


def get_db():
    """FastAPI dependency — yields a DB session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
