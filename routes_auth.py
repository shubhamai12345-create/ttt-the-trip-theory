"""
TTT – The Trip Theory
Auth + User routes
POST /api/auth/register   — create account + 500 welcome coins
POST /api/auth/login      — return JWT
GET  /api/auth/me         — current user profile
PUT  /api/auth/preferences— update travel preferences
GET  /api/itineraries     — list user's saved trips
POST /api/itineraries     — save trip + earn 100 coins
GET  /api/coins           — coin balance + ledger
POST /api/coins/spend     — spend coins
"""

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session
from typing import Optional, List
from datetime import datetime

from database import get_db, User, Itinerary, Preference, CoinsLedger
from auth import (
    hash_password, verify_password,
    create_access_token,
    get_current_user, get_optional_user,
)

router = APIRouter(prefix="/api", tags=["auth"])


# ─── Pydantic Schemas ─────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    email:        str
    password:     str
    full_name:    Optional[str] = None
    instagram:    Optional[str] = None

class LoginRequest(BaseModel):
    email:    str
    password: str

class PreferencesRequest(BaseModel):
    budget_range:  Optional[str] = None   # budget | mid | luxury | ultra
    travel_style:  Optional[str] = None
    pace:          Optional[str] = None   # slow | moderate | fast
    accommodation: Optional[str] = None

class ItineraryCreate(BaseModel):
    title:       str
    destination: str
    start_date:  Optional[str] = None
    end_date:    Optional[str] = None
    content:     Optional[str] = None
    status:      Optional[str] = "draft"

class SpendCoinsRequest(BaseModel):
    amount: int
    reason: str


# ─── Helper ───────────────────────────────────────────────────────────────────

def _credit_coins(db: Session, user: User, amount: int, reason: str):
    """Add coins to user balance and log to ledger."""
    user.coins_balance = (user.coins_balance or 0) + amount
    entry = CoinsLedger(user_id=user.id, amount=amount, reason=reason)
    db.add(entry)


def _debit_coins(db: Session, user: User, amount: int, reason: str):
    """Deduct coins from user balance and log to ledger. Raises 400 if insufficient."""
    if (user.coins_balance or 0) < amount:
        raise HTTPException(status_code=400, detail=f"Insufficient coins. Balance: {user.coins_balance}, Required: {amount}")
    user.coins_balance -= amount
    entry = CoinsLedger(user_id=user.id, amount=-amount, reason=reason)
    db.add(entry)


def _user_dict(user: User) -> dict:
    return {
        "id":               user.id,
        "email":            user.email,
        "full_name":        user.full_name,
        "coins_balance":    user.coins_balance,
        "travel_persona":   user.travel_persona,
        "instagram_handle": user.instagram_handle,
        "created_at":       user.created_at.isoformat() if user.created_at else None,
    }


# ─── Register ─────────────────────────────────────────────────────────────────

@router.post("/auth/register", status_code=201)
def register(req: RegisterRequest, db: Session = Depends(get_db)):
    # Check duplicate email
    existing = db.query(User).filter(User.email == req.email.lower().strip()).first()
    if existing:
        raise HTTPException(status_code=409, detail="Email already registered")

    if len(req.password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters")

    user = User(
        email           = req.email.lower().strip(),
        full_name       = req.full_name,
        hashed_password = hash_password(req.password),
        instagram_handle= req.instagram,
        coins_balance   = 0,
    )
    db.add(user)
    db.flush()  # get user.id before ledger entry

    # Welcome bonus
    _credit_coins(db, user, 500, "Welcome bonus — thanks for joining TTT!")
    db.commit()
    db.refresh(user)

    token = create_access_token({"sub": str(user.id), "email": user.email})
    return {
        "success": True,
        "token":   token,
        "user":    _user_dict(user),
        "message": "Welcome to TTT! You've received 500 welcome coins 🎉",
    }


# ─── Login ────────────────────────────────────────────────────────────────────

@router.post("/auth/login")
def login(req: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == req.email.lower().strip()).first()
    if not user or not verify_password(req.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    if not user.is_active:
        raise HTTPException(status_code=403, detail="Account deactivated")

    token = create_access_token({"sub": str(user.id), "email": user.email})
    return {
        "success": True,
        "token":   token,
        "user":    _user_dict(user),
    }


# ─── Me ───────────────────────────────────────────────────────────────────────

@router.get("/auth/me")
def get_me(current_user: User = Depends(get_current_user)):
    prefs = current_user.preferences
    return {
        "user": _user_dict(current_user),
        "preferences": {
            "budget_range":  prefs.budget_range  if prefs else None,
            "travel_style":  prefs.travel_style  if prefs else None,
            "pace":          prefs.pace           if prefs else None,
            "accommodation": prefs.accommodation  if prefs else None,
        } if prefs else None,
    }


# ─── Preferences ─────────────────────────────────────────────────────────────

@router.put("/auth/preferences")
def update_preferences(
    req: PreferencesRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    prefs = db.query(Preference).filter(Preference.user_id == current_user.id).first()
    if not prefs:
        prefs = Preference(user_id=current_user.id)
        db.add(prefs)

    if req.budget_range  is not None: prefs.budget_range  = req.budget_range
    if req.travel_style  is not None: prefs.travel_style  = req.travel_style
    if req.pace          is not None: prefs.pace          = req.pace
    if req.accommodation is not None: prefs.accommodation = req.accommodation

    db.commit()
    db.refresh(prefs)
    return {"success": True, "preferences": {
        "budget_range":  prefs.budget_range,
        "travel_style":  prefs.travel_style,
        "pace":          prefs.pace,
        "accommodation": prefs.accommodation,
    }}


# ─── Itineraries ─────────────────────────────────────────────────────────────

@router.get("/itineraries")
def list_itineraries(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    trips = (
        db.query(Itinerary)
        .filter(Itinerary.user_id == current_user.id)
        .order_by(Itinerary.created_at.desc())
        .all()
    )
    return {"itineraries": [
        {
            "id":          t.id,
            "title":       t.title,
            "destination": t.destination,
            "start_date":  t.start_date,
            "end_date":    t.end_date,
            "status":      t.status,
            "created_at":  t.created_at.isoformat() if t.created_at else None,
        }
        for t in trips
    ]}


@router.post("/itineraries", status_code=201)
def save_itinerary(
    req: ItineraryCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    trip = Itinerary(
        user_id     = current_user.id,
        title       = req.title,
        destination = req.destination,
        start_date  = req.start_date,
        end_date    = req.end_date,
        content     = req.content,
        status      = req.status or "draft",
    )
    db.add(trip)
    db.flush()

    # Earn 100 coins for saving a trip
    _credit_coins(db, current_user, 100, f"Saved itinerary: {req.title}")
    db.commit()
    db.refresh(trip)
    db.refresh(current_user)

    return {
        "success":       True,
        "itinerary_id":  trip.id,
        "coins_earned":  100,
        "coins_balance": current_user.coins_balance,
        "message":       f"Trip saved! You earned 100 TTT Coins 🪙",
    }


# ─── Coins ────────────────────────────────────────────────────────────────────

@router.get("/coins")
def get_coins(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    ledger = (
        db.query(CoinsLedger)
        .filter(CoinsLedger.user_id == current_user.id)
        .order_by(CoinsLedger.created_at.desc())
        .limit(50)
        .all()
    )
    return {
        "balance": current_user.coins_balance,
        "ledger": [
            {
                "amount":     e.amount,
                "reason":     e.reason,
                "created_at": e.created_at.isoformat() if e.created_at else None,
            }
            for e in ledger
        ],
    }


@router.post("/coins/spend")
def spend_coins(
    req: SpendCoinsRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if req.amount <= 0:
        raise HTTPException(status_code=400, detail="Amount must be positive")

    _debit_coins(db, current_user, req.amount, req.reason)
    db.commit()
    db.refresh(current_user)

    return {
        "success":         True,
        "coins_spent":     req.amount,
        "balance_after":   current_user.coins_balance,
    }
