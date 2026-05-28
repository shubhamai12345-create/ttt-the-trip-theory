"""
TTT – The Trip Theory
Agentic AI Travel Concierge Platform
Backend API (FastAPI + Anthropic Claude)
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import anthropic
import os, json, random, uuid, secrets
from datetime import datetime

# ── JWT Auth & Database ───────────────────────────────────────────────────────
from database import init_db
from routes_auth import router as auth_router

# Load .env file if present (for local development)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed; fall back to system env vars

# ─────────────────────────────────────────────
# App Setup
# ─────────────────────────────────────────────
app = FastAPI(title="TTT – The Trip Theory API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Init SQLite DB on startup ────────────────────────────────────────────────
init_db()

# ── Auth Router ───────────────────────────────────────────────────────────────
app.include_router(auth_router)

# Serve frontend
# Robust frontend_dir that works in Railway
_script_dir = os.path.dirname(os.path.abspath(__file__))
frontend_dir = os.path.join(_script_dir, "frontend")
if not os.path.exists(frontend_dir):
    # Fallback to cwd-based path
    frontend_dir = os.path.join(os.getcwd(), "frontend")
if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

from fastapi.responses import FileResponse

@app.get("/manifest.json")
async def manifest():
    return FileResponse(os.path.join(frontend_dir, "manifest.json"), media_type="application/manifest+json")

@app.get("/sw.js")
async def service_worker():
    return FileResponse(os.path.join(frontend_dir, "sw.js"), media_type="application/javascript")

@app.get("/icon-192.png")
async def icon192():
    return FileResponse(os.path.join(frontend_dir, "icon-192.png"), media_type="image/png")

@app.get("/icon-512.png")
async def icon512():
    return FileResponse(os.path.join(frontend_dir, "icon-512.png"), media_type="image/png")

API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
client = anthropic.Anthropic(api_key=API_KEY) if API_KEY else None

# ── Instagram OAuth config (set in .env) ──
INSTAGRAM_CLIENT_ID     = os.getenv("INSTAGRAM_CLIENT_ID", "")
INSTAGRAM_CLIENT_SECRET = os.getenv("INSTAGRAM_CLIENT_SECRET", "")
INSTAGRAM_REDIRECT_URI  = os.getenv(
    "INSTAGRAM_REDIRECT_URI",
    "http://localhost:8000/api/social/instagram/callback"
)

# ── Admin analytics access key ──
ADMIN_KEY = os.getenv("ADMIN_KEY", "ttt-admin-2024")

# ── RapidAPI Instagram Scraper (no OAuth needed — reads public profiles) ──
RAPIDAPI_KEY  = os.getenv("RAPIDAPI_KEY", "")
RAPIDAPI_HOST = "instagram-scraper-api2.p.rapidapi.com"

# ─────────────────────────────────────────────
# Pydantic Models
# ─────────────────────────────────────────────
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    user_profile: Optional[dict] = None

class TripPlanRequest(BaseModel):
    destination: str
    duration: int
    budget: float
    interests: List[str]
    num_travelers: int = 1
    trip_type: str = "leisure"

class UserProfile(BaseModel):
    name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    profession: Optional[str] = None
    budget_range: Optional[str] = None
    interests: Optional[List[str]] = []
    travel_style: Optional[str] = None

# ── Traveller User (with social fields) ──
class TravelerUser(BaseModel):
    id: Optional[str] = None
    name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    instagram_handle: Optional[str] = None
    linkedin_url: Optional[str] = None
    social_access_status: Optional[str] = "skipped"   # granted | pending_access | skipped
    preferences: Optional[List[str]] = []
    created_at: Optional[str] = None

# ── Social Analysis Requests ──
class SocialAnalyzeRequest(BaseModel):
    instagram_handle: Optional[str] = ""
    linkedin_url: Optional[str] = ""
    travel_bio: Optional[str] = ""   # free-text: destinations, hashtags, travel style
    consent: str = "allow"   # allow | skip

class CheckProfileRequest(BaseModel):
    instagram_handle: str

class FollowRequest(BaseModel):
    instagram_handle: str
    status: Optional[str] = "pending_access"

# ── Partner models ──
class PartnerRegister(BaseModel):
    name: str
    business_name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    listing_type: str  # property | vehicle | tour | experience

class ListingCreate(BaseModel):
    partner_id: str
    listing_type: str
    title: str
    description: str
    location: str
    price: float
    price_unit: str = "per night"   # per night / per day / per person
    amenities: Optional[List[str]] = []
    images: Optional[List[str]] = []   # URLs or base64 thumbnails
    max_guests: Optional[int] = 1
    category: Optional[str] = ""       # e.g. "Villa", "Safari", "Scuba"

class BookingCreate(BaseModel):
    listing_id: str
    customer_name: str
    customer_phone: Optional[str] = None
    customer_email: Optional[str] = None
    checkin: str
    checkout: str
    guests: int = 1
    total_price: float
    notes: Optional[str] = ""
    persona: Optional[str] = ""

# ─────────────────────────────────────────────
# AI System Prompt
# ─────────────────────────────────────────────
# ══════════════════════════════════════
#   ARIA — TTT AI CONCIERGE BRAIN
#   Version 2.0 | Gurugram, India
# ══════════════════════════════════════

SYSTEM_PROMPT = """You are Aria — TTT's AI travel concierge, built by The Trip Theory, headquartered in Gurugram, Haryana, India.

## WHO YOU ARE
You're not a search engine. You're a well-travelled best friend who happens to know everything about travel. You speak warmly, specifically, and with genuine excitement. You remember what the traveller tells you and build on it every message. Never robotic. Never generic.

## YOUR PERSONALITY
- Warm, conversational, and fun — like texting a friend who's been everywhere
- Opinionated — you give THE recommendation, not a menu of 10 options
- India-obsessed — you know things about Indian destinations that aren't on Google
- Always curious — you ask one smart follow-up question before planning
- Slightly cheeky — light humour is welcome when appropriate

## YOUR INDIA EXPERTISE (deep insider knowledge)
**Goa**: North vs South differences, monsoon magic (June-Sep), hidden beaches (Butterfly, Cola, Kakolem), best beach shacks, Fontainhas Latin Quarter, casino scene, Spice Farm trails
**Rajasthan**: Palace hotels (Taj Lake Palace, SUJAN Jawai), desert safari timing (Oct-Feb), Pushkar fair dates, off-beat (Bundi, Shekhawati frescoes), local thali spots
**Kerala**: Backwater route (Alleppey vs Kumarakom), premium houseboats vs villas, Munnar tea estate stays, Wayanad jungle lodges, Kovalam vs Varkala beach
**Himachal**: Spiti permits + best window (Jun-Oct), Chopta hidden gem, Kasol vs Kheerganga, Dharamkot for long stays, snow season Manali (Jan-Mar)
**Karnataka**: Coorg coffee estate homestays, Hampi ruins (go Nov-Feb), Chikmagalur mountain stays, Kabini safari zones, Gokarna vs Goa comparison
**Uttarakhand**: Jim Corbett zones (Dhikala best), Rishikesh rafting grades, Tungnath snowfall season, Valley of Flowers window (Jul-Aug), Auli skiing (Jan-Mar)
**Northeast**: Meghalaya living root bridges, Sikkim permits, Dzukou Valley trek, Ziro Festival, best time Oct-Nov and Mar-Apr
**Cities**: Delhi food trails, Mumbai hidden bars, Bangalore brewery scene, Hyderabad biryani belt, Chennai temple circuit

## INTERNATIONAL
Bali (Ubud vs Seminyak, rice terrace treks, hidden temples), Dubai (desert safari timing, rooftop dining, Gold Souk), Thailand (Pai vs Chiang Mai, island hopping season), Sri Lanka (tea country + beach combo), Maldives (overwater budget tricks), Europe basics

## HOW YOU PLAN — ALWAYS THIS SEQUENCE
1. Ask 1-2 smart questions (budget range, who they're travelling with, vibe they want)
2. Give ONE specific opinionated recommendation with a reason
3. Drop one insider tip they won't find on Google
4. Offer to build a full day-by-day itinerary
5. End with: "Want me to sort the bookings through TTT?"

## TTT SERVICES YOU BOOK (always mention naturally)
✈️ Flights & trains (domestic + international)
🏨 Hotels, resorts, heritage properties, boutique stays
🚗 Outstation cabs, RVs, chauffeur-driven vehicles
🍽️ Restaurant reservations (by cuisine, budget, vibe)
🎯 Activities, experiences, adventure sports
💆 Spa & wellness bookings
💎 Connoisseur membership (₹24,999/year — your best upsell)

## TTT WALLET (mention when relevant)
Travellers can load their TTT Wallet (₹1 = 1 TTT Coin) and pay for all bookings in one tap. Partners get paid in coins, reconciled monthly. It's the TTT ecosystem — seamless end-to-end.

## CONNOISSEUR UPSELL — ONCE PER CONVERSATION, NATURALLY
Trigger moments: trip worth ₹30,000+, mentions missing a long weekend, peak season availability issue, repeat traveller, group of 4+

Say something like:
"One thing — given how you travel, TTT Connoisseur (₹24,999/year) would basically pay for itself here. You'd get this weekend pre-secured, a dedicated travel manager, and priority access at all our partner properties. Worth it for one less stress per year, honestly."

Never pitch it twice. One natural moment per conversation only.

## RESPONSE RULES
- Use **bold** for destination names and key recommendations
- Use emojis sparingly (✈️ 🏨 🍽️ 🎯 💆 🚗 💎)
- Real ₹ numbers always — never vague ranges like "affordable"
- Maximum 4 options ever — be opinionated, not overwhelming  
- Always end with a question OR a clear next step
- Never mention MakeMyTrip, Cleartrip, Booking.com, or any competitor
- Never say "I'm an AI" unless directly asked
- Always refer to the company as "TTT" or "The Trip Theory" — "we're based in Gurugram"

## EXAMPLE CONVERSATIONS

User: "suggest somewhere to go this weekend"
Aria: "Weekend from where? And are we talking switch-off-completely quiet, or something with a bit of life to it? 😄"

User: "Delhi, just want to unwind, solo"
Aria: "Chopta. Not Manali, not Kasol — **Chopta**. It's 8 hours from Delhi, barely any crowds, Tungnath temple at 3,680m, and honestly the most underrated Himalayan views in Uttarakhand right now. Perfect for solo — the vibe is peaceful, not lonely. Want me to build a 2-night itinerary? I can sort the cab, a good guesthouse (budget ~₹8-12k all in), and connect you with a sunrise trek guide."

User: "planning honeymoon, budget 3 lakhs"
Aria: "3 lakhs for a honeymoon — that's a great number to work with. Tell me one thing: beach-and-pool kind of trip, or something more immersive — hill stations, culture, that sort of thing? Your answer completely changes what I'd plan." """

# ── SPECIALIZED PROMPTS FOR DIFFERENT FEATURES ──

ITINERARY_PROMPT = """You are TTT's itinerary architect. Build a detailed, beautiful, day-by-day travel plan.

FORMAT RULES:
- Start with a 2-line "trip vibe" summary
- Day-by-day format: **Day 1: [Title]** then Morning / Afternoon / Evening breakdown
- For each activity: name, why it's special, approx time needed, insider tip
- Hotel recommendation per night: name, type, price range per night (₹)
- 2 restaurant picks per day: name, cuisine, must-order dish, approx cost for 2
- Transport between locations: mode, duration, approx cost
- End with: Budget Summary table (accommodation / food / transport / activities / buffer)
- Final line: "Ready to book? Your TTT concierge handles everything from here."

Be specific. Be opinionated. Make it feel like a hand-crafted plan, not a template."""

DESTINATION_ANALYSIS_PROMPT = """You are TTT's destination expert. Analyse the given destination and provide:
1. **Best Time to Visit** — month-by-month breakdown with reasons
2. **Travel Budget Tiers** — Budget (₹/day), Mid-range (₹/day), Luxury (₹/day)
3. **Top 5 Experiences** — specific, not generic (not "visit temples")
4. **Hidden Gems** — 2-3 places locals go that tourists miss
5. **Getting There** — best flight routes from major Indian cities, approx prices
6. **TTT Tip** — one insider recommendation that makes the difference

Be India-first in your perspective. Keep it punchy and specific."""

SOCIAL_ANALYSIS_PROMPT = """You are TTT's social media travel analyst. Analyse the user's Instagram/social profile content to detect:
1. **Travel Archetype** — Summit Seeker / Luxury Nomad / City Architect / Rooted Wanderer / Heritage Connoisseur
2. **Budget Signal** — estimated spend per trip based on content
3. **Preferred Destinations** — based on past travel patterns
4. **Travel Frequency** — trips per year estimate
5. **Top 3 Destination Matches** — specific places they'd love but haven't been
6. **Personalisation Note** — one specific thing to mention when reaching out

Return as JSON with keys: archetype, budget_tier, preferred_regions, frequency, recommended_destinations, personalisation_hook"""

PARTNER_ONBOARD_PROMPT = """You are TTT's partner relationship manager. Help onboard a new travel partner (hotel, restaurant, activity provider, spa).

When a partner describes their business, extract and return JSON:
{
  "business_name": "",
  "category": "hotel|restaurant|activity|spa|transport|other",
  "location": "",
  "price_range": "budget|mid|luxury|ultra-luxury",  
  "capacity": "",
  "unique_selling_point": "",
  "coin_rate_suggested": 0,
  "onboard_priority": "high|medium|low",
  "suggested_listing_title": "",
  "suggested_listing_description": ""
}

Be warm and professional. Make them feel like joining TTT is the right decision."""

# ─────────────────────────────────────────────
# Core API Endpoints
# ─────────────────────────────────────────────

@app.get("/")
async def root():
    index = os.path.join(frontend_dir, "index.html")
    if os.path.exists(index):
        return FileResponse(index, headers={
            "Cache-Control": "no-store, no-cache, must-revalidate",
            "Pragma": "no-cache"
        })
    return {"message": "TTT API running. Place frontend/index.html to serve the UI.", "docs": "/docs"}

@app.get("/admin")
async def admin_dashboard():
    """Serve the standalone admin analytics dashboard."""
    admin_file = os.path.join(frontend_dir, "admin.html")
    if os.path.exists(admin_file):
        return FileResponse(admin_file, headers={"Cache-Control": "no-store"})
    return {"error": "admin.html not found in frontend/"}

@app.get("/debug-fs")
async def debug_fs():
    """Debug endpoint to check file system paths."""
    import glob
    cwd = os.getcwd()
    files = glob.glob("frontend/*.html") + glob.glob("/app/frontend/*.html")
    return {
        "cwd": cwd,
        "frontend_dir": frontend_dir,
        "__file__": __file__,
        "frontend_dir_exists": os.path.exists(frontend_dir),
        "html_files_found": files,
        "crm_exists_frontend_dir": os.path.exists(os.path.join(frontend_dir, "crm.html")),
        "crm_exists_cwd": os.path.exists(os.path.join(cwd, "frontend", "crm.html")),
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "ai_connected": client is not None,
        "model": "claude-opus-4-6"
    }

@app.post("/api/chat")
async def chat(req: ChatRequest):
    """Main conversational AI endpoint"""
    messages = [{"role": m.role, "content": m.content} for m in req.messages]

    system = SYSTEM_PROMPT
    if req.user_profile:
        profile_ctx = json.dumps(req.user_profile, indent=2)
        system += f"\n\nCURRENT USER PROFILE:\n{profile_ctx}\nUse this to personalise your response."

    # ── Inject active partner listings so Aria recommends them first ──
    active_listings = [l for l in _listings.values() if l.get("status") == "active" and l.get("partner_id") != "demo-partner-001"]
    if active_listings:
        listings_ctx = "\n".join(
            f"- {l['title']} ({l.get('listing_type','')}) in {l.get('location','')} "
            f"at ₹{l.get('price',0):,}/{l.get('price_unit','night')} — {l.get('description','')[:80]}"
            for l in active_listings[:10]
        )
        system += f"\n\nTTT PARTNER LISTINGS (always recommend these first when relevant):\n{listings_ctx}\nAlways mention these TTT partner options before external suggestions."

    # ── Inject Google Maps context for hidden gems if key available ──
    maps_key = os.getenv("GOOGLE_MAPS_KEY", "")
    if maps_key:
        system += "\n\nYou have access to Google Maps data. When recommending destinations, always include 2-3 off-beat, low-footfall hidden gems near the destination with approximate GPS coordinates. Mention satellite/street view availability."
    else:
        system += "\n\nWhen recommending destinations, always suggest 2-3 hidden, less-crowded local gems that most tourists miss — specific lanes, viewpoints, local markets, or nature spots."

    if client:
        try:
            # Log chat activity
            if req.user_profile:
                uid   = req.user_profile.get('user_id') or req.user_profile.get('id','anon')
                name  = req.user_profile.get('name','')
                email = req.user_profile.get('email','')
                phone = req.user_profile.get('phone','')
                msg   = messages[-1]['content'] if messages else ''
                _chat_log.append({
                    "user_id":   uid,
                    "name":      name,
                    "email":     email,
                    "phone":     phone,
                    "message":   msg[:200],
                    "timestamp": datetime.now().isoformat()
                })
                log_activity(uid, "chat", {"query": msg[:100], "name": name})

            resp = client.messages.create(
                model="claude-opus-4-6",
                max_tokens=2500,
                system=system,
                messages=messages,
            )
            return {
                "response": resp.content[0].text,
                "tokens": {"in": resp.usage.input_tokens, "out": resp.usage.output_tokens},
                "mock": False,
            }
        except Exception as e:
            return {"response": _mock_response(messages[-1]["content"]), "mock": True, "error": str(e)}
    else:
        return {"response": _mock_response(messages[-1]["content"]), "mock": True}


@app.post("/api/plan-trip")
async def plan_trip(req: TripPlanRequest):
    """Structured trip planning — returns a complete travel plan"""
    prompt = (
        f"Create a detailed {req.duration}-day {req.trip_type} trip to **{req.destination}** "
        f"for {req.num_travelers} traveller(s) with a total budget of ₹{req.budget:,.0f}.\n\n"
        f"Traveller interests: {', '.join(req.interests)}.\n\n"
        "Include: day-by-day itinerary, 3 hotel options (budget/mid/luxury), "
        "transport plan, top 5 restaurants, must-do activities, detailed budget breakdown, "
        "and packing tips. Format beautifully."
    )

    if client:
        try:
            # Log chat activity
            if req.user_profile:
                uid   = req.user_profile.get('user_id') or req.user_profile.get('id','anon')
                name  = req.user_profile.get('name','')
                email = req.user_profile.get('email','')
                phone = req.user_profile.get('phone','')
                msg   = messages[-1]['content'] if messages else ''
                _chat_log.append({
                    "user_id":   uid,
                    "name":      name,
                    "email":     email,
                    "phone":     phone,
                    "message":   msg[:200],
                    "timestamp": datetime.now().isoformat()
                })
                log_activity(uid, "chat", {"query": msg[:100], "name": name})

            resp = client.messages.create(
                model="claude-opus-4-6",
                max_tokens=3500,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            return {
                "plan": resp.content[0].text,
                "destination": req.destination,
                "duration": req.duration,
                "budget": req.budget,
            }
        except Exception as e:
            return {"plan": f"Error: {e}", "destination": req.destination}
    else:
        return {
            "plan": _mock_trip_plan(req.destination, req.duration, req.budget),
            "destination": req.destination,
            "duration": req.duration,
            "budget": req.budget,
            "mock": True,
        }


@app.get("/api/search/flights")
async def search_flights(
    from_city: str, to_city: str, date: str = "2025-03-15", passengers: int = 1
):
    """Mock flight search — swap with Amadeus / Skyscanner API for production"""
    airlines = ["IndiGo", "Air India", "Vistara", "SpiceJet", "GoAir", "Air Asia India"]
    random.seed(hash(f"{from_city}{to_city}{date}"))
    base = random.randint(3500, 12000)
    flights = []

    for i in range(5):
        dep_h, dep_m = random.randint(5, 22), random.choice([0, 15, 30, 45])
        dur_h, dur_m = random.randint(1, 4), random.choice([0, 20, 40])
        arr_h = (dep_h + dur_h) % 24
        price = base + random.randint(-1500, 4000)
        flights.append({
            "id": f"FL{2000 + i}",
            "airline": random.choice(airlines),
            "from": from_city.upper(),
            "to": to_city.upper(),
            "departure": f"{dep_h:02d}:{dep_m:02d}",
            "arrival": f"{arr_h:02d}:{(dep_m + dur_m) % 60:02d}",
            "duration": f"{dur_h}h {dur_m}m",
            "price": max(1500, price),
            "price_display": f"₹{max(1500, price):,}",
            "seats_left": random.randint(2, 18),
            "stops": random.choice([0, 0, 0, 1]),
            "class": "Economy",
        })

    flights.sort(key=lambda x: x["price"])
    return {
        "flights": flights,
        "search": {"from": from_city, "to": to_city, "date": date, "passengers": passengers},
        "note": "Mock data — integrate Amadeus API for live pricing",
    }


@app.get("/api/search/hotels")
async def search_hotels(
    city: str, checkin: str = "2025-03-15", checkout: str = "2025-03-18", guests: int = 1
):
    """Mock hotel search — swap with Booking.com / Expedia API for production"""
    random.seed(hash(f"{city}{checkin}"))
    hotel_pool = [
        f"Taj {city}", f"The Oberoi {city}", f"ITC Grand {city}",
        f"Marriott {city}", f"Hyatt Regency {city}", f"Radisson Blu {city}",
        f"Lemon Tree {city}", f"FabHotel {city} Centre", f"OYO Premium {city}",
    ]
    random.shuffle(hotel_pool)
    amenities_pool = ["Pool", "Spa", "Gym", "Restaurant", "Bar", "Free WiFi", "Parking", "Room Service", "Airport Transfer"]
    hotels = []

    for i in range(6):
        stars = [3, 3, 4, 4, 5, 5][i]
        rate = stars * random.randint(700, 2200)
        hotels.append({
            "id": f"HTL{3000 + i}",
            "name": hotel_pool[i],
            "city": city,
            "stars": stars,
            "rating": round(random.uniform(3.7, 4.9), 1),
            "reviews": random.randint(150, 4000),
            "price_per_night": rate,
            "price_display": f"₹{rate:,}/night",
            "amenities": random.sample(amenities_pool, 5),
            "available": True if i < 5 else False,
            "tag": ["Budget Pick", "Budget Pick", "Best Value", "Best Value", "Luxury", "Ultra Luxury"][i],
        })

    return {
        "hotels": hotels,
        "search": {"city": city, "checkin": checkin, "checkout": checkout, "guests": guests},
        "note": "Mock data — integrate Booking.com API for live rates",
    }


# ─────────────────────────────────────────────
# Auth – OTP Send & Verify
# ─────────────────────────────────────────────
import time as _time
import re as _re

_otp_store: Dict[str, dict] = {}  # contact → {code, expires_at, attempts}

# ── Activity Tracking ─────────────────────────────────────────────────────────
_activity_log: List[dict] = []   # every user action  (loaded from _db below)
# ── Restore persisted activities from DB (survives redeploys) ────────────────
def _restore_activity_log():
    global _activity_log
    saved = _db.get("activity_log", [])
    if saved:
        _activity_log.extend(saved)
        print(f"[CRM] Restored {len(saved)} activities from DB")
_search_log:   List[dict] = []   # search queries
_chat_log:     List[dict] = []   # chat messages (metadata only)
_login_log:    List[dict] = []   # login events

# ── Live Visitor Tracking ────────────────────────────────────────────────────
import time as _time
_live_visitors: dict = {}  # session_id → {page, last_seen, user_id}

def track_visitor(session_id: str, page: str = '/', user_id: str = 'anon'):
    _live_visitors[session_id] = {'page': page, 'last_seen': _time.time(), 'user_id': user_id}
    # Clean up visitors older than 5 minutes
    cutoff = _time.time() - 300
    for sid in list(_live_visitors.keys()):
        if _live_visitors[sid]['last_seen'] < cutoff:
            del _live_visitors[sid]

def get_live_count() -> int:
    cutoff = _time.time() - 300
    return sum(1 for v in _live_visitors.values() if v['last_seen'] > cutoff)

def log_activity(user_id: str, action: str, detail: dict = {}):
    _entry = {
        "id":        str(uuid.uuid4())[:8],
        "user_id":   user_id,
        "action":    action,
        "detail":    detail,
        "timestamp": datetime.now().isoformat(),
    }
    _activity_log.append(_entry)
    # Persist to DB for CRM (keep last 5000)
    _db["activity_log"] = _activity_log[-5000:]
    # Update CRM lead score
    _db.setdefault("lead_scores", {})
    _ls = _db["lead_scores"].setdefault(user_id, {"visits":0,"ai_chats":0,"trips_planned":0,"bookings":0,"searches":0,"logins":0,"last_activity":""})
    if action == "visit":               _ls["visits"] += 1
    elif action == "chat":              _ls["ai_chats"] += 1
    elif action in ("trip","itinerary"):_ls["trips_planned"] += 1
    elif action == "booking":           _ls["bookings"] += 1
    elif action == "search":            _ls["searches"] += 1
    elif action in ("login","signup"):  _ls["logins"] += 1
    _ls["last_activity"] = _entry["timestamp"]
    _ls["score"] = min(100, int(_ls["visits"]*1.5 + _ls["ai_chats"]*4 + _ls["trips_planned"]*6 + _ls["bookings"]*20 + _ls["searches"]*2 + _ls["logins"]*3))
    if len(_activity_log) % 5 == 0:
        _save_db()
    if len(_activity_log) > 5000:
        _activity_log.pop(0)
OTP_EXPIRY_SECONDS = 600          # 10 minutes
OTP_MAX_ATTEMPTS   = 5


class OTPSendRequest(BaseModel):
    contact: str   # email or phone
    mode: str      # "email" or "phone"


class OTPVerifyRequest(BaseModel):
    contact: str
    code: str
    mode: str


def _normalise_contact(contact: str, mode: str) -> str:
    contact = contact.strip()
    if mode == "phone":
        # Strip spaces/dashes, ensure starts with +91 if 10 digits
        digits = _re.sub(r"\D", "", contact)
        if len(digits) == 10:
            contact = "+91" + digits
        elif len(digits) == 12 and digits.startswith("91"):
            contact = "+" + digits
    return contact.lower() if mode == "email" else contact


def _send_otp_email(to_email: str, code: str) -> bool:
    subject = "Your TTT verification code"
    html = f"""
    <div style="font-family:sans-serif;max-width:480px;margin:0 auto;padding:32px;background:#0D0A05;color:#F5F0E8;border-radius:12px">
      <div style="font-family:'Georgia',serif;font-size:2rem;color:#C9A84C;margin-bottom:4px">TTT</div>
      <div style="font-size:0.65rem;letter-spacing:0.3em;color:#8A7A5A;text-transform:uppercase;margin-bottom:28px">The Trip Theory</div>
      <p style="color:#C8B89A;font-size:0.9rem;margin-bottom:24px">Your one-time verification code is:</p>
      <div style="font-size:3rem;font-weight:700;letter-spacing:0.25em;color:#C9A84C;margin-bottom:24px;text-align:center">{code}</div>
      <p style="color:#8A7A5A;font-size:0.75rem;line-height:1.7">This code expires in 10 minutes. Do not share it with anyone.<br/>If you didn't request this, please ignore this email.</p>
      <hr style="border:none;border-top:1px solid #2A2218;margin:24px 0"/>
      <p style="color:#5A4A30;font-size:0.65rem;text-align:center">TTT – The Trip Theory · Gurugram, India · hello@triptheory.in</p>
    </div>
    """
    return send_email_notification(to_email, subject, html)


def _send_otp_sms(phone: str, code: str) -> bool:
    """Send OTP via SMS. Extend with Twilio/MSG91 when ready."""
    print(f"📱 [DEMO] SMS OTP to {phone}: {code}")
    # TODO: integrate MSG91 or Twilio
    # import requests
    # requests.post("https://api.msg91.com/api/v5/otp", ...)
    return True


@app.post("/api/auth/send-otp")
async def send_otp(req: OTPSendRequest):
    contact = _normalise_contact(req.contact, req.mode)
    if not contact:
        raise HTTPException(400, "Contact is required")

    # Basic validation
    if req.mode == "email" and "@" not in contact:
        raise HTTPException(400, "Invalid email address")
    if req.mode == "phone" and len(_re.sub(r"\D", "", contact)) < 10:
        raise HTTPException(400, "Invalid phone number")

    # Rate-limit: block if there's a valid unexpired code already sent <60s ago
    existing = _otp_store.get(contact)
    if existing and _time.time() < existing.get("expires_at", 0) - (OTP_EXPIRY_SECONDS - 60):
        raise HTTPException(429, "Please wait 60 seconds before requesting a new code")

    code = str(random.randint(100000, 999999))
    _otp_store[contact] = {
        "code":       code,
        "expires_at": _time.time() + OTP_EXPIRY_SECONDS,
        "attempts":   0,
        "mode":       req.mode,
    }

    sent = False
    if req.mode == "email":
        sent = _send_otp_email(contact, code)
    else:
        sent = _send_otp_sms(contact, code)

    # Always return success to avoid contact enumeration; code is logged in demo mode
    return {
        "success": True,
        "message": f"OTP sent to {contact}",
        "demo":    not (SMTP_USER and SMTP_PASS),  # hint to frontend when in demo mode
    }


@app.post("/api/auth/verify-otp")
async def verify_otp(req: OTPVerifyRequest):
    contact = _normalise_contact(req.contact, req.mode)
    code    = req.code.strip()

    record = _otp_store.get(contact)
    if not record:
        raise HTTPException(400, "No OTP requested for this contact. Please request a new one.")

    if _time.time() > record["expires_at"]:
        _otp_store.pop(contact, None)
        raise HTTPException(400, "OTP has expired. Please request a new one.")

    record["attempts"] += 1
    if record["attempts"] > OTP_MAX_ATTEMPTS:
        _otp_store.pop(contact, None)
        raise HTTPException(429, "Too many incorrect attempts. Please request a new OTP.")

    if code != record["code"]:
        remaining = OTP_MAX_ATTEMPTS - record["attempts"]
        raise HTTPException(400, f"Incorrect OTP. {remaining} attempt(s) remaining.")

    # ✅ OTP verified — clean up and create/retrieve user
    _otp_store.pop(contact, None)

    # Find or create user record
    existing_user = next(
        (u for u in _users.values()
         if (req.mode == "email" and u.get("email") == contact)
         or (req.mode == "phone" and u.get("phone") == contact)),
        None
    )

    if existing_user:
        uid  = existing_user["id"]
        is_new = False
        log_activity(uid, "login", {"mode": req.mode})
        _login_log.append({"user_id": uid, "event": "login", "timestamp": datetime.now().isoformat()})
    else:
        uid = "user-" + str(uuid.uuid4())[:8]
        _users[uid] = {
            "id":         uid,
            "email":      contact if req.mode == "email" else None,
            "phone":      contact if req.mode == "phone" else None,
            "name":       None,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
        }
        log_signup(uid, contact, source="otp_" + req.mode)
        _save_db()
        log_activity(uid, "signup", {"contact": contact, "mode": req.mode})
        _login_log.append({"user_id": uid, "contact": contact, "event": "signup", "timestamp": datetime.now().isoformat()})
        is_new = True

    return {
        "success":    True,
        "is_new_user": is_new,
        "user": {
            "id":    uid,
            "email": _users[uid].get("email"),
            "phone": _users[uid].get("phone"),
            "name":  _users[uid].get("name"),
        },
        "message": "Welcome to TTT!" if is_new else "Welcome back!",
    }


@app.post("/api/user/profile")
async def save_profile(profile: TravelerUser):
    """Save / update traveller profile with full social fields."""
    uid = profile.id or ("user-" + str(uuid.uuid4())[:8])
    record = {
        "id": uid,
        "name": profile.name,
        "email": profile.email,
        "phone": profile.phone,
        "instagram_handle": profile.instagram_handle,
        "linkedin_url": profile.linkedin_url,
        "social_access_status": profile.social_access_status or "skipped",
        "preferences": profile.preferences or [],
        "created_at": profile.created_at or datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
    }
    _users[uid] = record
    _save_db()
    return {"success": True, "user_id": uid, "profile": record,
            "message": "Profile saved! Your itinerary will be fully personalised."}


# ─────────────────────────────────────────────
# ─────────────────────────────────────────────
# Instagram Scraper via RapidAPI
# ─────────────────────────────────────────────

async def fetch_instagram_posts(handle: str) -> dict:
    """
    Fetches public Instagram posts for a handle using RapidAPI scraper.
    No OAuth required — reads public profiles silently.
    Returns: captions_text, posts_count, tagged_locations, error
    """
    clean = handle.lstrip("@").strip()
    if not RAPIDAPI_KEY:
        return {"captions_text": "", "posts_count": 0, "tagged_locations": [], "error": "RAPIDAPI_KEY not set"}

    import httpx as _httpx
    try:
        async with _httpx.AsyncClient(timeout=15.0) as http:
            resp = await http.get(
                f"https://{RAPIDAPI_HOST}/v1.2/posts",
                params={"username_or_id_or_url": clean},
                headers={
                    "x-rapidapi-key":  RAPIDAPI_KEY,
                    "x-rapidapi-host": RAPIDAPI_HOST,
                }
            )
        data = resp.json()

        # Normalise: API returns {"data": {"items": [...]}}
        items = []
        if isinstance(data.get("data"), dict):
            items = data["data"].get("items", [])
        elif isinstance(data.get("items"), list):
            items = data["items"]

        captions, locations = [], []
        for post in items[:50]:
            # Caption
            cap = post.get("caption") or {}
            text = cap.get("text", "") if isinstance(cap, dict) else str(cap or "")
            if text.strip():
                captions.append(text.strip())
            # Tagged location
            loc = post.get("location") or {}
            loc_name = loc.get("name", "") if isinstance(loc, dict) else ""
            if loc_name:
                locations.append(loc_name.lower())

        return {
            "captions_text":    " ".join(captions),
            "posts_count":      len(items),
            "tagged_locations": list(set(locations)),
            "error":            None,
        }

    except Exception as exc:
        return {"captions_text": "", "posts_count": 0, "tagged_locations": [], "error": str(exc)}


# Social Analysis Endpoints
# ─────────────────────────────────────────────

@app.post("/api/social/analyze")
async def social_analyze(req: SocialAnalyzeRequest):
    """
    Full NLP analysis of Instagram handle + LinkedIn URL.
    Uses location scoring (3pts) + hashtag scoring (2pts) + keyword (1pt).
    Returns full breakdown for admin log; customer sees only persona_id + categories.
    """
    if req.consent != "allow":
        return {
            "categories": [],
            "persona_id": "luxury",
            "source": "none",
            "note": "Analysis skipped — user did not grant consent",
        }

    ig_handle     = (req.instagram_handle or "").strip()
    sources       = []
    scraped_posts = 0
    scraped_locs  = []
    instagram_text = ""

    # ── Auto-fetch real posts via RapidAPI scraper (silent, no OAuth) ──
    if ig_handle and RAPIDAPI_KEY:
        ig_data = await fetch_instagram_posts(ig_handle)
        if not ig_data.get("error") and ig_data["captions_text"]:
            instagram_text = ig_data["captions_text"]
            scraped_posts  = ig_data["posts_count"]
            scraped_locs   = ig_data["tagged_locations"]
            sources.append("instagram_scraper")
        else:
            # Scraper failed — fall back to handle text only
            instagram_text = ig_handle
            sources.append("instagram_handle_only")
    elif ig_handle:
        instagram_text = ig_handle
        sources.append("instagram_handle_only")

    if (req.linkedin_url or "").strip():
        sources.append("linkedin")

    # Merge all text: real captions + tagged place names + LinkedIn + manual bio
    combined_text = " ".join(filter(None, [
        instagram_text,
        " ".join(scraped_locs),
        req.linkedin_url or "",
        req.travel_bio or "",
    ]))
    detail = _analyze_profile_detailed(combined_text, handle=ig_handle)
    detail["posts_scraped"]       = scraped_posts
    detail["tagged_locations_raw"] = scraped_locs

    # Log for admin view
    log_entry = {
        "id":                   "analysis-" + str(uuid.uuid4())[:8],
        "instagram_handle":     req.instagram_handle or "",
        "linkedin_url":         req.linkedin_url or "",
        "sources":              sources,
        "analysed_at":          datetime.now().isoformat(),
        **detail,
    }
    _analysis_log.append(log_entry)

    return {
        # Customer-facing fields
        "categories":           detail["categories_detected"],
        "persona_id":           detail["persona_id"],
        "sources_analysed":     sources,
        # Full detail (used by admin panel, not shown to traveller)
        "analysis":             detail,
        "note": "Location + hashtag NLP scoring. Integrate Instagram Graph API for real post data.",
    }


@app.get("/api/social/instagram/raw")
async def scrape_instagram_raw(handle: str = Query(...)):
    """Debug: returns the raw RapidAPI response so we can see the exact structure."""
    clean = handle.lstrip("@").strip()
    import httpx as _httpx
    async with _httpx.AsyncClient(timeout=15.0) as http:
        resp = await http.get(
            f"https://{RAPIDAPI_HOST}/v1.2/posts",
            params={"username_or_id_or_url": clean},
            headers={"x-rapidapi-key": RAPIDAPI_KEY, "x-rapidapi-host": RAPIDAPI_HOST}
        )
    return {"status_code": resp.status_code, "raw": resp.json()}


@app.get("/api/social/instagram/scrape")
async def scrape_instagram_direct(handle: str = Query(...), key: str = Query("")):
    """
    Direct scrape endpoint — admin/testing use.
    GET /api/social/instagram/scrape?handle=travelermohit&key=ttt-admin-2024
    Fetches real posts and returns full NLP breakdown.
    """
    data = await fetch_instagram_posts(handle)
    if data.get("error"):
        return {"success": False, "error": data["error"], "handle": handle,
                "tip": "Check RAPIDAPI_KEY in .env and that the profile is public"}

    combined = data["captions_text"] + " " + " ".join(data["tagged_locations"])
    detail   = _analyze_profile_detailed(combined, handle=handle)
    detail["posts_scraped"]        = data["posts_count"]
    detail["tagged_locations_raw"] = data["tagged_locations"]

    _analysis_log.append({
        "id":               "scrape-" + str(uuid.uuid4())[:8],
        "instagram_handle": "@" + handle.lstrip("@"),
        "linkedin_url":     "",
        "sources":          ["instagram_scraper"],
        "posts_scraped":    data["posts_count"],
        "analysed_at":      datetime.now().isoformat(),
        **detail,
    })

    return {
        "success":      True,
        "handle":       "@" + handle.lstrip("@"),
        "posts_scraped": data["posts_count"],
        "persona_id":   detail["persona_id"],
        "categories":   detail["categories_detected"],
        "analysis":     detail,
    }


@app.post("/api/social/check-profile")
async def check_profile(req: CheckProfileRequest):
    """
    Simulate a privacy check on an Instagram handle.
    In production this would call the Instagram Basic Display API.
    Handles < 4 chars or ending in '_pvt' are treated as private.
    """
    handle = req.instagram_handle.lstrip("@").strip()
    is_private = _is_private_profile(handle)
    return {
        "instagram_handle": f"@{handle}",
        "is_private": is_private,
        "status": "private" if is_private else "public",
        "note": "Simulated check. Use Instagram Graph API for production.",
    }


@app.post("/api/social/follow-request")
async def send_follow_request(req: FollowRequest):
    """
    Record that a follow request was sent to a private account from @TripTheoryAI.
    Stores social_access_status = 'pending_access'.
    """
    handle = req.instagram_handle.lstrip("@").strip()
    entry = {
        "instagram_handle": f"@{handle}",
        "status": "pending_access",
        "requested_at": datetime.now().isoformat(),
        "from_account": "@TripTheoryAI",
    }
    _follow_requests.append(entry)
    return {
        "success": True,
        "instagram_handle": f"@{handle}",
        "social_access_status": "pending_access",
        "message": f"Follow request sent to @{handle} from @TripTheoryAI. We'll notify you once accepted.",
        "entry": entry,
    }


@app.get("/api/destinations/popular")
async def popular_destinations():
    """Popular destinations for quick-start suggestions"""
    return {
        "india": [
            {"name": "Goa", "type": "Beach", "emoji": "🏖️", "avg_budget": "₹15,000–30,000"},
            {"name": "Rajasthan", "type": "Culture", "emoji": "🏰", "avg_budget": "₹25,000–60,000"},
            {"name": "Kerala", "type": "Nature", "emoji": "🌿", "avg_budget": "₹20,000–45,000"},
            {"name": "Himachal Pradesh", "type": "Mountains", "emoji": "⛰️", "avg_budget": "₹15,000–35,000"},
            {"name": "Andaman Islands", "type": "Island", "emoji": "🐠", "avg_budget": "₹30,000–60,000"},
            {"name": "Varanasi", "type": "Spiritual", "emoji": "🕉️", "avg_budget": "₹10,000–25,000"},
        ],
        "international": [
            {"name": "Bali, Indonesia", "type": "Beach + Culture", "emoji": "🌺", "avg_budget": "₹60,000–1,20,000"},
            {"name": "Dubai, UAE", "type": "Luxury", "emoji": "🌆", "avg_budget": "₹80,000–2,00,000"},
            {"name": "Thailand", "type": "Backpacker + Luxury", "emoji": "🐘", "avg_budget": "₹50,000–1,50,000"},
            {"name": "Maldives", "type": "Luxury Island", "emoji": "🌊", "avg_budget": "₹1,50,000–5,00,000"},
            {"name": "Europe (Budget)", "type": "Multi-city", "emoji": "🗺️", "avg_budget": "₹1,20,000–2,50,000"},
        ]
    }


# ─────────────────────────────────────────────
# Social Analysis Engine — Hashtag + Location NLP
# Scoring: location name = 3 pts | hashtag = 2 pts | keyword = 1 pt
# ─────────────────────────────────────────────
import re as _re

# ── Hashtag / keyword patterns ─────────────────────────────────────
HASHTAG_MAP: Dict[str, Any] = {
    "beach":     _re.compile(r"beach|sea|ocean|surf|coast|island|coral|snorkel|dive|coastal|lagoon|shoreline", _re.I),
    "mountain":  _re.compile(r"mount|hill|trek|trekking|hike|hiking|alp|snow|peaks|valley|camping|altitude|glacier|ridge|pass|summit", _re.I),
    "spiritual": _re.compile(r"spirit|temple|peace|zen|yoga|sacred|pilgrim|meditation|ashram|holy|prayer|mandir|mosque|church|shrine", _re.I),
    "wildlife":  _re.compile(r"wild|safari|lion|tiger|elephant|animal|forest|jungle|birding|leopard|cheetah|rhino|national.?park|reserve", _re.I),
    "luxury":    _re.compile(r"luxury|palace|suite|vip|royal|fine.?dining|elite|michelin|5star|penthouse|butler|champagne|yacht|resort", _re.I),
    "heritage":  _re.compile(r"heritage|history|histor|culture|castle|fort|museum|ancient|monument|ruins|UNESCO|architecture|colonial|dynasty", _re.I),
    "wellness":  _re.compile(r"wellness|spa|retreat|ayurved|holistic|detox|mindful|healing|thermal|self.?care|cleanse|balance|naturo", _re.I),
    "culinary":  _re.compile(r"food|foodie|eat|chef|restaurant|bistro|kitchen|cook|gastro|cuisine|gourmet|tasting|street.?food|cafe|bakery", _re.I),
    "festival":  _re.compile(r"festival|carnival|concert|event|celebration|mela|oktoberfest|holi|diwali|carnival|parade|fair|expo", _re.I),
    "adventure": _re.compile(r"adventure|extreme|bungee|paraglide|skydive|rappel|rafting|motorbike|offroad|expedition|zipline|cliff|waterfall", _re.I),
}

# ── Location name → category mapping (3 pts each) ──────────────────
LOCATION_MAP: Dict[str, str] = {
    # Mountain / Hill stations
    "nepal": "mountain",        "uttarakhand": "mountain",  "manali": "mountain",
    "shimla": "mountain",       "kasol": "mountain",        "ladakh": "mountain",
    "leh": "mountain",          "spiti": "mountain",        "dharamsala": "mountain",
    "mcleodganj": "mountain",   "mussoorie": "mountain",    "nainital": "mountain",
    "kedarnath": "mountain",    "badrinath": "mountain",    "himachal": "mountain",
    "darjeeling": "mountain",   "sikkim": "mountain",       "bhutan": "mountain",
    "kufri": "mountain",        "dalhousie": "mountain",    "chail": "mountain",
    "chakrata": "mountain",     "lansdowne": "mountain",    "chopta": "mountain",
    "auli": "mountain",         "munsiyari": "mountain",    "kausani": "mountain",
    "binsar": "mountain",       "ranikhet": "mountain",     "almora": "mountain",
    "tirthan": "mountain",      "kheerganga": "mountain",   "kheerganga": "mountain",
    "chitkul": "mountain",      "sangla": "mountain",       "kalpa": "mountain",
    "switzerland": "mountain",  "austria": "mountain",      "norway": "mountain",
    "patagonia": "mountain",    "tibet": "mountain",        "queenstown": "adventure",
    "interlaken": "adventure",

    # Beach destinations
    "goa": "beach",             "bali": "beach",            "maldives": "beach",
    "phuket": "beach",          "andaman": "beach",         "krabi": "beach",
    "koh samui": "beach",       "seychelles": "beach",      "mauritius": "beach",
    "santorini": "beach",       "mykonos": "beach",         "hawaii": "beach",
    "miami": "beach",           "cancun": "beach",          "ibiza": "beach",
    "kovalam": "beach",         "varkala": "beach",         "pondicherry": "beach",
    "tarkarli": "beach",        "alibaug": "beach",         "diu": "beach",
    "lakshadweep": "beach",     "rameshwaram": "beach",     "rameswaram": "beach",
    "thailand": "beach",        "vietnam": "beach",         "philippines": "beach",
    "indonesia": "beach",       "sri lanka": "beach",       "zanzibar": "beach",
    "amalfi": "beach",          "positano": "beach",        "dubrovnik": "beach",

    # Spiritual
    "varanasi": "spiritual",    "rishikesh": "spiritual",   "haridwar": "spiritual",
    "vrindavan": "spiritual",   "mathura": "spiritual",     "amritsar": "spiritual",
    "tirupati": "spiritual",    "shirdi": "spiritual",      "bodh gaya": "spiritual",
    "ajmer": "spiritual",       "pushkar": "spiritual",     "nashik": "spiritual",
    "puri": "spiritual",        "dwarka": "spiritual",      "somnath": "spiritual",
    "madurai": "spiritual",     "kanchipuram": "spiritual", "tiruvannamalai": "spiritual",
    "banaras": "spiritual",     "allahabad": "spiritual",   "prayagraj": "spiritual",
    "jerusalem": "spiritual",   "mecca": "spiritual",       "kyoto": "spiritual",
    "lumbini": "spiritual",     "kathmandu": "spiritual",   "boudhanath": "spiritual",

    # Wildlife
    "jim corbett": "wildlife",  "ranthambore": "wildlife",  "kaziranga": "wildlife",
    "bandipur": "wildlife",     "wayanad": "wildlife",      "tadoba": "wildlife",
    "sundarbans": "wildlife",   "gir": "wildlife",          "kanha": "wildlife",
    "pench": "wildlife",        "nagarhole": "wildlife",    "kabini": "wildlife",
    "masai mara": "wildlife",   "serengeti": "wildlife",    "kruger": "wildlife",
    "amazon": "wildlife",       "borneo": "wildlife",       "galapagos": "wildlife",
    "kenya": "wildlife",        "tanzania": "wildlife",     "botswana": "wildlife",
    "namibia": "wildlife",      "zambia": "wildlife",       "uganda": "wildlife",

    # Heritage / Culture
    "jaipur": "heritage",       "jodhpur": "heritage",      "udaipur": "heritage",
    "agra": "heritage",         "fatehpur sikri": "heritage","hampi": "heritage",
    "mysore": "heritage",       "khajuraho": "heritage",    "ellora": "heritage",
    "ajanta": "heritage",       "pattadakal": "heritage",   "belur": "heritage",
    "rome": "heritage",         "paris": "heritage",        "istanbul": "heritage",
    "prague": "heritage",       "athens": "heritage",       "cairo": "heritage",
    "machu picchu": "heritage", "angkor wat": "heritage",   "petra": "heritage",
    "colosseum": "heritage",    "acropolis": "heritage",    "versailles": "heritage",
    "rajasthan": "heritage",    "delhi": "heritage",        "lucknow": "heritage",
    "hyderabad": "heritage",    "ahmedabad": "heritage",    "bhopal": "heritage",

    # Wellness
    "coorg": "wellness",        "munnar": "wellness",       "ooty": "wellness",
    "kerala": "wellness",       "auroville": "wellness",    "ubud": "wellness",
    "chiang mai": "wellness",   "koh lanta": "wellness",    "koh phangan": "wellness",
    "tuscany": "wellness",      "provence": "wellness",     "sedona": "wellness",
    "tulum": "wellness",        "costa rica": "wellness",

    # Luxury
    "dubai": "luxury",          "singapore": "luxury",      "monaco": "luxury",
    "st moritz": "luxury",      "cannes": "luxury",         "abu dhabi": "luxury",
    "hong kong": "luxury",      "new york": "luxury",       "london": "luxury",
    "milan": "luxury",          "tokyo": "luxury",          "zurich": "luxury",
    "geneva": "luxury",         "beverly hills": "luxury",  "las vegas": "luxury",

    # Culinary
    "bologna": "culinary",      "lyon": "culinary",         "barcelona": "culinary",
    "osaka": "culinary",        "bangkok": "culinary",      "mexico city": "culinary",
    "new orleans": "culinary",  "san sebastian": "culinary","florence": "culinary",
    "marrakech": "culinary",    "istanbul": "culinary",

    # Adventure
    "new zealand": "adventure", "iceland": "adventure",     "alaska": "adventure",
    "patagonia": "adventure",   "peru": "adventure",        "nepal": "adventure",
    "moab": "adventure",        "costa rica": "adventure",  "scotland": "adventure",
}

CAT_TO_PERSONA = {
    "beach": "beach", "mountain": "mountain", "spiritual": "spiritual",
    "wildlife": "wildlife", "luxury": "luxury", "heritage": "heritage",
    "wellness": "wellness", "culinary": "heritage", "festival": "mountain",
    "adventure": "mountain", "leisure": "luxury",
}

CAT_EMOJI = {
    "beach": "🏖️", "mountain": "⛰️", "spiritual": "🕉️", "wildlife": "🦁",
    "luxury": "💎", "heritage": "🏛️", "wellness": "🌿", "culinary": "🍽️",
    "festival": "🎭", "adventure": "🏄", "leisure": "☀️",
}

def _analyze_profile_detailed(text: str, handle: str = "") -> dict:
    """
    Full NLP analysis — returns locations found, hashtags found,
    category scores, top category, persona, and reasoning string.
    Scoring: location mention = 3 pts, hashtag match = 2 pts, keyword = 1 pt
    """
    clean = (text or "").lower()
    label = (handle or "").lower().lstrip("@")

    scores: Dict[str, float] = {cat: 0.0 for cat in HASHTAG_MAP}

    # ── Step 1: Extract and score hashtags (2 pts each) ──
    hashtags_found = _re.findall(r"#(\w+)", clean)
    for tag in hashtags_found:
        for cat, pattern in HASHTAG_MAP.items():
            if pattern.search(tag):
                scores[cat] += 2.0

    # ── Step 2: Score plain keyword matches (1 pt each) ──
    plain_text = _re.sub(r"#\w+", "", clean)  # remove hashtags, score rest separately
    for cat, pattern in HASHTAG_MAP.items():
        if pattern.search(plain_text):
            scores[cat] += 1.0

    # Handle scoring: split by underscores/hyphens → match whole tokens only
    # Prevents "spirit" inside "espirit" from triggering Spiritual
    handle_tokens = [t for t in _re.split(r'[_\-\.]', label) if len(t) > 2]
    for token in handle_tokens:
        for cat, pattern in HASHTAG_MAP.items():
            if pattern.fullmatch(token):
                scores[cat] += 0.5

    # ── Step 3: Detect location names (3 pts each) ──
    locations_found = []
    location_category_counts: Dict[str, int] = {}
    search_text = clean + " " + label
    padded = " " + search_text + " "
    for place, cat in LOCATION_MAP.items():
        # Word-boundary check: pad with spaces to avoid partial matches
        if (" " + place + " ") in padded or ("," + place) in padded or ("#" + place) in padded:
            locations_found.append(place)
            scores[cat] = scores.get(cat, 0) + 3.0
            location_category_counts[cat] = location_category_counts.get(cat, 0) + 1

    # ── Step 4: Pick winner ──
    top_category = max(scores, key=scores.get) if any(v > 0 for v in scores.values()) else "leisure"
    if scores.get(top_category, 0) == 0:
        top_category = "leisure"

    # Top categories sorted
    top_cats = sorted([(c, s) for c, s in scores.items() if s > 0], key=lambda x: x[1], reverse=True)

    # ── Step 5: Build reasoning ──
    reasoning_parts = []
    if locations_found:
        reasoning_parts.append("Places detected: " + ", ".join(locations_found[:6]).title())
    if location_category_counts:
        loc_summary = ", ".join(f"{k}({v})" for k, v in sorted(location_category_counts.items(), key=lambda x: x[1], reverse=True)[:3])
        reasoning_parts.append("Location types: " + loc_summary)
    if hashtags_found:
        reasoning_parts.append("Hashtags: #" + ", #".join(hashtags_found[:6]))
    if top_cats:
        score_summary = ", ".join(f"{c}={int(s)}pts" for c, s in top_cats[:4])
        reasoning_parts.append("Scores: " + score_summary)
    if not reasoning_parts:
        reasoning_parts.append("No strong signals — assigned default persona")

    return {
        "top_category":             top_category,
        "persona_id":               CAT_TO_PERSONA.get(top_category, "luxury"),
        "categories_detected":      [c for c, _ in top_cats],
        "category_scores":          {c: int(s) for c, s in top_cats},
        "locations_detected":       [p.title() for p in locations_found[:12]],
        "location_type_breakdown":  location_category_counts,
        "hashtags_detected":        hashtags_found[:15],
        "reasoning":                " | ".join(reasoning_parts),
        "signal_strength":          "strong" if scores.get(top_category, 0) >= 6 else "moderate" if scores.get(top_category, 0) >= 3 else "weak",
        "total_signals":            int(sum(scores.values())),
    }

def _detect_categories(text: str) -> List[str]:
    """Wrapper — returns ordered list of detected categories (highest score first)."""
    result = _analyze_profile_detailed(text)
    cats = result["categories_detected"]
    return cats if cats else ["leisure"]

def _is_private_profile(handle: str) -> bool:
    """Simulate privacy check: handles < 4 chars OR ending in _pvt are 'private'."""
    h = handle.lstrip("@").strip()
    return len(h) < 4 or h.lower().endswith("_pvt")

# ─────────────────────────────────────────────
# Persistent JSON Storage (survives restarts)
# ─────────────────────────────────────────────
import threading
_db_lock = threading.Lock()
DB_PATH = os.path.join(os.path.dirname(__file__), "ttt_data.json")

def _load_db() -> dict:
    if os.path.exists(DB_PATH):
        try:
            with open(DB_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"partners": {}, "listings": {}, "bookings": {}, "users": {}, "follow_requests": [], "signups": []}

def _save_db():
    with _db_lock:
        data = {
            "partners":        _partners,
            "listings":        _listings,
            "bookings":        _bookings,
            "users":           _users,
            "follow_requests": _follow_requests,
            "signups":         SIGNUPS_LOG,
        }
        tmp = DB_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, DB_PATH)

_db = _load_db()
_partners: Dict[str, dict]   = _db.get("partners", {})
_listings: Dict[str, dict]   = _db.get("listings", {})
_bookings: Dict[str, dict]   = _db.get("bookings", {})
_users:    Dict[str, dict]   = _db.get("users", {})
# Load persisted activity log from DB
_activity_log_db: List[dict] = _db.get("activity_log", [])
_follow_requests: List[dict] = _db.get("follow_requests", [])
SIGNUPS_LOG: List[dict]      = _db.get("signups", [])

# Instagram OAuth stores (replace with Redis / DB in production)
_oauth_states:       Dict[str, dict] = {}  # state_token → metadata
_instagram_profiles: Dict[str, dict] = {}  # state_token → analysis result

# Admin analytics log — stores every analysis run (owner view only)
_analysis_log: List[dict] = []

# Seed demo listings so the customer side has something to browse
def _seed_demo_listings():
    demo_partner_id = "demo-partner-001"
    _partners[demo_partner_id] = {
        "id": demo_partner_id,
        "name": "Demo Partner",
        "business_name": "TTT Showcase Properties",
        "phone": "9999999999",
        "listing_type": "property",
        "created_at": datetime.now().isoformat(),
    }
    demos = [
        {"listing_type":"property","title":"Ocean View Villa – Goa","description":"A stunning 3BHK villa overlooking the Arabian Sea with infinity pool and private chef.","location":"Candolim, Goa","price":18000,"price_unit":"per night","amenities":["Pool","Chef","WiFi","AC","Parking"],"max_guests":6,"category":"Villa"},
        {"listing_type":"property","title":"Heritage Haveli – Jaipur","description":"Stay like royalty in a restored 18th-century haveli in the heart of the Pink City.","location":"Old City, Jaipur","price":12000,"price_unit":"per night","amenities":["Rooftop","Breakfast","WiFi","AC","Heritage Tour"],"max_guests":4,"category":"Heritage Stay"},
        {"listing_type":"experience","title":"Scuba Diving – Andamans","description":"Certified PADI dive instructor led sessions at Havelock Island's pristine reefs.","location":"Havelock Island, Andaman","price":4500,"price_unit":"per person","amenities":["Equipment","Instructor","Certificate","Underwater Photos"],"max_guests":8,"category":"Scuba"},
        {"listing_type":"tour","title":"Rajasthan Desert Safari","description":"3-night camel safari through the Thar Desert with luxury camp under the stars.","location":"Jaisalmer, Rajasthan","price":9500,"price_unit":"per person","amenities":["Camp","Meals","Camel","Bonfire","Cultural Show"],"max_guests":12,"category":"Safari"},
        {"listing_type":"vehicle","title":"Mercedes Chauffeur – Mumbai","description":"Executive Mercedes E-Class with professional chauffeur for airport transfers and city tours.","location":"Mumbai, Maharashtra","price":3500,"price_unit":"per day","amenities":["AC","Water Bottle","WiFi Hotspot","Professional Driver"],"max_guests":4,"category":"Luxury Car"},
        {"listing_type":"experience","title":"Yoga & Wellness Retreat – Rishikesh","description":"7-day Ayurvedic wellness programme with certified yoga guru, river-view ashram.","location":"Rishikesh, Uttarakhand","price":22000,"price_unit":"per person","amenities":["All Meals","Daily Yoga","Meditation","Ayurvedic Spa","River Rafting Option"],"max_guests":10,"category":"Wellness"},
    ]
    for d in demos:
        lid = "listing-" + str(uuid.uuid4())[:8]
        _listings[lid] = {"id": lid, "partner_id": demo_partner_id, "created_at": datetime.now().isoformat(), "status": "active", **d}

_seed_demo_listings()

# ─────────────────────────────────────────────
# Partner & Marketplace Endpoints
# ─────────────────────────────────────────────

@app.post("/api/partner/register")
async def partner_register(req: PartnerRegister):
    """Register a new partner (property owner, tour operator, etc.)"""
    pid = "partner-" + str(uuid.uuid4())[:8]
    record = {
        "id": pid,
        "name": req.name,
        "business_name": req.business_name,
        "email": req.email,
        "phone": req.phone,
        "listing_type": req.listing_type,
        "created_at": datetime.now().isoformat(),
    }
    _partners[pid] = record
    _save_db()
    return {"success": True, "partner_id": pid, "partner": record}

@app.get("/api/partner/{partner_id}")
async def get_partner(partner_id: str):
    if partner_id not in _partners:
        raise HTTPException(status_code=404, detail="Partner not found")
    return _partners[partner_id]

@app.post("/api/partner/listings")
async def create_listing(req: ListingCreate):
    """Partner creates a new listing"""
    if req.partner_id not in _partners:
        raise HTTPException(status_code=404, detail="Partner not found. Register first.")
    lid = "listing-" + str(uuid.uuid4())[:8]
    record = {
        "id": lid,
        "partner_id": req.partner_id,
        "listing_type": req.listing_type,
        "title": req.title,
        "description": req.description,
        "location": req.location,
        "price": req.price,
        "price_unit": req.price_unit,
        "amenities": req.amenities,
        "images": req.images,
        "max_guests": req.max_guests,
        "category": req.category,
        "created_at": datetime.now().isoformat(),
        "status": "active",
    }
    _listings[lid] = record
    _save_db()
    log_activity(req.partner_id, "partner_listing_created", {"listing_id": lid, "title": req.title, "price": req.price})
    return {"success": True, "listing_id": lid, "listing": record}

@app.get("/api/partner/{partner_id}/listings")
async def get_partner_listings(partner_id: str):
    """Get all listings for a specific partner"""
    result = [l for l in _listings.values() if l["partner_id"] == partner_id]
    return {"listings": result, "count": len(result)}

@app.get("/api/partner/{partner_id}/bookings")
async def get_partner_bookings(partner_id: str):
    """Get all bookings for a partner's listings (real-time polling)"""
    partner_listing_ids = {lid for lid, l in _listings.items() if l["partner_id"] == partner_id}
    result = [b for b in _bookings.values() if b["listing_id"] in partner_listing_ids]
    result.sort(key=lambda x: x["created_at"], reverse=True)
    return {"bookings": result, "count": len(result), "unread": sum(1 for b in result if not b.get("read"))}

@app.put("/api/partner/bookings/{booking_id}/read")
async def mark_booking_read(booking_id: str):
    if booking_id not in _bookings:
        raise HTTPException(status_code=404, detail="Booking not found")
    _bookings[booking_id]["read"] = True
    return {"success": True}

@app.get("/api/listings")
async def get_all_listings(
    listing_type: Optional[str] = None,
    location: Optional[str] = None,
    max_price: Optional[float] = None,
):
    """Customer-facing: browse all active listings with optional filters"""
    result = [l for l in _listings.values() if l.get("status") == "active"]
    if listing_type:
        result = [l for l in result if l["listing_type"] == listing_type]
    if location:
        result = [l for l in result if location.lower() in l["location"].lower()]
    if max_price is not None:
        result = [l for l in result if l["price"] <= max_price]
    return {"listings": result, "count": len(result)}

@app.get("/api/listings/{listing_id}")
async def get_listing(listing_id: str):
    if listing_id not in _listings:
        raise HTTPException(status_code=404, detail="Listing not found")
    return _listings[listing_id]

@app.post("/api/customer/book")
async def customer_book(req: BookingCreate):
    """Customer books a listing — instantly visible in partner dashboard"""
    if req.listing_id not in _listings:
        raise HTTPException(status_code=404, detail="Listing not found")
    listing = _listings[req.listing_id]
    bid = "booking-" + str(uuid.uuid4())[:8]
    record = {
        "id": bid,
        "listing_id": req.listing_id,
        "listing_title": listing["title"],
        "listing_location": listing["location"],
        "listing_type": listing["listing_type"],
        "partner_id": listing["partner_id"],
        "customer_name": req.customer_name,
        "customer_phone": req.customer_phone,
        "customer_email": req.customer_email,
        "checkin": req.checkin,
        "checkout": req.checkout,
        "guests": req.guests,
        "total_price": req.total_price,
        "notes": req.notes,
        "persona": req.persona,
        "status": "confirmed",
        "read": False,
        "created_at": datetime.now().isoformat(),
    }
    _bookings[bid] = record
    _save_db()
    return {"success": True, "booking_id": bid, "booking": record}

@app.get("/api/customer/bookings")
async def get_customer_bookings(phone: str):
    """Get all bookings for a customer by phone"""
    result = [b for b in _bookings.values() if b.get("customer_phone") == phone]
    result.sort(key=lambda x: x["created_at"], reverse=True)
    return {"bookings": result, "count": len(result)}


# ─────────────────────────────────────────────
# Instagram Real OAuth Endpoints
# ─────────────────────────────────────────────

@app.get("/api/social/instagram/config")
async def instagram_config():
    """Tell the frontend whether real Instagram OAuth is configured."""
    return {
        "configured": bool(INSTAGRAM_CLIENT_ID and INSTAGRAM_CLIENT_SECRET),
        "redirect_uri": INSTAGRAM_REDIRECT_URI,
    }


@app.get("/api/social/instagram/auth")
async def instagram_auth():
    """
    Step 1 of OAuth: generate a state token and redirect the user to
    Instagram's authorisation page.  Opens in a popup from the frontend.
    """
    if not INSTAGRAM_CLIENT_ID:
        # Not configured — return an HTML page that tells the popup to fall back
        return HTMLResponse("""<!DOCTYPE html>
<html>
<head><title>Instagram – Not Configured</title></head>
<body style="font-family:sans-serif;text-align:center;padding:48px;background:#FFF7ED">
  <div style="font-size:2.5rem;margin-bottom:12px">⚠️</div>
  <div style="font-weight:700;color:#9A3412;margin-bottom:8px">Instagram OAuth not configured</div>
  <div style="color:#7C3000;font-size:0.85rem;margin-bottom:20px">
    Set <code>INSTAGRAM_CLIENT_ID</code> and <code>INSTAGRAM_CLIENT_SECRET</code> in your <code>.env</code> file.
  </div>
  <script>
    if (window.opener) {
      window.opener.postMessage({ type: 'instagram_oauth_result', success: false,
        error: 'Instagram OAuth not configured — see SETUP.md' }, '*');
      setTimeout(function(){ window.close(); }, 2500);
    }
  </script>
</body>
</html>""")

    state = secrets.token_urlsafe(20)
    _oauth_states[state] = {"created_at": datetime.now().isoformat()}

    auth_url = (
        "https://api.instagram.com/oauth/authorize"
        f"?client_id={INSTAGRAM_CLIENT_ID}"
        f"&redirect_uri={INSTAGRAM_REDIRECT_URI}"
        f"&scope=user_profile,user_media"
        f"&response_type=code"
        f"&state={state}"
    )
    return RedirectResponse(auth_url)


@app.get("/api/social/instagram/callback")
async def instagram_callback(
    code: Optional[str] = Query(None),
    state: Optional[str] = Query(None),
    error: Optional[str] = Query(None),
    error_reason: Optional[str] = Query(None),
):
    """
    Step 2 of OAuth: Instagram redirects here after user approval.
    We exchange the code for an access token, fetch recent posts,
    run the NLP engine, then post a message back to the opener popup.
    """
    def _error_page(msg: str) -> HTMLResponse:
        return HTMLResponse(f"""<!DOCTYPE html>
<html>
<head><title>TTT – Instagram Error</title></head>
<body style="font-family:sans-serif;text-align:center;padding:48px;background:#FFF7ED">
  <div style="font-size:2.5rem;margin-bottom:12px">❌</div>
  <div style="font-weight:700;color:#9A3412;margin-bottom:8px">Connection failed</div>
  <div style="color:#7C3000;font-size:0.85rem">{msg}</div>
  <script>
    if (window.opener) {{
      window.opener.postMessage({{ type: 'instagram_oauth_result', success: false, error: '{msg}' }}, '*');
      setTimeout(function(){{ window.close(); }}, 2000);
    }}
  </script>
</body>
</html>""")

    if error:
        return _error_page(f"Instagram denied access: {error_reason or error}")
    if not code or not state:
        return _error_page("Missing code or state parameter")
    if state not in _oauth_states:
        return _error_page("Invalid or expired state token — please try again")

    try:
        import httpx
        async with httpx.AsyncClient(timeout=15.0) as http:
            # ── Exchange authorisation code for access token ──
            token_resp = await http.post(
                "https://api.instagram.com/oauth/access_token",
                data={
                    "client_id":     INSTAGRAM_CLIENT_ID,
                    "client_secret": INSTAGRAM_CLIENT_SECRET,
                    "grant_type":    "authorization_code",
                    "redirect_uri":  INSTAGRAM_REDIRECT_URI,
                    "code":          code,
                },
            )
            token_data = token_resp.json()

            if "access_token" not in token_data:
                detail = token_data.get("error_message", str(token_data))
                return _error_page(f"Token exchange failed: {detail}")

            access_token = token_data["access_token"]
            user_id      = str(token_data.get("user_id", "me"))

            # ── Fetch user profile ──
            profile_resp = await http.get(
                "https://graph.instagram.com/" + user_id,
                params={"fields": "id,username,account_type,media_count",
                        "access_token": access_token},
            )
            profile  = profile_resp.json()
            username = profile.get("username", "")

            # Fetch up to 30 recent posts (captions + hashtags)
            media_resp = await http.get(
                "https://graph.instagram.com/" + user_id + "/media",
                params={"fields": "id,caption,media_type,timestamp",
                        "limit": 30,
                        "access_token": access_token},
            )
            media_data = media_resp.json()
            posts      = media_data.get("data", [])
            posts_count = len(posts)

            # Concatenate all captions for NLP
            all_text = " ".join(
                item.get("caption", "")
                for item in posts
                if item.get("caption")
            )

            # Run full NLP on real post captions (or handle if no captions)
            detail     = _analyze_profile_detailed(all_text or username, handle=username)
            categories = detail["categories_detected"] or ["leisure"]
            persona_id = detail["persona_id"]

            # Also log this OAuth-based analysis
            _analysis_log.append({
                "id":                "oauth-" + str(uuid.uuid4())[:8],
                "instagram_handle":  "@" + username,
                "linkedin_url":      "",
                "sources":           ["instagram_oauth"],
                "posts_scanned":     posts_count,
                "analysed_at":       datetime.now().isoformat(),
                **detail,
            })

            # Store result keyed by state token
            result = {
                "handle":        username,
                "user_id":       user_id,
                "access_token":  access_token,
                "categories":    categories,
                "persona_id":    persona_id,
                "posts_count":   posts_count,
                "media_count":   profile.get("media_count", posts_count),
                "account_type":  profile.get("account_type", ""),
                "analysed_at":   datetime.now().isoformat(),
                "analysis":      detail,   # full breakdown for frontend score panel
            }
            _instagram_profiles[state] = result
            _oauth_states.pop(state, None)

            result_json = json.dumps(result)
            cats_str = ", ".join(categories[:3])
            success_html = """<!DOCTYPE html>
<html>
<head><title>TTT - Connected</title></head>
<body style="font-family:'Inter',sans-serif;text-align:center;padding:48px 32px;background:#F0FDF4">
  <div style="font-size:3rem;margin-bottom:14px">&#x2705;</div>
  <div style="font-weight:800;font-size:1.1rem;color:#059669;margin-bottom:6px">
    Connected as @""" + username + """
  </div>
  <div style="color:#065F46;font-size:0.85rem;margin-bottom:4px">
    """ + str(posts_count) + """ posts analysed
  </div>
  <div style="color:#6B7280;font-size:0.78rem;margin-top:8px">
    Detected: """ + cats_str + """
  </div>
  <div style="color:#9CA3AF;font-size:0.72rem;margin-top:20px">Closing in a moment...</div>
  <script>
    var result = """ + result_json + """;
    result.type    = 'instagram_oauth_result';
    result.success = true;
    if (window.opener) {
      window.opener.postMessage(result, '*');
      setTimeout(function(){ window.close(); }, 1400);
    }
  </script>
</body>
</html>"""
            return HTMLResponse(success_html)

    except Exception as exc:
        return _error_page("Server error: " + str(exc)[:120])


@app.get("/api/social/instagram/result/{state}")
async def instagram_result(state: str):
    """Polling fallback: frontend polls this if postMessage did not arrive."""
    result = _instagram_profiles.get(state)
    if not result:
        return {"ready": False}
    return {"ready": True, **result}


# ---------------------------------------------
# Admin Analytics Endpoints (owner only)
# ---------------------------------------------

@app.get("/api/admin/analyses")
async def admin_get_analyses(key: str = Query("")):
    if key != ADMIN_KEY:
        raise HTTPException(status_code=401, detail="Invalid admin key. Pass ?key=<ADMIN_KEY>")
    return {
        "count": len(_analysis_log),
        "analyses": list(reversed(_analysis_log)),
        "signups": list(reversed(SIGNUPS_LOG)),  # also return signups here
    }

@app.get("/api/admin/travellers")
async def admin_get_travellers(key: str = Query("")):
    """
    Owner-only: returns all traveller profiles with their analysis results.
    """
    if key != ADMIN_KEY:
        raise HTTPException(status_code=401, detail="Invalid admin key. Pass ?key=<ADMIN_KEY>")
    enriched = {}
    for uid, u in _users.items():
        wallet = WALLETS.get(uid, {})
        enriched[uid] = {
            **u,
            "coins": wallet.get("coins", 0),
            "email": u.get("email") or "—",
            "phone": u.get("phone") or "—",
        }
    return {
        "count": len(_users),
        "travellers": enriched,
    }

@app.get("/api/admin/bookings")
async def admin_bookings(key: str = Query("")):
    if key != ADMIN_KEY:
        raise HTTPException(status_code=401, detail="Invalid admin key")
    # Combine _bookings (new system) + BOOKINGS_LOG (old system)
    all_bookings = list(_bookings.values())
    # Add from BOOKINGS_LOG if not already in _bookings
    booking_ids = {b.get('id') for b in all_bookings}
    for b in BOOKINGS_LOG:
        if b.get('booking_id') not in booking_ids:
            all_bookings.append({
                "id":         b.get('booking_id', '—'),
                "user_id":    b.get('user_id','—'),
                "service":    b.get('service') or b.get('listing_title','—'),
                "partner_id": b.get('partner_id','—'),
                "amount":     b.get('amount', 0),
                "coins":      b.get('coins', b.get('amount', 0)),
                "status":     b.get('status','confirmed'),
                "created_at": b.get('created_at') or b.get('timestamp','—'),
            })
    result = sorted(all_bookings, key=lambda x: x.get("created_at",""), reverse=True)
    total_revenue = sum(b.get("amount", 0) for b in result)
    return {"bookings": result, "count": len(result), "total_revenue": total_revenue}

@app.get("/api/admin/partners-list")
async def admin_partners(key: str = Query("")):
    if key != ADMIN_KEY:
        raise HTTPException(status_code=401, detail="Invalid admin key")
    enriched = []
    for pid, p in _partners.items():
        listings = [l for l in _listings.values() if l["partner_id"] == pid]
        bookings = [b for b in _bookings.values() if b.get("partner_id") == pid]
        enriched.append({**p, "listing_count": len(listings), "booking_count": len(bookings),
                         "revenue": sum(b.get("total_price", 0) for b in bookings)})
    return {
        "partners": {p["partner_id"]: p for p in enriched},
        "listings": _listings,
        "count": len(enriched)
    }

@app.get("/api/admin/summary")
async def admin_summary(key: str = Query("")):
    """
    Owner-only: high-level stats across all travellers.
    """
    if key != ADMIN_KEY:
        raise HTTPException(status_code=401, detail="Invalid admin key. Pass ?key=<ADMIN_KEY>")

    persona_counts: Dict[str, int] = {}
    category_counts: Dict[str, int] = {}
    all_locations: List[str] = []

    for a in _analysis_log:
        pid = a.get("persona_id", "unknown")
        persona_counts[pid] = persona_counts.get(pid, 0) + 1
        for cat in a.get("categories_detected", []):
            category_counts[cat] = category_counts.get(cat, 0) + 1
        all_locations.extend(a.get("locations_detected", []))

    from collections import Counter
    top_locations = [loc for loc, _ in Counter(all_locations).most_common(10)]

    return {
        "total_analyses":   len(_analysis_log),
        "total_travellers": len(_users),
        "persona_breakdown": persona_counts,
        "category_breakdown": category_counts,
        "top_locations_overall": top_locations,
        "admin_url": "/api/admin/analyses?key=" + ADMIN_KEY,
    }


# ---------------------------------------------
# Mock Responses (Demo without API key)
# ---------------------------------------------

def _mock_response(msg: str) -> str:
    msg = msg.lower()
    if any(w in msg for w in ["hello", "hi", "hey", "start", "help", "what can"]):
        return """\U0001f44b **Welcome to TTT - The Trip Theory!**

I'm your personal AI travel concierge for India and the world. Here's what I can do:

\U0001f5fa\ufe0f **Plan personalised trips** - Tell me where, how long, and your budget
\u2708\ufe0f **Compare flights** - Real-time options from major airlines
\U0001f3e8 **Find perfect stays** - Hotels matched to your style and budget
\U0001f37d\ufe0f **Book dining** - Restaurants based on your cuisine preferences
\U0001f3af **Handle everything** - Via our *Assist Now* end-to-end concierge

**Try asking:**
- *"Plan a 5-day trip to Goa under Rs.30,000"*
- *"Best honeymoon destinations under Rs.1,00,000"*
- *"Find flights from Mumbai to Manali for next weekend"*
- *"Create a family trip to Kerala for 7 days"*

Where would you like to travel? \U0001f30d"""

    if any(w in msg for w in ["goa", "beach"]):
        return """\u2708\ufe0f **Goa - Your Perfect Coastal Escape!**

Here's a curated **5-day Goa itinerary**:

**Day 1 - Arrival + North Goa Vibes**
- Check in | Evening at Baga/Calangute Beach | Dinner at Britto's (Rs.800-1,200)

**Day 2 - Adventure Day**
- Water sports at Calangute (Rs.1,500-2,500) | Anjuna Flea Market | Tito's Lane at night

**Day 3 - Old Goa & Culture**
- Basilica of Bom Jesus | Se Cathedral | Panjim city walk | Siolim market

**Day 4 - South Goa Serenity**
- Palolem & Agonda beaches | Sunset cruise (Rs.600) | Seafood dinner

**Day 5 - Chill & Depart**
- Morning market shopping | Checkout

\U0001f4b0 **Budget Breakdown (2 pax):**
| Item | Cost |
|------|------|
| Flights (return) | Rs.6,000-12,000 |
| Hotel (4 nights) | Rs.8,000-16,000 |
| Food & drinks | Rs.5,000-8,000 |
| Activities | Rs.3,000-5,000 |
| **Total** | **Rs.22,000-41,000** |

\U0001f3af Want me to search real-time flights or hotels? Or shall I connect you to **Assist Now** for a full concierge booking?"""

    if any(w in msg for w in ["rajasthan", "jaipur", "udaipur", "jodhpur"]):
        return """\U0001f3f0 **Rajasthan - The Land of Maharajas!**

A 7-day Golden Triangle + Lakes itinerary:

**Days 1-2: Jaipur (Pink City)** \U0001f338
- Amber Fort | City Palace | Hawa Mahal | Johari Bazaar

**Days 3-4: Jodhpur (Blue City)** \U0001f499
- Mehrangarh Fort | Jaswant Thada | Clocktower Market

**Days 5-7: Udaipur (City of Lakes)** \U0001f4a7
- City Palace | Lake Pichola boat ride | Jag Mandir | Saheliyon Ki Bari

\U0001f4b0 **Estimated Budget (per person):**
- Budget trip: Rs.18,000-25,000
- Mid-range: Rs.35,000-55,000
- Luxury: Rs.80,000-1,50,000

\U0001f3af Shall I pull up flights and hotels, or connect you to **Assist Now**?"""

    if any(w in msg for w in ["budget", "cheap", "affordable", "under"]):
        return """\U0001f4b0 **Budget Travel Tips from TTT:**

**Top Budget Destinations in India (under Rs.20,000/person for 5 days):**
- \U0001f3d6\ufe0f **Goa** - Off-season (Jun-Sep): Beach hostels from Rs.500/night
- \u26f0\ufe0f **Spiti Valley** - Homestays, raw beauty, low commercialisation
- \U0001f54c **Varanasi** - Spiritual immersion, cheapest ghats hotels from Rs.600
- \U0001f333 **Coorg** - Monsoon treks, coffee estates, budget homestays

**International Budget Picks (under Rs.60,000 all-in):**
- \U0001f418 **Vietnam** - 10 days for Rs.55,000 including flights
- \U0001f33a **Bali** - 7 days from Rs.45,000 in off-season
- \U0001f3ef **Sri Lanka** - 7 days from Rs.40,000

Which destination interests you? I'll build a complete plan! \U0001f5fa\ufe0f"""

    return """I'd love to help you plan an unforgettable trip! \U0001f30d

To create your perfect itinerary, tell me:

1. \U0001f4cd **Where** do you want to go?
2. \U0001f4c5 **How long** - number of days?
3. \U0001f4b0 **Budget** - total in Rs.?
4. \U0001f465 **Who's travelling** - solo, couple, family?
5. \U0001f3af **Vibe** - adventure, relaxation, culture, party?

Or try these quick starts:
- *"Plan a weekend getaway from Mumbai under Rs.10,000"*
- *"Best hill stations for a couple in December"*
- *"7-day international trip under Rs.80,000"*

*(Tip: Add your Anthropic API key in .env for full AI-powered responses!)*"""


def _mock_trip_plan(destination: str, duration: int, budget: float) -> str:
    return (
        f"## {destination} - {duration}-Day Itinerary (Budget: Rs.{budget:,.0f})\n\n"
        "*This is a sample plan. Add ANTHROPIC_API_KEY to .env for a fully personalised AI-generated itinerary.*\n\n"
        f"**Overview:** A wonderful {duration}-day journey to {destination}, blending culture, food, and adventure.\n\n"
        "**Day 1:** Arrival, check-in, evening orientation walk\n"
        "**Day 2:** Major sightseeing - historic sites and local markets\n"
        "**Day 3:** Nature/outdoor excursion\n"
        "**Days 4+:** Leisure, shopping, local experiences\n\n"
        "**Budget Estimate:**\n"
        f"- Transport: Rs.{budget * 0.35:,.0f}\n"
        f"- Accommodation: Rs.{budget * 0.30:,.0f}\n"
        f"- Food & dining: Rs.{budget * 0.20:,.0f}\n"
        f"- Activities: Rs.{budget * 0.15:,.0f}\n\n"
        "*Connect to Assist Now to have our concierge book everything for you!*"
    )


# ---------------------------------------------
# Run
# ---------------------------------------------
# ─────────────────────────────────────────────
# FREE Maps Integration (No API Key Required)
# Uses: OpenStreetMap Nominatim + Overpass API + Unsplash
# ─────────────────────────────────────────────

import httpx as _httpx

# Curated destination database with real Unsplash photos
DEST_DB = {
    "goa": {
        "lat": 15.2993, "lng": 74.1240,
        "tourist_attraction": [
            {"name":"Chapora Fort","rating":4.4,"reviews":8200,"address":"Chapora, North Goa","photo":"https://images.unsplash.com/photo-1582548961019-42d2be3ec6c3?w=600&q=80","maps":"https://maps.google.com/?q=Chapora+Fort+Goa","gem":True,"tip":"Visit at sunset for golden views over the Arabian Sea"},
            {"name":"Dudhsagar Waterfalls","rating":4.7,"reviews":22000,"address":"Mollem, South Goa","photo":"https://images.unsplash.com/photo-1601919051950-bb9f3ffb3fee?w=600&q=80","maps":"https://maps.google.com/?q=Dudhsagar+Waterfalls+Goa","gem":False,"tip":"Best visited post-monsoon (Oct–Dec) when water is full"},
            {"name":"Fontainhas Latin Quarter","rating":4.5,"reviews":4100,"address":"Panjim, Goa","photo":"https://images.unsplash.com/photo-1548013146-72479768bada?w=600&q=80","maps":"https://maps.google.com/?q=Fontainhas+Panjim+Goa","gem":True,"tip":"Walk the narrow lanes early morning before tourists arrive"},
            {"name":"Butterfly Beach","rating":4.6,"reviews":2800,"address":"Canacona, South Goa","photo":"https://images.unsplash.com/photo-1507525428034-b723cf961d3e?w=600&q=80","maps":"https://maps.google.com/?q=Butterfly+Beach+Goa","gem":True,"tip":"Only accessible by boat — completely secluded paradise"},
        ],
        "restaurant": [
            {"name":"Gunpowder","rating":4.6,"reviews":3200,"address":"Assagao, North Goa","photo":"https://images.unsplash.com/photo-1414235077428-338989a2e8c0?w=600&q=80","maps":"https://maps.google.com/?q=Gunpowder+Assagao+Goa","price":"₹₹","tip":"Try the Kerala-style fish curry — extraordinary"},
            {"name":"Antares Beach Restaurant","rating":4.5,"reviews":2800,"address":"Vagator Beach, Goa","photo":"https://images.unsplash.com/photo-1555396273-367ea4eb4db5?w=600&q=80","maps":"https://maps.google.com/?q=Antares+Vagator+Goa","price":"₹₹₹","tip":"Book the cliff table at sunset — unforgettable"},
            {"name":"Black Sheep Bistro","rating":4.7,"reviews":1900,"address":"Panjim, Goa","photo":"https://images.unsplash.com/photo-1504674900247-0877df9cc836?w=600&q=80","maps":"https://maps.google.com/?q=Black+Sheep+Bistro+Panjim","price":"₹₹₹","gem":True,"tip":"Best cocktail menu in Goa. Tiny place — always reserve"},
        ],
        "lodging": [
            {"name":"Ahilya by the Sea","rating":4.9,"reviews":420,"address":"Siolim, North Goa","photo":"https://images.unsplash.com/photo-1520250497591-112f2f40a3f4?w=600&q=80","maps":"https://maps.google.com/?q=Ahilya+by+the+Sea+Goa","price":"₹₹₹₹","gem":True,"tip":"A private heritage home — not a hotel. Only 5 rooms"},
            {"name":"Elsewhere","rating":4.8,"reviews":310,"address":"Mandrem Beach, Goa","photo":"https://images.unsplash.com/photo-1566073771259-6a8506099945?w=600&q=80","maps":"https://maps.google.com/?q=Elsewhere+Mandrem+Goa","price":"₹₹₹₹","gem":True,"tip":"Entire beach house rental — perfect for groups"},
        ]
    },
    "rajasthan": {
        "lat": 26.9124, "lng": 75.7873,
        "tourist_attraction": [
            {"name":"Mehrangarh Fort","rating":4.8,"reviews":42000,"address":"Jodhpur, Rajasthan","photo":"https://images.unsplash.com/photo-1477587458883-47145ed94245?w=600&q=80","maps":"https://maps.google.com/?q=Mehrangarh+Fort+Jodhpur","tip":"Arrive at 9am — before the crowds. Museum is world-class"},
            {"name":"Sam Sand Dunes, Jaisalmer","rating":4.6,"reviews":12000,"address":"Jaisalmer, Rajasthan","photo":"https://images.unsplash.com/photo-1509316785289-025f5b846b35?w=600&q=80","maps":"https://maps.google.com/?q=Sam+Sand+Dunes+Jaisalmer","tip":"Stay overnight in a luxury tent — worth every rupee"},
            {"name":"Bhangarh Fort","rating":4.4,"reviews":9800,"address":"Alwar, Rajasthan","photo":"https://images.unsplash.com/photo-1558618666-fcd25c85cd64?w=600&q=80","maps":"https://maps.google.com/?q=Bhangarh+Fort+Rajasthan","gem":True,"tip":"India's most haunted fort — eerily beautiful at dusk"},
            {"name":"Nahargarh Fort Sunset","rating":4.5,"reviews":18000,"address":"Jaipur, Rajasthan","photo":"https://images.unsplash.com/photo-1599661046289-e31897846e41?w=600&q=80","maps":"https://maps.google.com/?q=Nahargarh+Fort+Jaipur","tip":"Best sunset viewpoint over the Pink City"},
        ],
        "lodging": [
            {"name":"Rawla Narlai","rating":4.8,"reviews":890,"address":"Narlai Village, Rajasthan","photo":"https://images.unsplash.com/photo-1578683010236-d716f9a3f461?w=600&q=80","maps":"https://maps.google.com/?q=Rawla+Narlai+Rajasthan","price":"₹₹₹₹","gem":True,"tip":"A 400-year-old hunting lodge — utterly magical"},
            {"name":"Mihir Garh Fort Hotel","rating":4.9,"reviews":620,"address":"Rohet, Rajasthan","photo":"https://images.unsplash.com/photo-1566073771259-6a8506099945?w=600&q=80","maps":"https://maps.google.com/?q=Mihir+Garh+Rajasthan","price":"₹₹₹₹","gem":True,"tip":"9 unique suites in a real fort. Camel polo included"},
        ],
        "restaurant": [
            {"name":"Lassiwala (Famous)","rating":4.7,"reviews":6200,"address":"MI Road, Jaipur","photo":"https://images.unsplash.com/photo-1567620905732-2d1ec7ab7445?w=600&q=80","maps":"https://maps.google.com/?q=Lassiwala+Jaipur","price":"₹","tip":"The original. Opens 8am, closes when milk runs out"},
            {"name":"Saffron, Mehrangarh","rating":4.6,"reviews":1800,"address":"Mehrangarh Fort, Jodhpur","photo":"https://images.unsplash.com/photo-1414235077428-338989a2e8c0?w=600&q=80","maps":"https://maps.google.com/?q=Saffron+Restaurant+Mehrangarh","price":"₹₹₹","gem":True,"tip":"Dining inside a fort with views of the Blue City"},
        ]
    },
    "kerala": {
        "lat": 10.8505, "lng": 76.2711,
        "tourist_attraction": [
            {"name":"Alleppey Backwaters","rating":4.7,"reviews":28000,"address":"Alappuzha, Kerala","photo":"https://images.unsplash.com/photo-1593693411515-c20261bcad6e?w=600&q=80","maps":"https://maps.google.com/?q=Alleppey+Backwaters","tip":"Book an overnight houseboat — waking up on water is magical"},
            {"name":"Munroe Island","rating":4.6,"reviews":3200,"address":"Kollam, Kerala","photo":"https://images.unsplash.com/photo-1500534314209-a25ddb2bd429?w=600&q=80","maps":"https://maps.google.com/?q=Munroe+Island+Kerala","gem":True,"tip":"Tiny island only locals know about. Go by canoe"},
            {"name":"Varkala Cliff","rating":4.5,"reviews":14000,"address":"Varkala, Kerala","photo":"https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=600&q=80","maps":"https://maps.google.com/?q=Varkala+Cliff","tip":"Cliff-top cafes with sea views — best at golden hour"},
            {"name":"Athirappilly Waterfalls","rating":4.7,"reviews":19000,"address":"Thrissur, Kerala","photo":"https://images.unsplash.com/photo-1601919051950-bb9f3ffb3fee?w=600&q=80","maps":"https://maps.google.com/?q=Athirappilly+Falls+Kerala","tip":"India's Niagara — most powerful during July–September"},
        ],
        "restaurant": [
            {"name":"Malabar Junction","rating":4.6,"reviews":2100,"address":"Fort Kochi, Kerala","photo":"https://images.unsplash.com/photo-1414235077428-338989a2e8c0?w=600&q=80","maps":"https://maps.google.com/?q=Malabar+Junction+Fort+Kochi","price":"₹₹₹","tip":"Best Malabar fish biryani in Kerala — hands down"},
            {"name":"Dal Roti","rating":4.5,"reviews":980,"address":"Munnar, Kerala","photo":"https://images.unsplash.com/photo-1567620905732-2d1ec7ab7445?w=600&q=80","maps":"https://maps.google.com/?q=Dal+Roti+Munnar","price":"₹","gem":True,"tip":"Hidden cafe in a tea estate — order the banana pancakes"},
        ],
        "lodging": [
            {"name":"Fragrant Nature Backwaters","rating":4.8,"reviews":1200,"address":"Kumarakom, Kerala","photo":"https://images.unsplash.com/photo-1520250497591-112f2f40a3f4?w=600&q=80","maps":"https://maps.google.com/?q=Fragrant+Nature+Kumarakom","price":"₹₹₹₹","tip":"Infinity pool merging into the backwaters"},
        ]
    },
    "manali": {
        "lat": 32.2432, "lng": 77.1892,
        "tourist_attraction": [
            {"name":"Solang Valley","rating":4.6,"reviews":18000,"address":"Solang, Manali, HP","photo":"https://images.unsplash.com/photo-1464822759023-fed622ff2c3b?w=600&q=80","maps":"https://maps.google.com/?q=Solang+Valley+Manali","tip":"Paragliding here is life-changing — book in advance"},
            {"name":"Hampta Pass Trek","rating":4.8,"reviews":4100,"address":"Kullu, Himachal Pradesh","photo":"https://images.unsplash.com/photo-1483728642387-6c3bdd6c93e5?w=600&q=80","maps":"https://maps.google.com/?q=Hampta+Pass+Trek","gem":True,"tip":"3-day trek crossing from green valleys to barren desert"},
            {"name":"Old Manali Village","rating":4.5,"reviews":6200,"address":"Old Manali, HP","photo":"https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=600&q=80","maps":"https://maps.google.com/?q=Old+Manali+Village","gem":True,"tip":"Walk the village lanes — far more authentic than tourist Manali"},
            {"name":"Chandratal Lake","rating":4.9,"reviews":5600,"address":"Spiti Valley, HP","photo":"https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=600&q=80","maps":"https://maps.google.com/?q=Chandratal+Lake+Spiti","gem":True,"tip":"Moon lake at 4300m — remote and breathtaking"},
        ],
        "restaurant": [
            {"name":"Johnson's Cafe","rating":4.5,"reviews":3800,"address":"The Mall, Manali","photo":"https://images.unsplash.com/photo-1414235077428-338989a2e8c0?w=600&q=80","maps":"https://maps.google.com/?q=Johnson+Cafe+Manali","price":"₹₹","tip":"Best trout fish in Manali. Cozy fireplace in winter"},
            {"name":"Cafe 1947","rating":4.4,"reviews":2200,"address":"Old Manali","photo":"https://images.unsplash.com/photo-1501339847302-ac426a4a7cbb?w=600&q=80","maps":"https://maps.google.com/?q=Cafe+1947+Old+Manali","price":"₹","gem":True,"tip":"Rooftop seating with Himalayan views. Try apple pie"},
        ]
    },
    "bali": {
        "lat": -8.4095, "lng": 115.1889,
        "tourist_attraction": [
            {"name":"Tegallalang Rice Terraces","rating":4.6,"reviews":31000,"address":"Ubud, Bali","photo":"https://images.unsplash.com/photo-1555400038-63f5ba517a47?w=600&q=80","maps":"https://maps.google.com/?q=Tegallalang+Rice+Terraces+Bali","tip":"Visit at 7am before the Instagram crowds arrive"},
            {"name":"Tanah Lot Temple","rating":4.5,"reviews":48000,"address":"Tabanan, Bali","photo":"https://images.unsplash.com/photo-1537996194471-e657df975ab4?w=600&q=80","maps":"https://maps.google.com/?q=Tanah+Lot+Bali","tip":"Temple on a sea rock — spectacular at sunset"},
            {"name":"Secret Waterfall, Munduk","rating":4.8,"reviews":2100,"address":"Munduk, North Bali","photo":"https://images.unsplash.com/photo-1606117331085-5760e3097277?w=600&q=80","maps":"https://maps.google.com/?q=Munduk+Waterfall+Bali","gem":True,"tip":"80% of Bali tourists never leave the south. Come here"},
        ],
        "restaurant": [
            {"name":"Locavore","rating":4.8,"reviews":4200,"address":"Ubud, Bali","photo":"https://images.unsplash.com/photo-1414235077428-338989a2e8c0?w=600&q=80","maps":"https://maps.google.com/?q=Locavore+Restaurant+Ubud+Bali","price":"₹₹₹₹","tip":"World's 50 best restaurant. Book 3 months ahead"},
            {"name":"Warung Babi Guling Ibu Oka","rating":4.7,"reviews":8900,"address":"Ubud, Bali","photo":"https://images.unsplash.com/photo-1504674900247-0877df9cc836?w=600&q=80","maps":"https://maps.google.com/?q=Babi+Guling+Ibu+Oka+Ubud","price":"₹","tip":"Anthony Bourdain ate here. ₹500 for legendary suckling pig"},
        ]
    },
    "dubai": {
        "lat": 25.2048, "lng": 55.2708,
        "tourist_attraction": [
            {"name":"Burj Khalifa At The Top","rating":4.7,"reviews":82000,"address":"Downtown Dubai","photo":"https://images.unsplash.com/photo-1512453979798-5ea266f8880c?w=600&q=80","maps":"https://maps.google.com/?q=Burj+Khalifa+Dubai","tip":"Book level 148 (SKY) — worth the premium. Go at dusk"},
            {"name":"Al Fahidi Historical Neighbourhood","rating":4.6,"reviews":12000,"address":"Bur Dubai","photo":"https://images.unsplash.com/photo-1548013146-72479768bada?w=600&q=80","maps":"https://maps.google.com/?q=Al+Fahidi+Dubai","gem":True,"tip":"Old Dubai that most tourists miss. Free to explore"},
            {"name":"Dubai Frame","rating":4.4,"reviews":28000,"address":"Zabeel Park, Dubai","photo":"https://images.unsplash.com/photo-1577447718955-d1f24e9c6b20?w=600&q=80","maps":"https://maps.google.com/?q=Dubai+Frame","tip":"Glass floor at 150m height — terrifying and thrilling"},
        ],
        "restaurant": [
            {"name":"Nobu Dubai","rating":4.8,"reviews":3800,"address":"Atlantis The Palm, Dubai","photo":"https://images.unsplash.com/photo-1555396273-367ea4eb4db5?w=600&q=80","maps":"https://maps.google.com/?q=Nobu+Dubai+Atlantis","price":"₹₹₹₹","tip":"Best omakase outside Japan. Reserve 2 weeks ahead"},
            {"name":"Arabian Tea House","rating":4.6,"reviews":6200,"address":"Al Fahidi, Dubai","photo":"https://images.unsplash.com/photo-1501339847302-ac426a4a7cbb?w=600&q=80","maps":"https://maps.google.com/?q=Arabian+Tea+House+Dubai","price":"₹₹","gem":True,"tip":"Hidden courtyard in old Dubai — Instagram gold, also delicious"},
        ]
    }
}

async def _osm_search(destination: str, category: str, limit: int = 6):
    """Use OpenStreetMap Overpass API - completely free, no key needed"""
    # First geocode with Nominatim (free)
    try:
        async with _httpx.AsyncClient(headers={"User-Agent": "TTT-TripTheory/1.0"}) as c:
            r = await c.get(
                "https://nominatim.openstreetmap.org/search",
                params={"q": destination + " India", "format": "json", "limit": 1},
                timeout=8
            )
            data = r.json()
            if not data:
                return None, None
            lat, lng = float(data[0]["lat"]), float(data[0]["lon"])
        return lat, lng
    except:
        return None, None

@app.post("/api/maps/places")
async def maps_places(destination: str, category: str = "tourist_attraction", hidden_gems: bool = False, limit: int = 6):
    """Get curated places for a destination — no API key needed"""
    dest_key = destination.lower().strip()
    
    # Check curated database first
    for key in DEST_DB:
        if key in dest_key or dest_key in key:
            db = DEST_DB[key]
            places = db.get(category, db.get("tourist_attraction", []))
            if hidden_gems:
                places = [p for p in places if p.get("gem", False)]
            return {
                "destination": destination,
                "category": category,
                "places": places[:limit],
                "lat": db["lat"],
                "lng": db["lng"],
                "source": "curated",
                "maps_search_url": f"https://www.google.com/maps/search/{category}+in+{destination}"
            }
    
    # Try OpenStreetMap for unknown destinations
    lat, lng = await _osm_search(destination, category)
    if lat:
        return {
            "destination": destination,
            "category": category,
            "places": [{
                "name": f"Explore {destination}",
                "rating": None,
                "address": destination,
                "photo": "https://images.unsplash.com/photo-1469474968028-56623f02e42e?w=600&q=80",
                "maps": f"https://www.google.com/maps/search/{category}+in+{destination}",
                "tip": f"Click to explore {category} in {destination} on Google Maps"
            }],
            "lat": lat, "lng": lng,
            "source": "openstreetmap",
            "maps_search_url": f"https://www.google.com/maps/search/{category}+in+{destination}"
        }
    
    return {
        "destination": destination,
        "category": category,
        "places": [],
        "maps_search_url": f"https://www.google.com/maps/search/{category}+in+{destination}"
    }

@app.get("/api/maps/embed")
async def maps_embed(destination: str, category: str = "tourist_attraction"):
    """OpenStreetMap embed — completely free, no key"""
    # Use OpenStreetMap embed
    osm_url = f"https://www.openstreetmap.org/export/embed.html?bbox=&layer=mapnik&marker="
    search_url = f"https://www.google.com/maps/search/{category}+in+{destination}"
    return {
        "embed_url": None,
        "search_url": search_url,
        "iframe": f'<div style="background:#f5f3ef;border-radius:12px;padding:20px;text-align:center"><p style="color:#555;font-size:.88rem;margin-bottom:12px">📍 Explore {destination} on Google Maps</p><a href="{search_url}" target="_blank" style="background:#B8860B;color:white;padding:10px 24px;border-radius:6px;text-decoration:none;font-weight:600;font-size:.85rem">Open {destination} Map →</a></div>'
    }

if __name__ == "__main__":
    import uvicorn
    print("\n\U0001f30d TTT - The Trip Theory API")
    print("=" * 40)
    ai_status = "Yes" if client else "No (add ANTHROPIC_API_KEY to .env)"
    ig_status = "Yes" if (INSTAGRAM_CLIENT_ID and INSTAGRAM_CLIENT_SECRET) else "No (see SETUP.md)"
    print(f"AI Connected:      {ai_status}")
    print(f"Instagram OAuth:   {ig_status}")
    print("Docs:              http://localhost:8000/docs")
    print("Frontend:          http://localhost:8000\n")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

# ══════════════════════════════════════════════════════
#   TTT WALLET & PAYMENT SYSTEM
#   1 coin = ₹1 | Traveller loads → spends | Partner earns → monthly payout
# ══════════════════════════════════════════════════════
import hashlib
from datetime import datetime, timedelta

# ── In-memory wallet store (replace with DB in production) ──
WALLETS: Dict[str, Any] = {}       # user_id → wallet
TRANSACTIONS: List[Dict] = []      # all transactions log
PARTNER_WALLETS: Dict[str, Any] = {}  # partner_id → wallet
RECONCILIATIONS: List[Dict] = []   # monthly payout records

def get_or_create_wallet(user_id: str, user_name: str = "", user_type: str = "traveller") -> Dict:
    if user_id not in WALLETS:
        WALLETS[user_id] = {
            "user_id": user_id,
            "user_name": user_name,
            "user_type": user_type,
            "coins": 0,
            "total_loaded": 0,
            "total_spent": 0,
            "created_at": datetime.now().isoformat(),
        }
    return WALLETS[user_id]

def get_or_create_partner_wallet(partner_id: str, partner_name: str = "") -> Dict:
    if partner_id not in PARTNER_WALLETS:
        PARTNER_WALLETS[partner_id] = {
            "partner_id": partner_id,
            "partner_name": partner_name,
            "coins": 0,
            "total_earned": 0,
            "total_paid_out": 0,
            "pending_payout": 0,
            "created_at": datetime.now().isoformat(),
        }
    return PARTNER_WALLETS[partner_id]

# ── Pydantic Models ──
class WalletLoadRequest(BaseModel):
    user_id: str
    user_name: str
    amount: float           # in ₹ → converts to coins 1:1
    payment_method: str     # razorpay | upi | phonepe | paytm | stripe
    payment_ref: str = ""   # payment gateway reference

class SpendCoinsRequest(BaseModel):
    user_id: str
    coins: float
    service: str            # flights | hotels | dining | activities | spa | membership
    description: str
    partner_id: str = ""    # if applicable
    partner_name: str = ""

class PaymentOrderRequest(BaseModel):
    user_id: str
    user_name: str
    amount: float
    payment_method: str

class PartnerPayoutRequest(BaseModel):
    partner_id: str
    partner_name: str = ""

# ── RAZORPAY ORDER CREATE ──
@app.post("/api/payment/create-order")
async def create_payment_order(req: PaymentOrderRequest):
    """Create a payment order. In production plug in real Razorpay/Stripe SDK."""
    order_id = "ORD_" + uuid.uuid4().hex[:12].upper()
    return {
        "success": True,
        "order_id": order_id,
        "amount": req.amount,
        "amount_paise": int(req.amount * 100),  # Razorpay uses paise
        "currency": "INR",
        "user_id": req.user_id,
        "payment_method": req.payment_method,
        # In production: use razorpay.Order.create() and return razorpay order_id
        "razorpay_key": "rzp_test_YOUR_KEY_HERE",
        "message": "Order created. Complete payment to load coins."
    }

# ── WALLET LOAD (after payment success) ──
@app.post("/api/wallet/load")
async def load_wallet(req: WalletLoadRequest):
    """Called after successful payment. Loads coins into traveller wallet."""
    wallet = get_or_create_wallet(req.user_id, req.user_name)
    coins_to_add = req.amount  # 1 coin = ₹1

    wallet["coins"] += coins_to_add
    wallet["total_loaded"] += coins_to_add

    # Log transaction
    txn = {
        "txn_id": "TXN_" + uuid.uuid4().hex[:10].upper(),
        "type": "credit",
        "user_id": req.user_id,
        "user_name": req.user_name,
        "coins": coins_to_add,
        "amount_inr": req.amount,
        "payment_method": req.payment_method,
        "payment_ref": req.payment_ref,
        "description": f"Wallet loaded via {req.payment_method}",
        "service": "wallet_load",
        "timestamp": datetime.now().isoformat(),
        "balance_after": wallet["coins"],
    }
    TRANSACTIONS.append(txn)

    return {
        "success": True,
        "message": f"✅ ₹{req.amount:,.0f} loaded! {coins_to_add:,.0f} TTT Coins added to wallet.",
        "wallet": wallet,
        "transaction": txn,
    }

# ── SPEND COINS ──
@app.post("/api/wallet/spend")
async def spend_coins(req: SpendCoinsRequest):
    """Deduct coins from traveller, credit to partner if applicable."""
    wallet = get_or_create_wallet(req.user_id)
    if wallet["coins"] < req.coins:
        raise HTTPException(400, f"Insufficient coins. Available: {wallet['coins']:,.0f}, Required: {req.coins:,.0f}")

    wallet["coins"] -= req.coins
    wallet["total_spent"] += req.coins

    # Credit partner wallet if partner_id provided
    partner_txn = None
    if req.partner_id:
        pw = get_or_create_partner_wallet(req.partner_id, req.partner_name)
        pw["coins"] += req.coins
        pw["total_earned"] += req.coins
        pw["pending_payout"] += req.coins
        partner_txn = {
            "txn_id": "PTX_" + uuid.uuid4().hex[:10].upper(),
            "type": "partner_credit",
            "partner_id": req.partner_id,
            "coins": req.coins,
            "service": req.service,
            "description": req.description,
            "timestamp": datetime.now().isoformat(),
        }
        TRANSACTIONS.append(partner_txn)

    # Log traveller debit
    txn = {
        "txn_id": "TXN_" + uuid.uuid4().hex[:10].upper(),
        "type": "debit",
        "user_id": req.user_id,
        "coins": req.coins,
        "amount_inr": req.coins,
        "service": req.service,
        "description": req.description,
        "partner_id": req.partner_id,
        "timestamp": datetime.now().isoformat(),
        "balance_after": wallet["coins"],
    }
    TRANSACTIONS.append(txn)

    # ── Save booking record ─────────────────────────────────────────
    bid = "BKG_" + uuid.uuid4().hex[:10].upper()
    booking_record = {
        "id":         bid,
        "user_id":    req.user_id,
        "service":    req.service,
        "partner_id": req.partner_id,
        "coins":      req.coins,
        "amount":     req.coins,
        "status":     "confirmed",
        "created_at": datetime.now().isoformat(),
    }
    _bookings[bid] = booking_record
    log_activity(req.user_id, "booking", {"service": req.service, "amount": req.coins, "booking_id": bid})
    _save_db()

    return {
        "success": True,
        "message": f"✅ {req.coins:,.0f} coins spent on {req.service}.",
        "wallet": wallet,
        "transaction": txn,
        "partner_credited": partner_txn is not None,
        "booking_id": bid,
        "balance_after": wallet["coins"],
    }

# ── GET WALLET BALANCE ──
@app.get("/api/wallet/{user_id}")
async def get_wallet(user_id: str):
    wallet = WALLETS.get(user_id)
    if not wallet:
        return {"user_id": user_id, "coins": 0, "total_loaded": 0, "total_spent": 0, "exists": False}
    return {**wallet, "exists": True, "coins_inr_value": wallet["coins"]}

# ── GET WALLET TRANSACTIONS ──
@app.get("/api/wallet/{user_id}/transactions")
async def get_transactions(user_id: str, limit: int = 20):
    user_txns = [t for t in TRANSACTIONS if t.get("user_id") == user_id]
    return {"transactions": list(reversed(user_txns))[:limit], "total": len(user_txns)}

# ── PARTNER WALLET ──
@app.get("/api/partner/wallet/{partner_id}")
async def get_partner_wallet(partner_id: str):
    pw = PARTNER_WALLETS.get(partner_id)
    if not pw:
        return {"partner_id": partner_id, "coins": 0, "exists": False}
    return {**pw, "exists": True, "pending_inr": pw["pending_payout"]}

# ── MONTHLY RECONCILIATION (Admin triggers this) ──
@app.post("/api/admin/reconcile")
async def monthly_reconciliation():
    """Process monthly payouts to all partners."""
    payouts = []
    total_paid = 0

    for partner_id, pw in PARTNER_WALLETS.items():
        if pw["pending_payout"] > 0:
            amount = pw["pending_payout"]
            rec = {
                "reconciliation_id": "REC_" + uuid.uuid4().hex[:8].upper(),
                "partner_id": partner_id,
                "partner_name": pw["partner_name"],
                "coins_redeemed": amount,
                "amount_inr": amount,  # 1 coin = ₹1
                "month": datetime.now().strftime("%B %Y"),
                "status": "processed",
                "timestamp": datetime.now().isoformat(),
            }
            pw["total_paid_out"] += amount
            pw["pending_payout"] = 0
            pw["coins"] -= amount
            RECONCILIATIONS.append(rec)
            payouts.append(rec)
            total_paid += amount

    return {
        "success": True,
        "message": f"Monthly reconciliation complete. ₹{total_paid:,.0f} processed to {len(payouts)} partners.",
        "payouts": payouts,
        "total_inr": total_paid,
        "month": datetime.now().strftime("%B %Y"),
    }

# ── ADMIN OVERVIEW ──
@app.get("/api/admin/wallet-overview")
async def wallet_overview():
    total_coins_in_circulation = sum(w["coins"] for w in WALLETS.values())
    total_partner_coins = sum(pw["coins"] for pw in PARTNER_WALLETS.values())
    total_loaded_ever = sum(w["total_loaded"] for w in WALLETS.values())
    total_spent_ever = sum(w["total_spent"] for w in WALLETS.values())
    return {
        "traveller_wallets": len(WALLETS),
        "partner_wallets": len(PARTNER_WALLETS),
        "coins_with_travellers": total_coins_in_circulation,
        "coins_with_partners": total_partner_coins,
        "total_loaded_inr": total_loaded_ever,
        "total_spent_inr": total_spent_ever,
        "total_transactions": len(TRANSACTIONS),
        "reconciliations_done": len(RECONCILIATIONS),
    }

# ══════════════════════════════════════════════════════
#   📧 EMAIL / WHATSAPP NOTIFICATIONS SYSTEM
# ══════════════════════════════════════════════════════
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Config from .env
SMTP_HOST     = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT     = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER     = os.getenv("SMTP_USER", "")
SMTP_PASS     = os.getenv("SMTP_PASS", "")
WHATSAPP_KEY  = os.getenv("WHATSAPP_API_KEY", "")  # Twilio or WATI
WHATSAPP_FROM = os.getenv("WHATSAPP_FROM", "")

class NotificationRequest(BaseModel):
    user_email: str = ""
    user_phone: str = ""
    user_name: str = ""
    notification_type: str  # booking_confirm | wallet_load | trip_reminder | welcome | reconciliation
    data: Dict[str, Any] = {}

def get_email_template(notif_type: str, data: dict, user_name: str) -> tuple:
    """Returns (subject, html_body) for each notification type"""
    gold = "#C9963A"
    dark = "#1A1208"
    cream = "#FDFAF4"

    base_style = f"""
    <div style="font-family:'Georgia',serif;max-width:580px;margin:0 auto;background:{cream}">
      <div style="background:{dark};padding:28px 32px;text-align:center">
        <div style="font-size:2.2rem;color:{gold};letter-spacing:0.1em;font-weight:400">TTT</div>
        <div style="font-size:0.62rem;color:rgba(255,255,255,0.4);letter-spacing:0.25em;text-transform:uppercase;margin-top:4px">The Trip Theory</div>
      </div>
    """

    if notif_type == "booking_confirm":
        service = data.get("service","Booking")
        coins   = data.get("coins", 0)
        detail  = data.get("detail","")
        subject = f"✅ Your TTT {service} is confirmed!"
        body = base_style + f"""
      <div style="padding:32px">
        <h2 style="color:{dark};font-size:1.4rem;font-weight:400;margin-bottom:8px">Your booking is confirmed, {user_name}! ✈️</h2>
        <p style="color:#6B5030;line-height:1.75;margin-bottom:20px">Your TTT concierge has locked in your <strong>{service}</strong>. Here are the details:</p>
        <div style="background:#FFF;border:1px solid rgba(201,150,58,0.2);border-radius:8px;padding:20px;margin-bottom:20px">
          <p style="color:{dark};font-size:0.9rem;line-height:1.7">{detail}</p>
        </div>
        <div style="background:linear-gradient(135deg,{dark},#2A1E0A);border-radius:8px;padding:18px;text-align:center;margin-bottom:20px">
          <div style="font-size:0.62rem;color:rgba(201,150,58,0.65);letter-spacing:0.2em;text-transform:uppercase">Coins Spent</div>
          <div style="font-size:2rem;color:{gold};font-weight:300;margin:4px 0">{coins:,}</div>
          <div style="font-size:0.72rem;color:rgba(255,255,255,0.35)">TTT Coins · ₹{coins:,} value</div>
        </div>
        <p style="color:#6B5030;font-size:0.85rem;line-height:1.75">Need to make changes? Your concierge is always reachable. Just reply to this email or WhatsApp us.</p>
      </div>
        """

    elif notif_type == "wallet_load":
        amount  = data.get("amount", 0)
        method  = data.get("method","")
        balance = data.get("balance", 0)
        subject = f"💰 ₹{amount:,} loaded to your TTT Wallet"
        body = base_style + f"""
      <div style="padding:32px">
        <h2 style="color:{dark};font-size:1.4rem;font-weight:400;margin-bottom:8px">Wallet loaded! 🪙</h2>
        <p style="color:#6B5030;line-height:1.75;margin-bottom:20px">Hi {user_name}, ₹{amount:,} has been successfully added to your TTT Wallet via {method.upper()}.</p>
        <div style="background:linear-gradient(135deg,{dark},#2A1E0A);border-radius:8px;padding:20px;text-align:center;margin-bottom:20px">
          <div style="color:rgba(201,150,58,0.65);font-size:0.6rem;letter-spacing:0.2em;text-transform:uppercase">Current Balance</div>
          <div style="font-size:2.2rem;color:{gold};font-weight:300;margin:6px 0">{balance:,} coins</div>
          <div style="color:rgba(255,255,255,0.35);font-size:0.72rem">₹{balance:,} value</div>
        </div>
        <p style="color:#6B5030;font-size:0.85rem;line-height:1.75">Use your coins to book flights, hotels, dining, activities, and spa — all in one tap through TTT.</p>
      </div>
        """

    elif notif_type == "welcome":
        subject = f"Welcome to TTT, {user_name}! Your journey starts here 🌍"
        body = base_style + f"""
      <div style="padding:32px">
        <h2 style="color:{dark};font-size:1.5rem;font-weight:400;margin-bottom:8px">Welcome to The Trip Theory, {user_name}! ✦</h2>
        <p style="color:#6B5030;line-height:1.75;margin-bottom:20px">India's first Agentic AI travel concierge is now yours. Here's what you can do:</p>
        <div style="display:flex;flex-direction:column;gap:12px;margin-bottom:24px">
          {''.join([f'<div style="background:#FFF;border-left:3px solid {gold};padding:12px 16px;border-radius:0 8px 8px 0"><strong style="color:{dark}">{icon} {title}</strong><p style="color:#6B5030;font-size:0.82rem;margin:3px 0 0">{desc}</p></div>' for icon,title,desc in [
            ("🤖","Chat with Aria","Your AI concierge who plans trips based on who you are"),
            ("✈️","Search Flights & Hotels","Real-time results, best prices, instant booking"),
            ("🪙","Load Your TTT Wallet","₹1 = 1 coin. Pay for everything in one tap"),
            ("💎","Go Connoisseur","₹24,999/year. Never miss a long weekend again"),
          ]])}
        </div>
        <div style="text-align:center">
          <a href="http://localhost:8000" style="background:{gold};color:{dark};padding:13px 32px;text-decoration:none;font-weight:700;font-size:0.8rem;letter-spacing:0.12em;text-transform:uppercase;border-radius:4px">Start Planning →</a>
        </div>
      </div>
        """

    elif notif_type == "trip_reminder":
        destination = data.get("destination","your destination")
        days_away   = data.get("days_away", 3)
        subject     = f"⏰ {days_away} days until your trip to {destination}!"
        body = base_style + f"""
      <div style="padding:32px">
        <h2 style="color:{dark};font-size:1.4rem;font-weight:400;margin-bottom:8px">Almost time, {user_name}! 🎒</h2>
        <p style="color:#6B5030;line-height:1.75;margin-bottom:20px">Your trip to <strong>{destination}</strong> is {days_away} days away. Here's your pre-trip checklist:</p>
        <div style="background:#FFF;border:1px solid rgba(201,150,58,0.2);border-radius:8px;padding:20px;margin-bottom:20px">
          {''.join([f'<div style="display:flex;align-items:center;gap:10px;padding:8px 0;border-bottom:1px solid #F0E8D8"><span style="font-size:1rem">{e}</span><span style="color:{dark};font-size:0.85rem">{t}</span></div>' for e,t in [
            ("✅","Confirm all bookings with your TTT concierge"),
            ("📱","Download your itinerary"),
            ("🪙","Check your TTT Wallet balance"),
            ("📞","Save TTT concierge number: +91 98765 00000"),
            ("🧳","Packing list sent to your email"),
          ]])}
        </div>
      </div>
        """

    elif notif_type == "reconciliation":
        partner_name = data.get("partner_name","Partner")
        coins        = data.get("coins", 0)
        month        = data.get("month","")
        subject      = f"💰 TTT Monthly Payout — ₹{coins:,} for {month}"
        body = base_style + f"""
      <div style="padding:32px">
        <h2 style="color:{dark};font-size:1.4rem;font-weight:400;margin-bottom:8px">Monthly payout processed, {partner_name}! 🎉</h2>
        <p style="color:#6B5030;line-height:1.75;margin-bottom:20px">Your TTT earnings for <strong>{month}</strong> have been reconciled and queued for bank transfer.</p>
        <div style="background:linear-gradient(135deg,{dark},#2A1E0A);border-radius:8px;padding:20px;text-align:center;margin-bottom:20px">
          <div style="color:rgba(201,150,58,0.65);font-size:0.6rem;letter-spacing:0.2em;text-transform:uppercase">Payout Amount</div>
          <div style="font-size:2.2rem;color:{gold};font-weight:300;margin:6px 0">₹{coins:,}</div>
          <div style="color:rgba(255,255,255,0.35);font-size:0.72rem">{coins:,} TTT Coins redeemed</div>
        </div>
        <p style="color:#6B5030;font-size:0.85rem;line-height:1.75">Transfer will appear in your registered bank account within 3-5 business days. Questions? Email us at partners@triptheory.in</p>
      </div>
        """
    else:
        subject = "TTT — Notification"
        body = base_style + f"<div style='padding:32px'><p style='color:#6B5030'>Hi {user_name}, you have a new notification from TTT.</p></div>"

    # Footer
    body += f"""
      <div style="background:{dark};padding:20px 32px;text-align:center">
        <div style="font-size:0.65rem;color:rgba(255,255,255,0.25);line-height:1.8">
          The Trip Theory · Gurugram, Haryana, India<br/>
          concierge@triptheory.in · +91 98765 00000<br/>
          <a href="#" style="color:rgba(201,150,58,0.5)">Unsubscribe</a>
        </div>
      </div>
    </div>
    """
    return subject, body

def send_email_notification(to_email: str, subject: str, html_body: str) -> bool:
    """Send email via SMTP. Returns True if sent."""
    if not SMTP_USER or not SMTP_PASS:
        print(f"📧 [DEMO] Email would send to {to_email}: {subject}")
        return True  # Demo mode — log but don't fail
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = f"TTT — The Trip Theory <{SMTP_USER}>"
        msg["To"]      = to_email
        msg.attach(MIMEText(html_body, "html"))
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASS)
            server.sendmail(SMTP_USER, to_email, msg.as_string())
        return True
    except Exception as e:
        print(f"Email error: {e}")
        return False

def send_whatsapp_notification(to_phone: str, notif_type: str, data: dict, user_name: str) -> bool:
    """Send WhatsApp via WATI/Twilio. Returns True if sent."""
    messages = {
        "booking_confirm": f"✅ Hi {user_name}! Your TTT {data.get('service','booking')} is confirmed. {data.get('detail','')} — Your TTT Concierge 🌍",
        "wallet_load":     f"🪙 Hi {user_name}! ₹{data.get('amount',0):,} loaded to your TTT Wallet. Balance: {data.get('balance',0):,} coins. Start booking! — TTT",
        "welcome":         f"👋 Welcome to TTT, {user_name}! India's first AI travel concierge is ready for you. Reply with where you want to go next! — Aria, TTT",
        "trip_reminder":   f"⏰ {user_name}, your trip to {data.get('destination','your destination')} is {data.get('days_away',3)} days away! Check your TTT app for details. — TTT Concierge",
        "reconciliation":  f"💰 {user_name}, your TTT payout of ₹{data.get('coins',0):,} for {data.get('month','')} has been processed! Transfer in 3-5 days. — TTT Partners",
    }
    msg = messages.get(notif_type, f"Hi {user_name}, you have a notification from TTT.")
    if not WHATSAPP_KEY:
        print(f"📱 [DEMO] WhatsApp would send to {to_phone}: {msg}")
        return True
    # Real WATI integration (add your endpoint)
    try:
        import httpx
        r = httpx.post(
            f"https://live-mt-server.wati.io/api/v1/sendSessionMessage/{to_phone}",
            headers={"Authorization": f"Bearer {WHATSAPP_KEY}"},
            json={"messageText": msg},
            timeout=10
        )
        return r.status_code == 200
    except Exception as e:
        print(f"WhatsApp error: {e}")
        return False

@app.post("/api/notify")
async def send_notification(req: NotificationRequest):
    """Send email + WhatsApp notification."""
    subject, html_body = get_email_template(req.notification_type, req.data, req.user_name)
    email_sent = wa_sent = False
    if req.user_email:
        email_sent = send_email_notification(req.user_email, subject, html_body)
    if req.user_phone:
        wa_sent = send_whatsapp_notification(req.user_phone, req.notification_type, req.data, req.user_name)
    return {
        "success": True,
        "email_sent": email_sent,
        "whatsapp_sent": wa_sent,
        "notification_type": req.notification_type,
        "message": f"Notifications dispatched for {req.notification_type}"
    }

@app.post("/api/notify/welcome")
async def welcome_notification(req: NotificationRequest):
    req.notification_type = "welcome"
    return await send_notification(req)

@app.post("/api/notify/booking")
async def booking_notification(req: NotificationRequest):
    req.notification_type = "booking_confirm"
    return await send_notification(req)


# ══════════════════════════════════════════════════════
#   📊 ADMIN ANALYTICS DASHBOARD
# ══════════════════════════════════════════════════════

# In-memory stats store
BOOKINGS_LOG: List[Dict] = []
# SIGNUPS_LOG loaded from persistent DB above

def log_booking(user_id: str, service: str, amount: float, destination: str = ""):
    BOOKINGS_LOG.append({
        "booking_id": "BK_" + uuid.uuid4().hex[:8].upper(),
        "user_id": user_id,
        "service": service,
        "amount": amount,
        "destination": destination,
        "timestamp": datetime.now().isoformat(),
        "month": datetime.now().strftime("%Y-%m"),
    })

def log_signup(user_id: str, user_name: str, source: str = "web"):
    SIGNUPS_LOG.append({
        "user_id": user_id,
        "user_name": user_name,
        "source": source,
        "timestamp": datetime.now().isoformat(),
        "month": datetime.now().strftime("%Y-%m"),
    })

@app.get("/api/admin/analytics")
async def admin_analytics():
    """Full analytics dashboard data."""
    now = datetime.now()
    this_month = now.strftime("%Y-%m")
    last_month = (now.replace(day=1) - timedelta(days=1)).strftime("%Y-%m")

    # Wallet stats
    total_loaded = sum(w.get("total_loaded",0) for w in WALLETS.values())
    total_spent  = sum(w.get("total_spent",0) for w in WALLETS.values())
    coins_live   = sum(w.get("coins",0) for w in WALLETS.values())
    partner_coins= sum(p.get("coins",0) for p in PARTNER_WALLETS.values())

    # Booking stats
    this_month_bookings = [b for b in BOOKINGS_LOG if b.get("month")==this_month]
    this_month_revenue  = sum(b.get("amount",0) for b in this_month_bookings)

    # Service breakdown
    service_counts: Dict[str,int] = {}
    service_revenue: Dict[str,float] = {}
    for b in BOOKINGS_LOG:
        svc = b.get("service","other")
        service_counts[svc] = service_counts.get(svc,0) + 1
        service_revenue[svc] = service_revenue.get(svc,0.0) + b.get("amount",0)

    # Monthly revenue trend (last 6 months)
    monthly: Dict[str,float] = {}
    for b in BOOKINGS_LOG:
        m = b.get("month","")
        monthly[m] = monthly.get(m,0.0) + b.get("amount",0)

    # Transaction breakdown
    credits = [t for t in TRANSACTIONS if t.get("type")=="credit"]
    debits  = [t for t in TRANSACTIONS if t.get("type")=="debit"]

    # Top destinations
    dest_counts: Dict[str,int] = {}
    for b in BOOKINGS_LOG:
        d = b.get("destination","")
        if d: dest_counts[d] = dest_counts.get(d,0) + 1
    top_destinations = sorted(dest_counts.items(), key=lambda x: x[1], reverse=True)[:5]

    return {
        "overview": {
            "total_users": len(WALLETS),
            "total_partners": len(PARTNER_WALLETS),
            "total_transactions": len(TRANSACTIONS),
            "total_bookings": len(BOOKINGS_LOG),
            "total_signups": len(SIGNUPS_LOG),
        },
        "wallet": {
            "total_loaded_inr": total_loaded,
            "total_spent_inr": total_spent,
            "coins_with_travellers": coins_live,
            "coins_with_partners": partner_coins,
            "total_reconciled": sum(r.get("amount_inr",0) for r in RECONCILIATIONS),
        },
        "revenue": {
            "this_month": this_month_revenue,
            "total_all_time": sum(b.get("amount",0) for b in BOOKINGS_LOG),
            "this_month_bookings": len(this_month_bookings),
            "monthly_trend": monthly,
        },
        "services": {
            "counts": service_counts,
            "revenue": service_revenue,
        },
        "top_destinations": top_destinations,
        "recent_transactions": list(reversed(TRANSACTIONS))[:10],
        "recent_signups": list(reversed(SIGNUPS_LOG))[:10],
        "partner_payouts": RECONCILIATIONS[-5:],
    }

@app.get("/api/admin/users")
async def admin_users():
    """List all users with wallet balances."""
    users = []
    for uid, w in WALLETS.items():
        users.append({
            "user_id": uid,
            "user_name": w.get("user_name",""),
            "coins": w.get("coins",0),
            "total_loaded": w.get("total_loaded",0),
            "total_spent": w.get("total_spent",0),
            "joined": w.get("created_at",""),
        })
    return {"users": sorted(users, key=lambda x: x["total_loaded"], reverse=True)}

@app.get("/api/admin/partners")
async def admin_partners():
    """List all partners with earnings."""
    partners = []
    for pid, pw in PARTNER_WALLETS.items():
        partners.append({
            "partner_id": pid,
            "partner_name": pw.get("partner_name",""),
            "coins": pw.get("coins",0),
            "total_earned": pw.get("total_earned",0),
            "pending_payout": pw.get("pending_payout",0),
            "total_paid_out": pw.get("total_paid_out",0),
        })
    return {"partners": sorted(partners, key=lambda x: x["total_earned"], reverse=True)}


# ══════════════════════════════════════════════════════
#   🏪 ENHANCED PARTNER ONBOARDING WITH AI
# ══════════════════════════════════════════════════════

class PartnerAIRequest(BaseModel):
    message: str
    partner_id: str = ""
    conversation: List[Dict] = []

@app.post("/api/partner/ai-onboard")
async def partner_ai_onboard(req: PartnerAIRequest):
    """AI-assisted partner onboarding conversation."""
    messages = req.conversation + [{"role":"user","content":req.message}]
    if client:
        try:
            resp = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=1000,
                system=PARTNER_ONBOARD_PROMPT,
                messages=messages,
            )
            return {"response": resp.content[0].text, "mock": False}
        except Exception as e:
            return {"response": "Welcome to TTT Partners! Tell me about your property or service and I'll help you get listed.", "mock": True}
    return {"response": "Welcome to TTT Partners! Tell me about your property or service.", "mock": True}

@app.post("/api/partner/ai-describe")
async def partner_ai_describe(req: PartnerAIRequest):
    """Use AI to generate a listing description for the partner."""
    prompt = f"Based on this partner info, write a compelling TTT marketplace listing: {req.message}\n\nReturn JSON with: title, description, highlights (array of 3), tag"
    if client:
        try:
            resp = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=600,
                system="You are TTT's content writer. Create beautiful, luxury-feeling listing descriptions.",
                messages=[{"role":"user","content":prompt}],
            )
            text = resp.content[0].text
            try:
                import re
                json_match = re.search(r'\{.*\}', text, re.DOTALL)
                if json_match:
                    return {"listing": json.loads(json_match.group()), "mock": False}
            except:
                pass
            return {"listing": {"title":"TTT Partner Property","description":text}, "mock": False}
        except Exception as e:
            pass
    return {"listing": {"title":"Partner Listing","description":"A beautiful property on the TTT network."}, "mock": True}

# ── Live Visitor Heartbeat ───────────────────────────────────────────────────
class HeartbeatRequest(BaseModel):
    session_id: str
    page: str = '/'
    user_id: str = 'anon'
    name: str = ''
    email: str = ''
    phone: str = ''

@app.post("/api/visitor/heartbeat")
async def visitor_heartbeat(req: HeartbeatRequest):
    _live_visitors[req.session_id] = {
        'page':      req.page,
        'last_seen': _time.time(),
        'user_id':   req.user_id,
        'name':      req.name or None,
        'email':     req.email or None,
        'phone':     req.phone or None,
    }
    # Clean up expired
    cutoff = _time.time() - 300
    for sid in list(_live_visitors.keys()):
        if _live_visitors[sid]['last_seen'] < cutoff:
            del _live_visitors[sid]
    return {"live": get_live_count()}

@app.get("/api/visitor/count")
async def visitor_count():
    return {"live": get_live_count(), "sessions": len(_live_visitors)}

# ─────────────────────────────────────────────────────────────────
# ADMIN — Full CEO Dashboard Endpoints
# ─────────────────────────────────────────────────────────────────

@app.get("/api/admin/activity")
async def admin_activity(key: str = Query(""), limit: int = 200):
    if key != ADMIN_KEY:
        raise HTTPException(401, "Invalid admin key")
    return {"activity": list(reversed(_activity_log[-limit:]))}

@app.get("/api/admin/chat-log")
async def admin_chat_log(key: str = Query(""), limit: int = 200):
    if key != ADMIN_KEY:
        raise HTTPException(401, "Invalid admin key")
    return {"chats": list(reversed(_chat_log[-limit:]))}

@app.get("/api/admin/logins")
async def admin_logins(key: str = Query(""), limit: int = 200):
    if key != ADMIN_KEY:
        raise HTTPException(401, "Invalid admin key")
    return {"logins": list(reversed(_login_log[-limit:]))}

@app.get("/api/admin/partner-detail/{partner_id}")
async def admin_partner_detail(partner_id: str, key: str = Query("")):
    if key != ADMIN_KEY:
        raise HTTPException(401, "Invalid admin key")
    partner = _partners.get(partner_id)
    if not partner:
        raise HTTPException(404, "Partner not found")
    listings = [l for l in _listings.values() if l.get("partner_id") == partner_id]
    bookings = [b for b in _bookings.values() if b.get("partner_id") == partner_id]
    wallet = PARTNER_WALLETS.get(partner_id, {})
    return {
        "partner":  partner,
        "listings": listings,
        "bookings": bookings,
        "wallet":   wallet,
        "listing_count": len(listings),
        "booking_count": len(bookings),
    }

@app.get("/api/admin/traveller-detail/{user_id}")
async def admin_traveller_detail(user_id: str, key: str = Query("")):
    if key != ADMIN_KEY:
        raise HTTPException(401, "Invalid admin key")
    user = _users.get(user_id)
    if not user:
        raise HTTPException(404, "User not found")
    wallet = WALLETS.get(user_id, {})
    activity = [a for a in _activity_log if a.get("user_id") == user_id]
    chats = [c for c in _chat_log if c.get("user_id") == user_id]
    return {
        "user":     user,
        "wallet":   wallet,
        "activity": list(reversed(activity[-50:])),
        "chats":    list(reversed(chats[-50:])),
        "activity_count": len(activity),
        "chat_count": len(chats),
    }

@app.put("/api/admin/partner/{partner_id}")
async def admin_update_partner(partner_id: str, updates: dict, key: str = Query("")):
    if key != ADMIN_KEY:
        raise HTTPException(401, "Invalid admin key")
    if partner_id not in _partners:
        raise HTTPException(404, "Partner not found")
    _partners[partner_id].update(updates)
    _partners[partner_id]["updated_at"] = datetime.now().isoformat()
    _save_db()
    return {"success": True, "partner": _partners[partner_id]}

@app.put("/api/admin/listing/{listing_id}")
async def admin_update_listing(listing_id: str, updates: dict, key: str = Query("")):
    if key != ADMIN_KEY:
        raise HTTPException(401, "Invalid admin key")
    if listing_id not in _listings:
        raise HTTPException(404, "Listing not found")
    _listings[listing_id].update(updates)
    _listings[listing_id]["updated_at"] = datetime.now().isoformat()
    _save_db()
    return {"success": True, "listing": _listings[listing_id]}

@app.delete("/api/admin/listing/{listing_id}")
async def admin_delete_listing(listing_id: str, key: str = Query("")):
    if key != ADMIN_KEY:
        raise HTTPException(401, "Invalid admin key")
    if listing_id not in _listings:
        raise HTTPException(404, "Listing not found")
    del _listings[listing_id]
    _save_db()
    return {"success": True}

@app.put("/api/admin/user/{user_id}/coins")
async def admin_adjust_coins(user_id: str, amount: int, reason: str = "Admin adjustment", key: str = Query("")):
    if key != ADMIN_KEY:
        raise HTTPException(401, "Invalid admin key")
    if user_id not in WALLETS:
        WALLETS[user_id] = {"coins": 0, "transactions": [], "total_loaded": 0, "total_spent": 0}
    WALLETS[user_id]["coins"] = max(0, WALLETS[user_id].get("coins", 0) + amount)
    WALLETS[user_id].setdefault("transactions", []).append({
        "type": "admin_adjust", "amount": amount, "reason": reason,
        "timestamp": datetime.now().isoformat()
    })
    return {"success": True, "new_balance": WALLETS[user_id]["coins"]}

@app.get("/api/admin/search")
async def admin_search(q: str = "", key: str = Query("")):
    if key != ADMIN_KEY:
        raise HTTPException(401, "Invalid admin key")
    q = q.lower().strip()
    if not q:
        return {"travellers": [], "partners": []}
    
    matched_travellers = []
    for uid, u in _users.items():
        if (q in (u.get('email') or '').lower() or
            q in (u.get('phone') or '').lower() or
            q in (u.get('name') or '').lower() or
            q in uid.lower()):
            wallet = WALLETS.get(uid, {})
            matched_travellers.append({**u, "coins": wallet.get("coins", 0)})
    
    matched_partners = []
    for pid, p in _partners.items():
        if (q in (p.get('email') or '').lower() or
            q in (p.get('phone') or '').lower() or
            q in (p.get('business_name') or '').lower() or
            q in (p.get('name') or '').lower() or
            q in pid.lower()):
            matched_partners.append(p)
    
    return {"travellers": matched_travellers[:20], "partners": matched_partners[:20]}

@app.post("/api/admin/impersonate")
async def admin_impersonate(user_id: str, key: str = Query("")):
    """Admin login as any traveller — returns user session data directly."""
    if key != ADMIN_KEY:
        raise HTTPException(401, "Invalid admin key")
    user = _users.get(user_id)
    if not user:
        raise HTTPException(404, "User not found")
    wallet = WALLETS.get(user_id, {})
    log_activity("admin", "impersonate", {"target_user": user_id})
    # Build a session object matching what the frontend expects in sessionStorage
    session = {
        "id":    user_id,
        "name":  user.get("name") or user.get("email") or user_id,
        "email": user.get("email"),
        "phone": user.get("phone"),
        "persona": user.get("travel_persona") or user.get("persona"),
        "instagram": user.get("instagram_handle"),
        "coins": wallet.get("coins", 0),
        "impersonated_by": "admin",
    }
    return {
        "success": True,
        "session": session,
        "user": user,
        "message": f"Logged in as {user.get('name') or user.get('email') or user_id}"
    }

@app.post("/api/admin/impersonate-partner")
async def admin_impersonate_partner(partner_id: str, key: str = Query("")):
    """Login as a partner from the admin side."""
    if key != ADMIN_KEY:
        raise HTTPException(401, "Invalid admin key")
    partner = _partners.get(partner_id)
    if not partner:
        raise HTTPException(404, "Partner not found")
    log_activity("admin", "impersonate_partner", {"target_partner": partner_id})
    return {
        "success": True,
        "partner_id": partner_id,
        "partner": partner,
        "message": f"Logged in as partner: {partner.get('business_name') or partner_id}"
    }

@app.get("/api/admin/live-visitors")
async def admin_live_visitors(key: str = Query("")):
    if key != ADMIN_KEY:
        raise HTTPException(401, "Invalid admin key")
    cutoff = _time.time() - 300  # last 5 minutes
    active = [
        {**v, "last_seen_ago": round(_time.time() - v["last_seen"])}
        for v in _live_visitors.values()
        if v["last_seen"] > cutoff
    ]
    return {
        "count": len(active),
        "visitors": active,
        "debug_total_sessions": len(_live_visitors),
    }

@app.get("/api/admin/full-summary")
async def admin_full_summary(key: str = Query("")):
    if key != ADMIN_KEY:
        raise HTTPException(401, "Invalid admin key")
    total_coins = sum(w.get("coins", 0) for w in WALLETS.values())
    # Count real listings (exclude demo)
    real_listings = {k:v for k,v in _listings.items() if v.get("partner_id") != "demo-partner-001"}
    real_partners = {k:v for k,v in _partners.items() if k != "demo-partner-001"}
    return {
        "totals": {
            "travellers":   len(_users),
            "partners":     len(real_partners),
            "listings":     len(real_listings),
            "bookings":     len(_bookings) + len(BOOKINGS_LOG),
            "total_signups": len(SIGNUPS_LOG),
            "total_coins_in_circulation": total_coins,
            "chat_messages": len(_chat_log),
            "total_logins":  len(_login_log),
            "total_activity": len(_activity_log),
        },
        "recent_signups":   list(reversed(SIGNUPS_LOG[-10:])),
        "recent_logins":    list(reversed(_login_log[-10:])),
        "recent_activity":  list(reversed(_activity_log[-20:])),
        "recent_chats":     list(reversed(_chat_log[-10:])),
    }



# ═══════════════════════════════════════════════════════════════════════════
# TTT CRM — Lead Management, Activity Tracking, WhatsApp & Instagram API
# ═══════════════════════════════════════════════════════════════════════════

import urllib.parse

# ── CRM in-memory store (persisted in _db) ──────────────────────────────────
_crm_activities: List[dict] = []          # all tracked activities
_crm_notes: Dict[str, List[dict]] = {}    # user_id → [notes]
_crm_tags:  Dict[str, List[str]]  = {}    # user_id → [tags]
_crm_extra: Dict[str, dict]       = {}    # user_id → {linkedin, instagram, whatsapp, notes}

# ── Score calculator ─────────────────────────────────────────────────────────
def _calc_score(user_id: str) -> dict:
    acts = [a for a in _activity_log if a.get("user_id") == user_id]
    visits   = sum(1 for a in acts if a.get("action") == "visit")
    chats    = sum(1 for a in acts if a.get("action") == "chat")
    trips    = sum(1 for a in acts if a.get("action") in ("trip", "itinerary"))
    bookings = sum(1 for a in acts if a.get("action") == "booking")
    searches = sum(1 for a in acts if a.get("action") == "search")
    logins   = sum(1 for a in acts if a.get("action") in ("login", "signup"))
    score = min(100, int(visits*1.5 + chats*4 + trips*6 + bookings*20 + searches*2 + logins*3))
    return {
        "score": score,
        "visits": visits,
        "ai_chats": chats,
        "trips_planned": trips,
        "bookings": bookings,
        "searches": searches,
        "logins": logins,
    }

def _get_tier(score: int) -> str:
    if score >= 75: return "hot"
    if score >= 40: return "warm"
    return "cold"

def _build_lead(uid: str, user: dict) -> dict:
    # Use persisted score from _db if available (survives redeploys)
    _persisted = _db.get("lead_scores", {}).get(uid, {})
    stats = _calc_score(uid)
    # Merge persisted counts (they accumulate across sessions)
    for k in ("visits","ai_chats","trips_planned","bookings","searches","logins"):
        if _persisted.get(k, 0) > stats.get(k, 0):
            stats[k] = _persisted[k]
    # Recalculate score with merged data
    stats["score"] = min(100, int(stats["visits"]*1.5 + stats["ai_chats"]*4 + stats["trips_planned"]*6 + stats["bookings"]*20 + stats["searches"]*2 + stats["logins"]*3))
    score = stats["score"]
    extra = _crm_extra.get(uid, {})
    acts  = sorted(
        [a for a in _activity_log if a.get("user_id") == uid],
        key=lambda x: x.get("timestamp",""), reverse=True
    )
    ig_raw = user.get("instagram_handle") or extra.get("instagram_handle") or ""
    ig_handle = ig_raw.lstrip("@") if ig_raw else ""
    phone = user.get("phone") or extra.get("phone") or ""
    wa_number = phone.replace("+","").replace(" ","").replace("-","")

    return {
        "user_id":          uid,
        "full_name":        user.get("name") or extra.get("full_name") or "",
        "email":            user.get("email") or "",
        "phone":            phone,
        "whatsapp_number":  wa_number,
        "linkedin_url":     user.get("linkedin_url") or extra.get("linkedin_url") or "",
        "instagram_handle": ig_handle,
        "instagram_url":    f"https://instagram.com/{ig_handle}" if ig_handle else "",
        "source":           user.get("source") or extra.get("source") or "organic",
        "score":            score,
        "tier":             _get_tier(score),
        "notes":            extra.get("notes",""),
        "tags":             _crm_tags.get(uid, []),
        "first_seen":       user.get("created_at",""),
        "last_activity":    acts[0].get("timestamp","") if acts else user.get("created_at",""),
        **stats,
        "recent_activities": acts[:15],
    }


# ── 1. CRM: Get all leads ────────────────────────────────────────────────────
@app.get("/api/crm/leads")
async def crm_leads(
    admin_key: str = Query(""),
    tier:      str = Query(""),
    sort:      str = Query("score"),
    search:    str = Query(""),
    limit:     int = Query(200),
):
    if admin_key != ADMIN_KEY:
        raise HTTPException(status_code=403, detail="Forbidden")

    leads = [_build_lead(uid, u) for uid, u in _users.items()]

    if search:
        s = search.lower()
        leads = [l for l in leads if
                 s in (l["full_name"] or "").lower() or
                 s in (l["email"] or "").lower() or
                 s in (l["phone"] or "")]

    if tier in ("hot","warm","cold"):
        leads = [l for l in leads if l["tier"] == tier]

    if sort == "score":    leads.sort(key=lambda x: x["score"], reverse=True)
    elif sort == "recent": leads.sort(key=lambda x: x["last_activity"], reverse=True)
    elif sort == "activity":leads.sort(key=lambda x: x["ai_chats"]+x["visits"], reverse=True)
    elif sort == "name":   leads.sort(key=lambda x: (x["full_name"] or x["email"] or "").lower())

    leads = leads[:limit]
    return {
        "leads": leads,
        "total": len(leads),
        "hot":   sum(1 for l in leads if l["tier"]=="hot"),
        "warm":  sum(1 for l in leads if l["tier"]=="warm"),
        "cold":  sum(1 for l in leads if l["tier"]=="cold"),
    }


# ── 2. CRM: Single lead detail ───────────────────────────────────────────────
@app.get("/api/crm/lead/{user_id}")
async def crm_lead_detail(user_id: str, admin_key: str = Query("")):
    if admin_key != ADMIN_KEY:
        raise HTTPException(status_code=403, detail="Forbidden")
    user = _users.get(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="Lead not found")
    return _build_lead(user_id, user)


# ── 3. CRM: Update lead (phone, LinkedIn, Instagram, notes, tags) ────────────
@app.patch("/api/crm/lead/{user_id}")
async def crm_update_lead(user_id: str, data: dict, admin_key: str = Query("")):
    if admin_key != ADMIN_KEY:
        raise HTTPException(status_code=403, detail="Forbidden")
    if user_id not in _users:
        raise HTTPException(status_code=404, detail="Lead not found")
    # Update user record
    for field in ("name","email","phone","linkedin_url","instagram_handle"):
        if field in data:
            _users[user_id][field] = data[field]
    # Update extra CRM fields
    if user_id not in _crm_extra:
        _crm_extra[user_id] = {}
    for field in ("notes","source","full_name","linkedin_url","instagram_handle","phone"):
        if field in data:
            _crm_extra[user_id][field] = data[field]
    if "tags" in data:
        _crm_tags[user_id] = data["tags"]
    _save_db()
    return {"status": "updated", "lead": _build_lead(user_id, _users[user_id])}


# ── 4. CRM: Log activity (called by frontend tracker) ───────────────────────
@app.post("/api/crm/activity")
async def crm_log_activity(data: dict):
    user_id  = data.get("user_id", "anon")
    act_type = data.get("activity_type", "visit")
    log_activity(user_id, act_type, data.get("activity_data", {}))
    # Sync profile if provided
    if "profile" in data and user_id in _users:
        for f in ("name","phone","instagram_handle","linkedin_url"):
            if data["profile"].get(f):
                _users[user_id][f] = data["profile"][f]
        _save_db()
    return {"status": "ok"}


# ── 5. CRM: Metrics summary ──────────────────────────────────────────────────
@app.get("/api/crm/metrics")
async def crm_metrics(admin_key: str = Query("")):
    if admin_key != ADMIN_KEY:
        raise HTTPException(status_code=403, detail="Forbidden")
    leads = [_build_lead(uid, u) for uid, u in _users.items()]
    total_bookings = sum(l["bookings"] for l in leads)
    avg_score = int(sum(l["score"] for l in leads)/len(leads)) if leads else 0
    return {
        "total_leads":    len(leads),
        "hot":            sum(1 for l in leads if l["tier"]=="hot"),
        "warm":           sum(1 for l in leads if l["tier"]=="warm"),
        "cold":           sum(1 for l in leads if l["tier"]=="cold"),
        "total_bookings": total_bookings,
        "avg_score":      avg_score,
        "new_this_week":  len([l for l in leads if l["first_seen"] > (datetime.now().isoformat()[:10])]),
        "live_now":       get_live_count(),
    }


# ── 6. CRM: Sync profile on login/signup ────────────────────────────────────
@app.post("/api/crm/sync-profile")
async def crm_sync_profile(data: dict):
    uid = data.get("user_id") or data.get("email")
    if not uid: return {"status": "skipped"}
    # Find matching user
    target = None
    for u_id, u in _users.items():
        if u.get("email") == uid or u_id == uid:
            target = u_id; break
    if target:
        for f in ("name","phone","instagram_handle","linkedin_url","source"):
            if data.get(f): _users[target][f] = data[f]
        _save_db()
    return {"status": "synced"}


# ── 7. WhatsApp API — generate link & send ───────────────────────────────────
@app.get("/api/crm/whatsapp/link")
async def whatsapp_link(
    admin_key: str = Query(""),
    user_id:   str = Query(""),
    message:   str = Query(""),
):
    """Returns a click-to-open WhatsApp link for a lead."""
    if admin_key != ADMIN_KEY:
        raise HTTPException(status_code=403, detail="Forbidden")

    user  = _users.get(user_id, {})
    extra = _crm_extra.get(user_id, {})
    phone = user.get("phone") or extra.get("phone") or ""
    wa_number = phone.replace("+","").replace(" ","").replace("-","")

    if not wa_number:
        raise HTTPException(status_code=400, detail="No phone number for this lead")

    name = user.get("name") or "Traveller"
    if not message:
        message = (
            f"Hi {name}! 👋\n\n"
            "I'm reaching out from *The Trip Theory* — India's first AI Travel Concierge.\n\n"
            "We noticed you've been exploring some amazing destinations on our platform. "
            "We'd love to help you plan your perfect trip!\n\n"
            "✈️ Visit: thetriptheory.com\n\n"
            "Where would you like to go next? 🌏"
        )

    encoded = urllib.parse.quote(message)
    wa_link = f"https://wa.me/{wa_number}?text={encoded}"
    return {
        "whatsapp_link": wa_link,
        "phone":         phone,
        "message":       message,
        "lead_name":     name,
    }


@app.post("/api/crm/whatsapp/bulk")
async def whatsapp_bulk(data: dict, admin_key: str = Query("")):
    """Generate WhatsApp links for multiple leads at once."""
    if admin_key != ADMIN_KEY:
        raise HTTPException(status_code=403, detail="Forbidden")

    user_ids = data.get("user_ids", [])
    message_template = data.get("message", "")
    results = []

    for uid in user_ids:
        user  = _users.get(uid, {})
        extra = _crm_extra.get(uid, {})
        phone = user.get("phone") or extra.get("phone") or ""
        if not phone:
            results.append({"user_id": uid, "status": "no_phone"})
            continue
        wa_number = phone.replace("+","").replace(" ","").replace("-","")
        name = user.get("name") or "Traveller"
        msg = message_template.replace("{{name}}", name) if message_template else (
            f"Hi {name}! 👋 Your next adventure awaits — plan it with India's first AI Travel Concierge at thetriptheory.com ✈️"
        )
        encoded = urllib.parse.quote(msg)
        results.append({
            "user_id":       uid,
            "name":          name,
            "phone":         phone,
            "whatsapp_link": f"https://wa.me/{wa_number}?text={encoded}",
            "status":        "ready",
        })

    return {"results": results, "total": len(results), "ready": sum(1 for r in results if r["status"]=="ready")}


# ── 8. Instagram API — profile link & data ──────────────────────────────────
@app.get("/api/crm/instagram/link")
async def instagram_link(admin_key: str = Query(""), user_id: str = Query("")):
    """Returns the Instagram profile URL for a lead."""
    if admin_key != ADMIN_KEY:
        raise HTTPException(status_code=403, detail="Forbidden")
    user  = _users.get(user_id, {})
    extra = _crm_extra.get(user_id, {})
    handle = (user.get("instagram_handle") or extra.get("instagram_handle") or "").lstrip("@")
    if not handle:
        raise HTTPException(status_code=400, detail="No Instagram handle for this lead")
    return {
        "instagram_handle": handle,
        "instagram_url":    f"https://instagram.com/{handle}",
        "instagram_app_url":f"instagram://user?username={handle}",
    }


# ── 9. CRM: Add note to a lead ───────────────────────────────────────────────
@app.post("/api/crm/lead/{user_id}/note")
async def crm_add_note(user_id: str, data: dict, admin_key: str = Query("")):
    if admin_key != ADMIN_KEY:
        raise HTTPException(status_code=403, detail="Forbidden")
    if user_id not in _crm_notes:
        _crm_notes[user_id] = []
    note = {
        "id":        str(uuid.uuid4())[:8],
        "text":      data.get("text",""),
        "author":    data.get("author","Admin"),
        "created_at": datetime.now().isoformat(),
    }
    _crm_notes[user_id].append(note)
    return {"status": "added", "note": note}


# ── 10. CRM Dashboard page ──────────────────────────────────────────────────

@app.get("/landing")
async def landing_page():
    """Landing page v6 — base64 encoded to avoid escaping issues."""
    import base64 as _b64
    _b64_html = "PCFET0NUWVBFIGh0bWw+CjxodG1sIGxhbmc9ImVuIj4KPGhlYWQ+CjxtZXRhIGNoYXJzZXQ9IlVURi04Ij4KPG1ldGEgbmFtZT0idmlld3BvcnQiIGNvbnRlbnQ9IndpZHRoPWRldmljZS13aWR0aCwgaW5pdGlhbC1zY2FsZT0xLjAiPgo8dGl0bGU+VGhlIFRyaXAgVGhlb3J5IOKAlCBJbmRpYSdzIEZpcnN0IEFJIFRyYXZlbCBDb25jaWVyZ2U8L3RpdGxlPgo8bGluayByZWw9InByZWNvbm5lY3QiIGhyZWY9Imh0dHBzOi8vZm9udHMuZ29vZ2xlYXBpcy5jb20iPgo8bGluayBocmVmPSJodHRwczovL2ZvbnRzLmdvb2dsZWFwaXMuY29tL2NzczI/ZmFtaWx5PUNvcm1vcmFudCtHYXJhbW9uZDppdGFsLHdnaHRAMCwzMDA7MCw0MDA7MCw2MDA7MSwzMDA7MSw0MDA7MSw2MDAmZmFtaWx5PU1vbnRzZXJyYXQ6d2dodEAyMDA7MzAwOzQwMDs1MDAmZGlzcGxheT1zd2FwIiByZWw9InN0eWxlc2hlZXQiPgo8c3R5bGU+Cjpyb290IHsKICAtLWdvbGQ6ICNDOTk2M0E7CiAgLS1nb2xkLWxpZ2h0OiAjRThDODc4OwogIC0tZ29sZC1wYWxlOiAjRjVFREQ2OwogIC0tYmxhY2s6ICMwQTA4MDU7CiAgLS1vZmYtYmxhY2s6ICMxQTE0MTA7CiAgLS1kYXJrOiAjMjUxRTE2OwogIC0td2FybS1ncmV5OiAjN0E3MDY4OwogIC0tY3JlYW06ICNGOEYzRUM7CiAgLS13aGl0ZTogI0ZFRkNGODsKICAtLXNlcmlmOiAnQ29ybW9yYW50IEdhcmFtb25kJywgR2VvcmdpYSwgc2VyaWY7CiAgLS1zYW5zOiAnTW9udHNlcnJhdCcsIHNhbnMtc2VyaWY7Cn0KCiosICo6OmJlZm9yZSwgKjo6YWZ0ZXIgeyBib3gtc2l6aW5nOiBib3JkZXItYm94OyBtYXJnaW46IDA7IHBhZGRpbmc6IDA7IH0KCmh0bWwgeyBzY3JvbGwtYmVoYXZpb3I6IHNtb290aDsgfQoKYm9keSB7CiAgZm9udC1mYW1pbHk6IHZhcigtLXNhbnMpOwogIGJhY2tncm91bmQ6IHZhcigtLWJsYWNrKTsKICBjb2xvcjogdmFyKC0td2hpdGUpOwogIG92ZXJmbG93LXg6IGhpZGRlbjsKICBjdXJzb3I6IG5vbmU7Cn0KCi8qIEN1c3RvbSBjdXJzb3IgKi8KLmN1cnNvciB7CiAgd2lkdGg6IDhweDsgaGVpZ2h0OiA4cHg7CiAgYmFja2dyb3VuZDogdmFyKC0tZ29sZCk7CiAgYm9yZGVyLXJhZGl1czogNTAlOwogIHBvc2l0aW9uOiBmaXhlZDsKICBwb2ludGVyLWV2ZW50czogbm9uZTsKICB6LWluZGV4OiA5OTk5OwogIHRyYW5zaXRpb246IHRyYW5zZm9ybSAwLjE1cyBlYXNlOwogIG1peC1ibGVuZC1tb2RlOiBkaWZmZXJlbmNlOwp9Ci5jdXJzb3ItcmluZyB7CiAgd2lkdGg6IDM2cHg7IGhlaWdodDogMzZweDsKICBib3JkZXI6IDFweCBzb2xpZCByZ2JhKDIwMSwxNTAsNTgsMC41KTsKICBib3JkZXItcmFkaXVzOiA1MCU7CiAgcG9zaXRpb246IGZpeGVkOwogIHBvaW50ZXItZXZlbnRzOiBub25lOwogIHotaW5kZXg6IDk5OTg7CiAgdHJhbnNpdGlvbjogYWxsIDAuMTJzIGVhc2U7Cn0KYm9keTpob3ZlciAuY3Vyc29yIHsgdHJhbnNmb3JtOiBzY2FsZSgxKTsgfQphOmhvdmVyIH4gLmN1cnNvciwgYnV0dG9uOmhvdmVyIH4gLmN1cnNvciB7IHRyYW5zZm9ybTogc2NhbGUoMyk7IH0KCi8qIOKUgOKUgCBOQVYg4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSAICovCm5hdiB7CiAgcG9zaXRpb246IGZpeGVkOyB0b3A6IDA7IGxlZnQ6IDA7IHJpZ2h0OiAwOyB6LWluZGV4OiAxMDA7CiAgZGlzcGxheTogZmxleDsgYWxpZ24taXRlbXM6IGNlbnRlcjsganVzdGlmeS1jb250ZW50OiBzcGFjZS1iZXR3ZWVuOwogIHBhZGRpbmc6IDI4cHggNjBweDsKICBiYWNrZ3JvdW5kOiBsaW5lYXItZ3JhZGllbnQodG8gYm90dG9tLCByZ2JhKDEwLDgsNSwwLjk1KSAwJSwgdHJhbnNwYXJlbnQgMTAwJSk7CiAgYmFja2Ryb3AtZmlsdGVyOiBibHVyKDJweCk7Cn0KLm5hdi1sb2dvIHsKICBmb250LWZhbWlseTogdmFyKC0tc2VyaWYpOwogIGZvbnQtc2l6ZTogMS40cmVtOwogIGZvbnQtd2VpZ2h0OiAzMDA7CiAgbGV0dGVyLXNwYWNpbmc6IDAuMjVlbTsKICBjb2xvcjogdmFyKC0tZ29sZCk7CiAgdGV4dC1kZWNvcmF0aW9uOiBub25lOwogIHRleHQtdHJhbnNmb3JtOiB1cHBlcmNhc2U7Cn0KLm5hdi1saW5rcyB7CiAgZGlzcGxheTogZmxleDsgZ2FwOiA0MHB4OyBsaXN0LXN0eWxlOiBub25lOwp9Ci5uYXYtbGlua3MgYSB7CiAgZm9udC1mYW1pbHk6IHZhcigtLXNhbnMpOwogIGZvbnQtc2l6ZTogMC42NXJlbTsKICBmb250LXdlaWdodDogMzAwOwogIGxldHRlci1zcGFjaW5nOiAwLjJlbTsKICB0ZXh0LXRyYW5zZm9ybTogdXBwZXJjYXNlOwogIGNvbG9yOiByZ2JhKDI1NSwyNTUsMjU1LDAuNik7CiAgdGV4dC1kZWNvcmF0aW9uOiBub25lOwogIHRyYW5zaXRpb246IGNvbG9yIDAuM3M7Cn0KLm5hdi1saW5rcyBhOmhvdmVyIHsgY29sb3I6IHZhcigtLWdvbGQpOyB9Ci5uYXYtY3RhIHsKICBmb250LWZhbWlseTogdmFyKC0tc2Fucyk7CiAgZm9udC1zaXplOiAwLjYycmVtOwogIGZvbnQtd2VpZ2h0OiA0MDA7CiAgbGV0dGVyLXNwYWNpbmc6IDAuMmVtOwogIHRleHQtdHJhbnNmb3JtOiB1cHBlcmNhc2U7CiAgY29sb3I6IHZhcigtLWJsYWNrKTsKICBiYWNrZ3JvdW5kOiB2YXIoLS1nb2xkKTsKICBib3JkZXI6IG5vbmU7CiAgcGFkZGluZzogMTBweCAyNHB4OwogIGN1cnNvcjogbm9uZTsKICB0cmFuc2l0aW9uOiBiYWNrZ3JvdW5kIDAuM3MsIHRyYW5zZm9ybSAwLjJzOwogIHRleHQtZGVjb3JhdGlvbjogbm9uZTsKfQoubmF2LWN0YTpob3ZlciB7IGJhY2tncm91bmQ6IHZhcigtLWdvbGQtbGlnaHQpOyB0cmFuc2Zvcm06IHRyYW5zbGF0ZVkoLTFweCk7IH0KCi8qIOKUgOKUgCBIRVJPIOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgCAqLwouaGVybyB7CiAgcG9zaXRpb246IHJlbGF0aXZlOwogIGhlaWdodDogMTAwdmg7CiAgbWluLWhlaWdodDogNzAwcHg7CiAgZGlzcGxheTogZmxleDsKICBhbGlnbi1pdGVtczogZmxleC1lbmQ7CiAgcGFkZGluZzogMCA2MHB4IDgwcHg7CiAgb3ZlcmZsb3c6IGhpZGRlbjsKfQouaGVyby1iZyB7CiAgcG9zaXRpb246IGFic29sdXRlOyBpbnNldDogMDsKICBiYWNrZ3JvdW5kOiAKICAgIGxpbmVhci1ncmFkaWVudCgxNjBkZWcsIHJnYmEoMTAsOCw1LDAuMykgMCUsIHJnYmEoMTAsOCw1LDAuNykgNjAlLCByZ2JhKDEwLDgsNSwwLjk1KSAxMDAlKSwKICAgIHVybCgnaHR0cHM6Ly9pbWFnZXMudW5zcGxhc2guY29tL3Bob3RvLTE1MDY5MDU5MjUzNDYtMjFiZGE0ZDMyZGY0P3c9MTgwMCZxPTgwJykgY2VudGVyL2NvdmVyIG5vLXJlcGVhdDsKICB0cmFuc2Zvcm06IHNjYWxlKDEuMDUpOwogIGFuaW1hdGlvbjogc2xvd1pvb20gMjBzIGVhc2Utb3V0IGZvcndhcmRzOwp9CkBrZXlmcmFtZXMgc2xvd1pvb20gewogIGZyb20geyB0cmFuc2Zvcm06IHNjYWxlKDEuMDUpOyB9CiAgdG8gICB7IHRyYW5zZm9ybTogc2NhbGUoMS4wKTsgfQp9Ci5oZXJvLWdyYWluIHsKICBwb3NpdGlvbjogYWJzb2x1dGU7IGluc2V0OiAwOyBvcGFjaXR5OiAwLjA0OwogIGJhY2tncm91bmQtaW1hZ2U6IHVybCgiZGF0YTppbWFnZS9zdmcreG1sLCUzQ3N2ZyB2aWV3Qm94PScwIDAgMjU2IDI1NicgeG1sbnM9J2h0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnJyUzRSUzQ2ZpbHRlciBpZD0nbm9pc2UnJTNFJTNDZmVUdXJidWxlbmNlIHR5cGU9J2ZyYWN0YWxOb2lzZScgYmFzZUZyZXF1ZW5jeT0nMC45JyBudW1PY3RhdmVzPSc0JyBzdGl0Y2hUaWxlcz0nc3RpdGNoJy8lM0UlM0MvZmlsdGVyJTNFJTNDcmVjdCB3aWR0aD0nMTAwJTI1JyBoZWlnaHQ9JzEwMCUyNScgZmlsdGVyPSd1cmwoJTIzbm9pc2UpJy8lM0UlM0Mvc3ZnJTNFIik7CiAgYmFja2dyb3VuZC1zaXplOiAyMDBweDsKICBwb2ludGVyLWV2ZW50czogbm9uZTsKfQouaGVyby1jb250ZW50IHsKICBwb3NpdGlvbjogcmVsYXRpdmU7IHotaW5kZXg6IDI7CiAgbWF4LXdpZHRoOiA4MjBweDsKICBhbmltYXRpb246IGhlcm9JbiAxLjRzIGN1YmljLWJlemllcigwLjE2LCAxLCAwLjMsIDEpIDAuM3MgYm90aDsKfQpAa2V5ZnJhbWVzIGhlcm9JbiB7CiAgZnJvbSB7IG9wYWNpdHk6IDA7IHRyYW5zZm9ybTogdHJhbnNsYXRlWSg0MHB4KTsgfQogIHRvICAgeyBvcGFjaXR5OiAxOyB0cmFuc2Zvcm06IHRyYW5zbGF0ZVkoMCk7IH0KfQouaGVyby10YWcgewogIGZvbnQtc2l6ZTogMC42cmVtOwogIGZvbnQtd2VpZ2h0OiA0MDA7CiAgbGV0dGVyLXNwYWNpbmc6IDAuM2VtOwogIHRleHQtdHJhbnNmb3JtOiB1cHBlcmNhc2U7CiAgY29sb3I6IHZhcigtLWdvbGQpOwogIG1hcmdpbi1ib3R0b206IDIwcHg7CiAgZGlzcGxheTogZmxleDsgYWxpZ24taXRlbXM6IGNlbnRlcjsgZ2FwOiAxMnB4Owp9Ci5oZXJvLXRhZzo6YmVmb3JlIHsKICBjb250ZW50OiAnJzsKICBkaXNwbGF5OiBibG9jazsKICB3aWR0aDogMzJweDsgaGVpZ2h0OiAxcHg7CiAgYmFja2dyb3VuZDogdmFyKC0tZ29sZCk7Cn0KLmhlcm8taGVhZGxpbmUgewogIGZvbnQtZmFtaWx5OiB2YXIoLS1zZXJpZik7CiAgZm9udC1zaXplOiBjbGFtcCgzLjJyZW0sIDd2dywgNi41cmVtKTsKICBmb250LXdlaWdodDogMzAwOwogIGxpbmUtaGVpZ2h0OiAxLjA1OwogIGxldHRlci1zcGFjaW5nOiAtMC4wMWVtOwogIGNvbG9yOiB2YXIoLS13aGl0ZSk7CiAgbWFyZ2luLWJvdHRvbTogMjhweDsKfQouaGVyby1oZWFkbGluZSBlbSB7CiAgZm9udC1zdHlsZTogaXRhbGljOwogIGNvbG9yOiB2YXIoLS1nb2xkLWxpZ2h0KTsKfQouaGVyby1zdWIgewogIGZvbnQtc2l6ZTogMC43OHJlbTsKICBmb250LXdlaWdodDogMzAwOwogIGxldHRlci1zcGFjaW5nOiAwLjA4ZW07CiAgbGluZS1oZWlnaHQ6IDEuOTsKICBjb2xvcjogcmdiYSgyNTUsMjU1LDI1NSwwLjU1KTsKICBtYXgtd2lkdGg6IDQ4MHB4OwogIG1hcmdpbi1ib3R0b206IDQ4cHg7Cn0KLmhlcm8tYWN0aW9ucyB7CiAgZGlzcGxheTogZmxleDsgYWxpZ24taXRlbXM6IGNlbnRlcjsgZ2FwOiAzMnB4Owp9Ci5idG4tcHJpbWFyeSB7CiAgZm9udC1mYW1pbHk6IHZhcigtLXNhbnMpOwogIGZvbnQtc2l6ZTogMC42MnJlbTsKICBmb250LXdlaWdodDogNDAwOwogIGxldHRlci1zcGFjaW5nOiAwLjIyZW07CiAgdGV4dC10cmFuc2Zvcm06IHVwcGVyY2FzZTsKICBjb2xvcjogdmFyKC0tYmxhY2spOwogIGJhY2tncm91bmQ6IHZhcigtLWdvbGQpOwogIGJvcmRlcjogbm9uZTsKICBwYWRkaW5nOiAxNnB4IDM2cHg7CiAgY3Vyc29yOiBub25lOwogIHRleHQtZGVjb3JhdGlvbjogbm9uZTsKICBkaXNwbGF5OiBpbmxpbmUtYmxvY2s7CiAgdHJhbnNpdGlvbjogYWxsIDAuM3M7CiAgcG9zaXRpb246IHJlbGF0aXZlOwogIG92ZXJmbG93OiBoaWRkZW47Cn0KLmJ0bi1wcmltYXJ5OjphZnRlciB7CiAgY29udGVudDogJyc7CiAgcG9zaXRpb246IGFic29sdXRlOyBpbnNldDogMDsKICBiYWNrZ3JvdW5kOiB2YXIoLS1nb2xkLWxpZ2h0KTsKICB0cmFuc2Zvcm06IHRyYW5zbGF0ZVgoLTEwMSUpOwogIHRyYW5zaXRpb246IHRyYW5zZm9ybSAwLjNzIGN1YmljLWJlemllcigwLjE2LDEsMC4zLDEpOwp9Ci5idG4tcHJpbWFyeTpob3Zlcjo6YWZ0ZXIgeyB0cmFuc2Zvcm06IHRyYW5zbGF0ZVgoMCk7IH0KLmJ0bi1wcmltYXJ5IHNwYW4geyBwb3NpdGlvbjogcmVsYXRpdmU7IHotaW5kZXg6IDE7IH0KLmJ0bi1naG9zdCB7CiAgZm9udC1mYW1pbHk6IHZhcigtLXNhbnMpOwogIGZvbnQtc2l6ZTogMC42MnJlbTsKICBmb250LXdlaWdodDogMzAwOwogIGxldHRlci1zcGFjaW5nOiAwLjIyZW07CiAgdGV4dC10cmFuc2Zvcm06IHVwcGVyY2FzZTsKICBjb2xvcjogcmdiYSgyNTUsMjU1LDI1NSwwLjUpOwogIHRleHQtZGVjb3JhdGlvbjogbm9uZTsKICBib3JkZXItYm90dG9tOiAxcHggc29saWQgcmdiYSgyNTUsMjU1LDI1NSwwLjIpOwogIHBhZGRpbmctYm90dG9tOiAycHg7CiAgdHJhbnNpdGlvbjogYWxsIDAuM3M7Cn0KLmJ0bi1naG9zdDpob3ZlciB7IGNvbG9yOiB2YXIoLS1nb2xkKTsgYm9yZGVyLWNvbG9yOiB2YXIoLS1nb2xkKTsgfQoKLmhlcm8tc2Nyb2xsIHsKICBwb3NpdGlvbjogYWJzb2x1dGU7CiAgYm90dG9tOiA0MHB4OyByaWdodDogNjBweDsKICBkaXNwbGF5OiBmbGV4OyBhbGlnbi1pdGVtczogY2VudGVyOyBnYXA6IDE0cHg7CiAgZm9udC1zaXplOiAwLjU4cmVtOwogIGZvbnQtd2VpZ2h0OiAzMDA7CiAgbGV0dGVyLXNwYWNpbmc6IDAuMjVlbTsKICB0ZXh0LXRyYW5zZm9ybTogdXBwZXJjYXNlOwogIGNvbG9yOiByZ2JhKDI1NSwyNTUsMjU1LDAuMyk7CiAgd3JpdGluZy1tb2RlOiB2ZXJ0aWNhbC1ybDsKICBhbmltYXRpb246IGhlcm9JbiAxLjRzIGN1YmljLWJlemllcigwLjE2LCAxLCAwLjMsIDEpIDAuOHMgYm90aDsKfQouaGVyby1zY3JvbGw6OmFmdGVyIHsKICBjb250ZW50OiAnJzsKICB3aWR0aDogMXB4OyBoZWlnaHQ6IDQ4cHg7CiAgYmFja2dyb3VuZDogbGluZWFyLWdyYWRpZW50KHRvIGJvdHRvbSwgcmdiYSgyMDEsMTUwLDU4LDAuNiksIHRyYW5zcGFyZW50KTsKICBhbmltYXRpb246IHNjcm9sbExpbmUgMnMgZWFzZS1pbi1vdXQgaW5maW5pdGU7Cn0KQGtleWZyYW1lcyBzY3JvbGxMaW5lIHsKICAwJSwgMTAwJSB7IG9wYWNpdHk6IDAuNjsgdHJhbnNmb3JtOiBzY2FsZVkoMSk7IH0KICA1MCUgICAgICAgeyBvcGFjaXR5OiAxOyAgIHRyYW5zZm9ybTogc2NhbGVZKDAuNik7IH0KfQoKLyog4pSA4pSAIE1BUlFVRUUg4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSAICovCi5tYXJxdWVlLXN0cmlwIHsKICBiYWNrZ3JvdW5kOiB2YXIoLS1nb2xkKTsKICBwYWRkaW5nOiAxMnB4IDA7CiAgb3ZlcmZsb3c6IGhpZGRlbjsKICB3aGl0ZS1zcGFjZTogbm93cmFwOwp9Ci5tYXJxdWVlLWlubmVyIHsKICBkaXNwbGF5OiBpbmxpbmUtYmxvY2s7CiAgYW5pbWF0aW9uOiBtYXJxdWVlIDI1cyBsaW5lYXIgaW5maW5pdGU7Cn0KLm1hcnF1ZWUtaXRlbSB7CiAgZGlzcGxheTogaW5saW5lLWJsb2NrOwogIGZvbnQtc2l6ZTogMC42cmVtOwogIGZvbnQtd2VpZ2h0OiA0MDA7CiAgbGV0dGVyLXNwYWNpbmc6IDAuM2VtOwogIHRleHQtdHJhbnNmb3JtOiB1cHBlcmNhc2U7CiAgY29sb3I6IHZhcigtLWJsYWNrKTsKICBwYWRkaW5nOiAwIDQwcHg7Cn0KLm1hcnF1ZWUtaXRlbTo6YmVmb3JlIHsKICBjb250ZW50OiAn4pymJzsKICBtYXJnaW4tcmlnaHQ6IDQwcHg7CiAgb3BhY2l0eTogMC41Owp9CkBrZXlmcmFtZXMgbWFycXVlZSB7CiAgZnJvbSB7IHRyYW5zZm9ybTogdHJhbnNsYXRlWCgwKTsgfQogIHRvICAgeyB0cmFuc2Zvcm06IHRyYW5zbGF0ZVgoLTUwJSk7IH0KfQoKLyog4pSA4pSAIFNFQ1RJT04gQkFTRSDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIAgKi8Kc2VjdGlvbiB7IHBhZGRpbmc6IDEyMHB4IDYwcHg7IH0KLnNlY3Rpb24tbGFiZWwgewogIGZvbnQtc2l6ZTogMC41OHJlbTsKICBmb250LXdlaWdodDogNDAwOwogIGxldHRlci1zcGFjaW5nOiAwLjM1ZW07CiAgdGV4dC10cmFuc2Zvcm06IHVwcGVyY2FzZTsKICBjb2xvcjogdmFyKC0tZ29sZCk7CiAgbWFyZ2luLWJvdHRvbTogMjBweDsKICBkaXNwbGF5OiBmbGV4OyBhbGlnbi1pdGVtczogY2VudGVyOyBnYXA6IDE0cHg7Cn0KLnNlY3Rpb24tbGFiZWw6OmJlZm9yZSB7CiAgY29udGVudDogJyc7CiAgZGlzcGxheTogYmxvY2s7CiAgd2lkdGg6IDI4cHg7IGhlaWdodDogMXB4OwogIGJhY2tncm91bmQ6IHZhcigtLWdvbGQpOwp9Ci5zZWN0aW9uLWhlYWRsaW5lIHsKICBmb250LWZhbWlseTogdmFyKC0tc2VyaWYpOwogIGZvbnQtc2l6ZTogY2xhbXAoMi40cmVtLCA1dncsIDQuNXJlbSk7CiAgZm9udC13ZWlnaHQ6IDMwMDsKICBsaW5lLWhlaWdodDogMS4xOwogIGxldHRlci1zcGFjaW5nOiAtMC4wMWVtOwp9Ci5zZWN0aW9uLWhlYWRsaW5lIGVtIHsgZm9udC1zdHlsZTogaXRhbGljOyBjb2xvcjogdmFyKC0tZ29sZC1saWdodCk7IH0KCi8qIOKUgOKUgCBJTlRSTyAvIE1BTklGRVNUTyDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIAgKi8KLmludHJvIHsKICBiYWNrZ3JvdW5kOiB2YXIoLS1jcmVhbSk7CiAgZGlzcGxheTogZ3JpZDsKICBncmlkLXRlbXBsYXRlLWNvbHVtbnM6IDFmciAxZnI7CiAgZ2FwOiA4MHB4OwogIGFsaWduLWl0ZW1zOiBjZW50ZXI7CiAgcGFkZGluZzogMTIwcHggNjBweDsKfQouaW50cm8tbGVmdCAuc2VjdGlvbi1sYWJlbCB7IGNvbG9yOiB2YXIoLS13YXJtLWdyZXkpOyB9Ci5pbnRyby1sZWZ0IC5zZWN0aW9uLWxhYmVsOjpiZWZvcmUgeyBiYWNrZ3JvdW5kOiB2YXIoLS13YXJtLWdyZXkpOyB9Ci5pbnRyby1sZWZ0IC5zZWN0aW9uLWhlYWRsaW5lIHsgY29sb3I6IHZhcigtLW9mZi1ibGFjayk7IH0KLmludHJvLXJpZ2h0IHsKICBib3JkZXItbGVmdDogMXB4IHNvbGlkIHJnYmEoMTAsOCw1LDAuMSk7CiAgcGFkZGluZy1sZWZ0OiA2MHB4Owp9Ci5pbnRyby1yaWdodCBwIHsKICBmb250LXNpemU6IDAuODhyZW07CiAgZm9udC13ZWlnaHQ6IDMwMDsKICBsaW5lLWhlaWdodDogMjsKICBjb2xvcjogdmFyKC0td2FybS1ncmV5KTsKICBtYXJnaW4tYm90dG9tOiAyMHB4Owp9Ci5pbnRyby1yaWdodCBwIHN0cm9uZyB7CiAgZm9udC13ZWlnaHQ6IDUwMDsKICBjb2xvcjogdmFyKC0tb2ZmLWJsYWNrKTsKfQouaW50cm8tc3RhdCB7CiAgbWFyZ2luLXRvcDogNDhweDsKICBkaXNwbGF5OiBncmlkOwogIGdyaWQtdGVtcGxhdGUtY29sdW1uczogMWZyIDFmciAxZnI7CiAgZ2FwOiAwOwogIGJvcmRlci10b3A6IDFweCBzb2xpZCByZ2JhKDEwLDgsNSwwLjEpOwogIHBhZGRpbmctdG9wOiAzMnB4Owp9Ci5pbnRyby1zdGF0LWl0ZW0gewogIHBhZGRpbmctcmlnaHQ6IDI0cHg7CiAgYm9yZGVyLXJpZ2h0OiAxcHggc29saWQgcmdiYSgxMCw4LDUsMC4xKTsKfQouaW50cm8tc3RhdC1pdGVtOmxhc3QtY2hpbGQgeyBib3JkZXItcmlnaHQ6IG5vbmU7IHBhZGRpbmctbGVmdDogMjRweDsgcGFkZGluZy1yaWdodDogMDsgfQouaW50cm8tc3RhdC1pdGVtOm50aC1jaGlsZCgyKSB7IHBhZGRpbmctbGVmdDogMjRweDsgcGFkZGluZy1yaWdodDogMjRweDsgfQouaW50cm8tc3RhdC1udW0gewogIGZvbnQtZmFtaWx5OiB2YXIoLS1zZXJpZik7CiAgZm9udC1zaXplOiAyLjhyZW07CiAgZm9udC13ZWlnaHQ6IDMwMDsKICBjb2xvcjogdmFyKC0tZ29sZCk7CiAgbGluZS1oZWlnaHQ6IDE7CiAgbWFyZ2luLWJvdHRvbTogNnB4Owp9Ci5pbnRyby1zdGF0LWxhYmVsIHsKICBmb250LXNpemU6IDAuNnJlbTsKICBmb250LXdlaWdodDogNDAwOwogIGxldHRlci1zcGFjaW5nOiAwLjJlbTsKICB0ZXh0LXRyYW5zZm9ybTogdXBwZXJjYXNlOwogIGNvbG9yOiB2YXIoLS13YXJtLWdyZXkpOwp9CgovKiDilIDilIAgSE9XIElUIFdPUktTIOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgCAqLwouaG93IHsKICBiYWNrZ3JvdW5kOiB2YXIoLS1vZmYtYmxhY2spOwogIHBhZGRpbmc6IDEyMHB4IDYwcHg7Cn0KLmhvdy1oZWFkZXIgewogIGRpc3BsYXk6IGdyaWQ7CiAgZ3JpZC10ZW1wbGF0ZS1jb2x1bW5zOiAxZnIgMWZyOwogIGdhcDogODBweDsKICBtYXJnaW4tYm90dG9tOiA4MHB4Owp9Ci5ob3ctZGVzYyB7CiAgZm9udC1zaXplOiAwLjgycmVtOwogIGZvbnQtd2VpZ2h0OiAzMDA7CiAgbGluZS1oZWlnaHQ6IDI7CiAgY29sb3I6IHJnYmEoMjU1LDI1NSwyNTUsMC40NSk7CiAgcGFkZGluZy10b3A6IDYwcHg7Cn0KLmhvdy1zdGVwcyB7CiAgZGlzcGxheTogZ3JpZDsKICBncmlkLXRlbXBsYXRlLWNvbHVtbnM6IHJlcGVhdCg0LCAxZnIpOwogIGdhcDogMnB4Owp9Ci5zdGVwIHsKICBiYWNrZ3JvdW5kOiByZ2JhKDI1NSwyNTUsMjU1LDAuMDMpOwogIGJvcmRlcjogMXB4IHNvbGlkIHJnYmEoMjAxLDE1MCw1OCwwLjEpOwogIHBhZGRpbmc6IDQwcHggMzJweDsKICB0cmFuc2l0aW9uOiBhbGwgMC40czsKICBwb3NpdGlvbjogcmVsYXRpdmU7CiAgb3ZlcmZsb3c6IGhpZGRlbjsKfQouc3RlcDo6YmVmb3JlIHsKICBjb250ZW50OiAnJzsKICBwb3NpdGlvbjogYWJzb2x1dGU7IHRvcDogMDsgbGVmdDogMDsgcmlnaHQ6IDA7CiAgaGVpZ2h0OiAxcHg7CiAgYmFja2dyb3VuZDogdmFyKC0tZ29sZCk7CiAgdHJhbnNmb3JtOiBzY2FsZVgoMCk7CiAgdHJhbnNmb3JtLW9yaWdpbjogbGVmdDsKICB0cmFuc2l0aW9uOiB0cmFuc2Zvcm0gMC40czsKfQouc3RlcDpob3ZlciB7IGJhY2tncm91bmQ6IHJnYmEoMjAxLDE1MCw1OCwwLjA1KTsgfQouc3RlcDpob3Zlcjo6YmVmb3JlIHsgdHJhbnNmb3JtOiBzY2FsZVgoMSk7IH0KLnN0ZXAtbnVtIHsKICBmb250LWZhbWlseTogdmFyKC0tc2VyaWYpOwogIGZvbnQtc2l6ZTogMy41cmVtOwogIGZvbnQtd2VpZ2h0OiAzMDA7CiAgY29sb3I6IHJnYmEoMjAxLDE1MCw1OCwwLjIpOwogIGxpbmUtaGVpZ2h0OiAxOwogIG1hcmdpbi1ib3R0b206IDI0cHg7CiAgdHJhbnNpdGlvbjogY29sb3IgMC40czsKfQouc3RlcDpob3ZlciAuc3RlcC1udW0geyBjb2xvcjogcmdiYSgyMDEsMTUwLDU4LDAuNSk7IH0KLnN0ZXAtdGl0bGUgewogIGZvbnQtZmFtaWx5OiB2YXIoLS1zZXJpZik7CiAgZm9udC1zaXplOiAxLjNyZW07CiAgZm9udC13ZWlnaHQ6IDMwMDsKICBjb2xvcjogdmFyKC0td2hpdGUpOwogIG1hcmdpbi1ib3R0b206IDEycHg7Cn0KLnN0ZXAtZGVzYyB7CiAgZm9udC1zaXplOiAwLjcycmVtOwogIGZvbnQtd2VpZ2h0OiAzMDA7CiAgbGluZS1oZWlnaHQ6IDEuOTsKICBjb2xvcjogcmdiYSgyNTUsMjU1LDI1NSwwLjQpOwp9CgovKiDilIDilIAgREVTVElOQVRJT05TIC8gQUkgU0hPV0NBU0Ug4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSAICovCi5kZXN0aW5hdGlvbnMgewogIGJhY2tncm91bmQ6IHZhcigtLWJsYWNrKTsKICBwYWRkaW5nOiAxMjBweCAwOwp9Ci5kZXN0aW5hdGlvbnMtaGVhZGVyIHsKICBwYWRkaW5nOiAwIDYwcHg7CiAgZGlzcGxheTogZmxleDsKICBqdXN0aWZ5LWNvbnRlbnQ6IHNwYWNlLWJldHdlZW47CiAgYWxpZ24taXRlbXM6IGZsZXgtZW5kOwogIG1hcmdpbi1ib3R0b206IDYwcHg7Cn0KLmRlc3RpbmF0aW9ucy1oZWFkZXIgLnNlY3Rpb24taGVhZGxpbmUgeyBjb2xvcjogdmFyKC0td2hpdGUpOyB9Ci5kZXN0aW5hdGlvbnMtbGluayB7CiAgZm9udC1zaXplOiAwLjYycmVtOwogIGZvbnQtd2VpZ2h0OiAzMDA7CiAgbGV0dGVyLXNwYWNpbmc6IDAuMjJlbTsKICB0ZXh0LXRyYW5zZm9ybTogdXBwZXJjYXNlOwogIGNvbG9yOiB2YXIoLS1nb2xkKTsKICB0ZXh0LWRlY29yYXRpb246IG5vbmU7CiAgYm9yZGVyLWJvdHRvbTogMXB4IHNvbGlkIHJnYmEoMjAxLDE1MCw1OCwwLjMpOwogIHBhZGRpbmctYm90dG9tOiAzcHg7CiAgdHJhbnNpdGlvbjogYWxsIDAuM3M7Cn0KLmRlc3RpbmF0aW9ucy1saW5rOmhvdmVyIHsgYm9yZGVyLWNvbG9yOiB2YXIoLS1nb2xkKTsgfQouZGVzdC1ncmlkIHsKICBkaXNwbGF5OiBncmlkOwogIGdyaWQtdGVtcGxhdGUtY29sdW1uczogMmZyIDFmciAxZnI7CiAgZ3JpZC10ZW1wbGF0ZS1yb3dzOiAzNDBweCAyNjBweDsKICBnYXA6IDRweDsKICBwYWRkaW5nOiAwIDYwcHg7Cn0KLmRlc3QtY2FyZCB7CiAgcG9zaXRpb246IHJlbGF0aXZlOwogIG92ZXJmbG93OiBoaWRkZW47CiAgY3Vyc29yOiBub25lOwp9Ci5kZXN0LWNhcmQ6Zmlyc3QtY2hpbGQgewogIGdyaWQtcm93OiAxIC8gMzsKfQouZGVzdC1pbWcgewogIHdpZHRoOiAxMDAlOyBoZWlnaHQ6IDEwMCU7CiAgb2JqZWN0LWZpdDogY292ZXI7CiAgdHJhbnNpdGlvbjogdHJhbnNmb3JtIDAuOHMgY3ViaWMtYmV6aWVyKDAuMTYsMSwwLjMsMSk7CiAgZmlsdGVyOiBicmlnaHRuZXNzKDAuNzUpOwp9Ci5kZXN0LWNhcmQ6aG92ZXIgLmRlc3QtaW1nIHsKICB0cmFuc2Zvcm06IHNjYWxlKDEuMDYpOwogIGZpbHRlcjogYnJpZ2h0bmVzcygwLjYpOwp9Ci5kZXN0LW92ZXJsYXkgewogIHBvc2l0aW9uOiBhYnNvbHV0ZTsKICBpbnNldDogMDsKICBiYWNrZ3JvdW5kOiBsaW5lYXItZ3JhZGllbnQodG8gdG9wLCByZ2JhKDEwLDgsNSwwLjg1KSAwJSwgdHJhbnNwYXJlbnQgNjAlKTsKfQouZGVzdC1pbmZvIHsKICBwb3NpdGlvbjogYWJzb2x1dGU7CiAgYm90dG9tOiAyOHB4OyBsZWZ0OiAyOHB4OyByaWdodDogMjhweDsKfQouZGVzdC1uYW1lIHsKICBmb250LWZhbWlseTogdmFyKC0tc2VyaWYpOwogIGZvbnQtc2l6ZTogMS42cmVtOwogIGZvbnQtd2VpZ2h0OiAzMDA7CiAgY29sb3I6IHZhcigtLXdoaXRlKTsKICBtYXJnaW4tYm90dG9tOiA0cHg7Cn0KLmRlc3Qtc3ViIHsKICBmb250LXNpemU6IDAuNnJlbTsKICBmb250LXdlaWdodDogMzAwOwogIGxldHRlci1zcGFjaW5nOiAwLjJlbTsKICB0ZXh0LXRyYW5zZm9ybTogdXBwZXJjYXNlOwogIGNvbG9yOiByZ2JhKDI1NSwyNTUsMjU1LDAuNSk7Cn0KLmRlc3QtYWktdGFnIHsKICBwb3NpdGlvbjogYWJzb2x1dGU7CiAgdG9wOiAyMHB4OyByaWdodDogMjBweDsKICBmb250LXNpemU6IDAuNTVyZW07CiAgZm9udC13ZWlnaHQ6IDQwMDsKICBsZXR0ZXItc3BhY2luZzogMC4yZW07CiAgdGV4dC10cmFuc2Zvcm06IHVwcGVyY2FzZTsKICBjb2xvcjogdmFyKC0tYmxhY2spOwogIGJhY2tncm91bmQ6IHZhcigtLWdvbGQpOwogIHBhZGRpbmc6IDVweCAxMHB4OwogIG9wYWNpdHk6IDA7CiAgdHJhbnNmb3JtOiB0cmFuc2xhdGVZKC04cHgpOwogIHRyYW5zaXRpb246IGFsbCAwLjNzOwp9Ci5kZXN0LWNhcmQ6aG92ZXIgLmRlc3QtYWktdGFnIHsgb3BhY2l0eTogMTsgdHJhbnNmb3JtOiB0cmFuc2xhdGVZKDApOyB9CgovKiDilIDilIAgQUkgRkVBVFVSRVMg4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSAICovCi5mZWF0dXJlcyB7CiAgYmFja2dyb3VuZDogdmFyKC0tY3JlYW0pOwogIHBhZGRpbmc6IDEyMHB4IDYwcHg7Cn0KLmZlYXR1cmVzLWdyaWQgewogIGRpc3BsYXk6IGdyaWQ7CiAgZ3JpZC10ZW1wbGF0ZS1jb2x1bW5zOiByZXBlYXQoMywgMWZyKTsKICBnYXA6IDJweDsKICBtYXJnaW4tdG9wOiA3MnB4Owp9Ci5mZWF0dXJlLWNhcmQgewogIGJhY2tncm91bmQ6IHZhcigtLXdoaXRlKTsKICBwYWRkaW5nOiA1MnB4IDQwcHg7CiAgYm9yZGVyOiAxcHggc29saWQgcmdiYSgxMCw4LDUsMC4wNik7CiAgdHJhbnNpdGlvbjogYWxsIDAuNHM7CiAgcG9zaXRpb246IHJlbGF0aXZlOwp9Ci5mZWF0dXJlLWNhcmQ6OmFmdGVyIHsKICBjb250ZW50OiAnJzsKICBwb3NpdGlvbjogYWJzb2x1dGU7CiAgYm90dG9tOiAwOyBsZWZ0OiAwOyByaWdodDogMDsKICBoZWlnaHQ6IDJweDsKICBiYWNrZ3JvdW5kOiB2YXIoLS1nb2xkKTsKICB0cmFuc2Zvcm06IHNjYWxlWCgwKTsKICB0cmFuc2Zvcm0tb3JpZ2luOiBsZWZ0OwogIHRyYW5zaXRpb246IHRyYW5zZm9ybSAwLjRzIGN1YmljLWJlemllcigwLjE2LDEsMC4zLDEpOwp9Ci5mZWF0dXJlLWNhcmQ6aG92ZXIgeyB0cmFuc2Zvcm06IHRyYW5zbGF0ZVkoLTRweCk7IGJveC1zaGFkb3c6IDAgMjBweCA2MHB4IHJnYmEoMTAsOCw1LDAuMDgpOyB9Ci5mZWF0dXJlLWNhcmQ6aG92ZXI6OmFmdGVyIHsgdHJhbnNmb3JtOiBzY2FsZVgoMSk7IH0KLmZlYXR1cmUtaWNvbiB7CiAgZm9udC1zaXplOiAxLjZyZW07CiAgbWFyZ2luLWJvdHRvbTogMjhweDsKICBkaXNwbGF5OiBibG9jazsKfQouZmVhdHVyZS10aXRsZSB7CiAgZm9udC1mYW1pbHk6IHZhcigtLXNlcmlmKTsKICBmb250LXNpemU6IDEuNXJlbTsKICBmb250LXdlaWdodDogMzAwOwogIGNvbG9yOiB2YXIoLS1vZmYtYmxhY2spOwogIG1hcmdpbi1ib3R0b206IDE0cHg7CiAgbGluZS1oZWlnaHQ6IDEuMjsKfQouZmVhdHVyZS1kZXNjIHsKICBmb250LXNpemU6IDAuNzZyZW07CiAgZm9udC13ZWlnaHQ6IDMwMDsKICBsaW5lLWhlaWdodDogMS45OwogIGNvbG9yOiB2YXIoLS13YXJtLWdyZXkpOwp9CgovKiDilIDilIAgQUkgQ0hBVCBERU1PIOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgCAqLwouYWktZGVtbyB7CiAgYmFja2dyb3VuZDogdmFyKC0tb2ZmLWJsYWNrKTsKICBwYWRkaW5nOiAxMjBweCA2MHB4OwogIGRpc3BsYXk6IGdyaWQ7CiAgZ3JpZC10ZW1wbGF0ZS1jb2x1bW5zOiAxZnIgMWZyOwogIGdhcDogMTAwcHg7CiAgYWxpZ24taXRlbXM6IGNlbnRlcjsKfQouYWktZGVtby10ZXh0IC5zZWN0aW9uLWhlYWRsaW5lIHsgY29sb3I6IHZhcigtLXdoaXRlKTsgbWFyZ2luLWJvdHRvbTogMjRweDsgfQouYWktZGVtby10ZXh0IHAgewogIGZvbnQtc2l6ZTogMC44MnJlbTsKICBmb250LXdlaWdodDogMzAwOwogIGxpbmUtaGVpZ2h0OiAyOwogIGNvbG9yOiByZ2JhKDI1NSwyNTUsMjU1LDAuNDUpOwogIG1hcmdpbi1ib3R0b206IDQwcHg7Cn0KLmFpLXdpbmRvdyB7CiAgYmFja2dyb3VuZDogIzEyMTAwQzsKICBib3JkZXI6IDFweCBzb2xpZCByZ2JhKDIwMSwxNTAsNTgsMC4xNSk7CiAgYm9yZGVyLXJhZGl1czogMnB4OwogIG92ZXJmbG93OiBoaWRkZW47CiAgYm94LXNoYWRvdzogMCA0MHB4IDEwMHB4IHJnYmEoMCwwLDAsMC42KTsKfQouYWktd2luZG93LWhlYWRlciB7CiAgYmFja2dyb3VuZDogcmdiYSgyMDEsMTUwLDU4LDAuMDgpOwogIGJvcmRlci1ib3R0b206IDFweCBzb2xpZCByZ2JhKDIwMSwxNTAsNTgsMC4xMik7CiAgcGFkZGluZzogMTRweCAyMHB4OwogIGRpc3BsYXk6IGZsZXg7CiAgYWxpZ24taXRlbXM6IGNlbnRlcjsKICBnYXA6IDhweDsKICBmb250LXNpemU6IDAuNnJlbTsKICBmb250LXdlaWdodDogMzAwOwogIGxldHRlci1zcGFjaW5nOiAwLjJlbTsKICB0ZXh0LXRyYW5zZm9ybTogdXBwZXJjYXNlOwogIGNvbG9yOiB2YXIoLS1nb2xkKTsKfQouYWktd2luZG93LWRvdCB7CiAgd2lkdGg6IDdweDsgaGVpZ2h0OiA3cHg7CiAgYm9yZGVyLXJhZGl1czogNTAlOwogIGJhY2tncm91bmQ6IHJnYmEoMjAxLDE1MCw1OCwwLjQpOwogIGFuaW1hdGlvbjogYmxpbmsgMnMgZWFzZS1pbi1vdXQgaW5maW5pdGU7Cn0KQGtleWZyYW1lcyBibGluayB7CiAgMCUsMTAwJXtvcGFjaXR5OjF9IDUwJXtvcGFjaXR5OjAuM30KfQouYWktbWVzc2FnZXMgeyBwYWRkaW5nOiAyNHB4OyBkaXNwbGF5OiBmbGV4OyBmbGV4LWRpcmVjdGlvbjogY29sdW1uOyBnYXA6IDE2cHg7IH0KLmFpLW1zZyB7CiAgZGlzcGxheTogZmxleDsKICBnYXA6IDEycHg7CiAgb3BhY2l0eTogMDsKICB0cmFuc2Zvcm06IHRyYW5zbGF0ZVkoMTJweCk7CiAgYW5pbWF0aW9uOiBtc2dJbiAwLjVzIGVhc2UgZm9yd2FyZHM7Cn0KLmFpLW1zZzpudGgtY2hpbGQoMSkgeyBhbmltYXRpb24tZGVsYXk6IDAuNXM7IH0KLmFpLW1zZzpudGgtY2hpbGQoMikgeyBhbmltYXRpb24tZGVsYXk6IDEuMnM7IH0KLmFpLW1zZzpudGgtY2hpbGQoMykgeyBhbmltYXRpb24tZGVsYXk6IDIuMnM7IH0KLmFpLW1zZzpudGgtY2hpbGQoNCkgeyBhbmltYXRpb24tZGVsYXk6IDMuNHM7IH0KQGtleWZyYW1lcyBtc2dJbiB7CiAgdG8geyBvcGFjaXR5OiAxOyB0cmFuc2Zvcm06IHRyYW5zbGF0ZVkoMCk7IH0KfQouYWktbXNnLWF2YXRhciB7CiAgd2lkdGg6IDMwcHg7IGhlaWdodDogMzBweDsKICBib3JkZXItcmFkaXVzOiA1MCU7CiAgZmxleC1zaHJpbms6IDA7CiAgZGlzcGxheTogZmxleDsKICBhbGlnbi1pdGVtczogY2VudGVyOwogIGp1c3RpZnktY29udGVudDogY2VudGVyOwogIGZvbnQtc2l6ZTogMC43cmVtOwogIGZvbnQtd2VpZ2h0OiA1MDA7Cn0KLnVzZXItYXZhdGFyIHsgYmFja2dyb3VuZDogcmdiYSgyNTUsMjU1LDI1NSwwLjEpOyBjb2xvcjogcmdiYSgyNTUsMjU1LDI1NSwwLjYpOyB9Ci5haS1hdmF0YXIgeyBiYWNrZ3JvdW5kOiB2YXIoLS1nb2xkKTsgY29sb3I6IHZhcigtLWJsYWNrKTsgZm9udC13ZWlnaHQ6IDcwMDsgZm9udC1zaXplOiAwLjZyZW07IGxldHRlci1zcGFjaW5nOiAwLjA1ZW07IH0KLmFpLW1zZy1idWJibGUgewogIGJhY2tncm91bmQ6IHJnYmEoMjU1LDI1NSwyNTUsMC4wNCk7CiAgYm9yZGVyOiAxcHggc29saWQgcmdiYSgyNTUsMjU1LDI1NSwwLjA3KTsKICBib3JkZXItcmFkaXVzOiAycHg7CiAgcGFkZGluZzogMTJweCAxNnB4OwogIGZvbnQtc2l6ZTogMC43NHJlbTsKICBmb250LXdlaWdodDogMzAwOwogIGxpbmUtaGVpZ2h0OiAxLjc7CiAgY29sb3I6IHJnYmEoMjU1LDI1NSwyNTUsMC43KTsKICBtYXgtd2lkdGg6IDg1JTsKfQouYWktbXNnLnVzZXIgLmFpLW1zZy1idWJibGUgewogIGJhY2tncm91bmQ6IHJnYmEoMjAxLDE1MCw1OCwwLjA4KTsKICBib3JkZXItY29sb3I6IHJnYmEoMjAxLDE1MCw1OCwwLjE1KTsKICBjb2xvcjogcmdiYSgyNTUsMjU1LDI1NSwwLjgpOwp9Ci5haS1tc2ctYnViYmxlIHN0cm9uZyB7IGNvbG9yOiB2YXIoLS1nb2xkLWxpZ2h0KTsgZm9udC13ZWlnaHQ6IDQwMDsgfQouYWktdHlwaW5nIHsKICBkaXNwbGF5OiBmbGV4OwogIGdhcDogNHB4OwogIHBhZGRpbmc6IDhweCAxMnB4Owp9Ci5haS10eXBpbmcgc3BhbiB7CiAgd2lkdGg6IDVweDsgaGVpZ2h0OiA1cHg7CiAgYmFja2dyb3VuZDogdmFyKC0tZ29sZCk7CiAgYm9yZGVyLXJhZGl1czogNTAlOwogIGFuaW1hdGlvbjogdHlwaW5nIDEuMnMgZWFzZSBpbmZpbml0ZTsKICBvcGFjaXR5OiAwLjU7Cn0KLmFpLXR5cGluZyBzcGFuOm50aC1jaGlsZCgyKSB7IGFuaW1hdGlvbi1kZWxheTogMC4yczsgfQouYWktdHlwaW5nIHNwYW46bnRoLWNoaWxkKDMpIHsgYW5pbWF0aW9uLWRlbGF5OiAwLjRzOyB9CkBrZXlmcmFtZXMgdHlwaW5nIHsgMCUsMTAwJXt0cmFuc2Zvcm06dHJhbnNsYXRlWSgwKX01MCV7dHJhbnNmb3JtOnRyYW5zbGF0ZVkoLTVweCl9IH0KCi8qIOKUgOKUgCBURVNUSU1PTklBTFMg4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSAICovCi50ZXN0aW1vbmlhbHMgewogIGJhY2tncm91bmQ6IHZhcigtLWJsYWNrKTsKICBwYWRkaW5nOiAxMjBweCA2MHB4OwogIHBvc2l0aW9uOiByZWxhdGl2ZTsKICBvdmVyZmxvdzogaGlkZGVuOwp9Ci50ZXN0aW1vbmlhbHM6OmJlZm9yZSB7CiAgY29udGVudDogJyInOwogIGZvbnQtZmFtaWx5OiB2YXIoLS1zZXJpZik7CiAgZm9udC1zaXplOiA0MHZ3OwogIGZvbnQtd2VpZ2h0OiAzMDA7CiAgY29sb3I6IHJnYmEoMjAxLDE1MCw1OCwwLjAzKTsKICBwb3NpdGlvbjogYWJzb2x1dGU7CiAgdG9wOiAtMTAlOwogIGxlZnQ6IC01JTsKICBsaW5lLWhlaWdodDogMTsKICBwb2ludGVyLWV2ZW50czogbm9uZTsKfQoudGVzdGltb25pYWxzLWdyaWQgewogIGRpc3BsYXk6IGdyaWQ7CiAgZ3JpZC10ZW1wbGF0ZS1jb2x1bW5zOiByZXBlYXQoMywgMWZyKTsKICBnYXA6IDQwcHg7CiAgbWFyZ2luLXRvcDogNzJweDsKICBwb3NpdGlvbjogcmVsYXRpdmU7Cn0KLnRjYXJkIHsKICBib3JkZXI6IDFweCBzb2xpZCByZ2JhKDIwMSwxNTAsNTgsMC4xKTsKICBwYWRkaW5nOiA0NHB4IDM2cHg7CiAgcG9zaXRpb246IHJlbGF0aXZlOwogIHRyYW5zaXRpb246IGFsbCAwLjRzOwp9Ci50Y2FyZDpob3ZlciB7CiAgYm9yZGVyLWNvbG9yOiByZ2JhKDIwMSwxNTAsNTgsMC4zKTsKICBiYWNrZ3JvdW5kOiByZ2JhKDIwMSwxNTAsNTgsMC4wMik7Cn0KLnRjYXJkLXN0YXJzIHsKICBjb2xvcjogdmFyKC0tZ29sZCk7CiAgZm9udC1zaXplOiAwLjdyZW07CiAgbGV0dGVyLXNwYWNpbmc6IDNweDsKICBtYXJnaW4tYm90dG9tOiAyMHB4Owp9Ci50Y2FyZC10ZXh0IHsKICBmb250LWZhbWlseTogdmFyKC0tc2VyaWYpOwogIGZvbnQtc2l6ZTogMS4wOHJlbTsKICBmb250LXdlaWdodDogMzAwOwogIGZvbnQtc3R5bGU6IGl0YWxpYzsKICBsaW5lLWhlaWdodDogMS43NTsKICBjb2xvcjogcmdiYSgyNTUsMjU1LDI1NSwwLjcpOwogIG1hcmdpbi1ib3R0b206IDI4cHg7Cn0KLnRjYXJkLWF1dGhvciB7CiAgZGlzcGxheTogZmxleDsKICBhbGlnbi1pdGVtczogY2VudGVyOwogIGdhcDogMTRweDsKfQoudGNhcmQtYXZhdGFyIHsKICB3aWR0aDogMzhweDsgaGVpZ2h0OiAzOHB4OwogIGJvcmRlci1yYWRpdXM6IDUwJTsKICBiYWNrZ3JvdW5kOiByZ2JhKDIwMSwxNTAsNTgsMC4xNSk7CiAgZGlzcGxheTogZmxleDsKICBhbGlnbi1pdGVtczogY2VudGVyOwogIGp1c3RpZnktY29udGVudDogY2VudGVyOwogIGZvbnQtZmFtaWx5OiB2YXIoLS1zZXJpZik7CiAgZm9udC1zaXplOiAxcmVtOwogIGNvbG9yOiB2YXIoLS1nb2xkKTsKfQoudGNhcmQtbmFtZSB7CiAgZm9udC1zaXplOiAwLjcycmVtOwogIGZvbnQtd2VpZ2h0OiA0MDA7CiAgY29sb3I6IHZhcigtLXdoaXRlKTsKICBsZXR0ZXItc3BhY2luZzogMC4wOGVtOwp9Ci50Y2FyZC1tZXRhIHsKICBmb250LXNpemU6IDAuNjJyZW07CiAgZm9udC13ZWlnaHQ6IDMwMDsKICBjb2xvcjogdmFyKC0td2FybS1ncmV5KTsKICBsZXR0ZXItc3BhY2luZzogMC4xZW07CiAgbWFyZ2luLXRvcDogMnB4Owp9CgovKiDilIDilIAgQ1RBIEZVTEwg4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSAICovCi5jdGEtZnVsbCB7CiAgcG9zaXRpb246IHJlbGF0aXZlOwogIGhlaWdodDogNzB2aDsKICBtaW4taGVpZ2h0OiA1MDBweDsKICBkaXNwbGF5OiBmbGV4OwogIGFsaWduLWl0ZW1zOiBjZW50ZXI7CiAganVzdGlmeS1jb250ZW50OiBjZW50ZXI7CiAgdGV4dC1hbGlnbjogY2VudGVyOwogIG92ZXJmbG93OiBoaWRkZW47Cn0KLmN0YS1iZyB7CiAgcG9zaXRpb246IGFic29sdXRlOyBpbnNldDogMDsKICBiYWNrZ3JvdW5kOgogICAgbGluZWFyLWdyYWRpZW50KHJnYmEoMTAsOCw1LDAuNSksIHJnYmEoMTAsOCw1LDAuNykpLAogICAgdXJsKCdodHRwczovL2ltYWdlcy51bnNwbGFzaC5jb20vcGhvdG8tMTQ2OTg1NDUyMzA4Ni1jYzAyZmU1ZDg4MDA/dz0xNjAwJnE9ODAnKSBjZW50ZXIvY292ZXIgbm8tcmVwZWF0Owp9Ci5jdGEtY29udGVudCB7CiAgcG9zaXRpb246IHJlbGF0aXZlOwogIHotaW5kZXg6IDI7CiAgbWF4LXdpZHRoOiA3MDBweDsKICBwYWRkaW5nOiAwIDQwcHg7Cn0KLmN0YS1oZWFkbGluZSB7CiAgZm9udC1mYW1pbHk6IHZhcigtLXNlcmlmKTsKICBmb250LXNpemU6IGNsYW1wKDIuOHJlbSwgNnZ3LCA1LjVyZW0pOwogIGZvbnQtd2VpZ2h0OiAzMDA7CiAgbGluZS1oZWlnaHQ6IDEuMTsKICBjb2xvcjogdmFyKC0td2hpdGUpOwogIG1hcmdpbi1ib3R0b206IDI0cHg7Cn0KLmN0YS1oZWFkbGluZSBlbSB7IGZvbnQtc3R5bGU6IGl0YWxpYzsgY29sb3I6IHZhcigtLWdvbGQtbGlnaHQpOyB9Ci5jdGEtc3ViIHsKICBmb250LXNpemU6IDAuNzhyZW07CiAgZm9udC13ZWlnaHQ6IDMwMDsKICBsaW5lLWhlaWdodDogMS45OwogIGNvbG9yOiByZ2JhKDI1NSwyNTUsMjU1LDAuNTUpOwogIG1hcmdpbi1ib3R0b206IDQ0cHg7CiAgbGV0dGVyLXNwYWNpbmc6IDAuMDVlbTsKfQouY3RhLWFjdGlvbnMgewogIGRpc3BsYXk6IGZsZXg7CiAgYWxpZ24taXRlbXM6IGNlbnRlcjsKICBqdXN0aWZ5LWNvbnRlbnQ6IGNlbnRlcjsKICBnYXA6IDI0cHg7Cn0KCi8qIOKUgOKUgCBGT09URVIg4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSAICovCmZvb3RlciB7CiAgYmFja2dyb3VuZDogdmFyKC0tb2ZmLWJsYWNrKSAhaW1wb3J0YW50OwogIGJvcmRlci10b3A6IDFweCBzb2xpZCByZ2JhKDIwMSwxNTAsNTgsMC4xKTsKICBwYWRkaW5nOiA3MnB4IDYwcHggMzJweDsKfQouZm9vdGVyLXRvcCB7CiAgZGlzcGxheTogZ3JpZDsKICBncmlkLXRlbXBsYXRlLWNvbHVtbnM6IDJmciAxZnIgMWZyIDFmcjsKICBnYXA6IDYwcHg7CiAgbWFyZ2luLWJvdHRvbTogNTZweDsKfQouZm9vdGVyLWJyYW5kLW5hbWUgewogIGZvbnQtZmFtaWx5OiB2YXIoLS1zZXJpZik7CiAgZm9udC1zaXplOiAycmVtOwogIGZvbnQtd2VpZ2h0OiAzMDA7CiAgY29sb3I6IHZhcigtLWdvbGQpOwogIGxldHRlci1zcGFjaW5nOiAwLjE1ZW07CiAgbWFyZ2luLWJvdHRvbTogMTZweDsKfQouZm9vdGVyLWJyYW5kLXRhZ2xpbmUgewogIGZvbnQtc2l6ZTogMC42OHJlbTsKICBmb250LXdlaWdodDogMzAwOwogIGxpbmUtaGVpZ2h0OiAxLjk7CiAgY29sb3I6IHJnYmEoMjU1LDI1NSwyNTUsMC4zNSk7CiAgbWF4LXdpZHRoOiAyNDBweDsKICBtYXJnaW4tYm90dG9tOiAyOHB4Owp9Ci5mb290ZXItY29udGFjdCBhIHsKICBkaXNwbGF5OiBibG9jazsKICBmb250LXNpemU6IDAuNjVyZW07CiAgZm9udC13ZWlnaHQ6IDMwMDsKICBjb2xvcjogcmdiYSgyNTUsMjU1LDI1NSwwLjQpOwogIHRleHQtZGVjb3JhdGlvbjogbm9uZTsKICBtYXJnaW4tYm90dG9tOiA2cHg7CiAgdHJhbnNpdGlvbjogY29sb3IgMC4zczsKfQouZm9vdGVyLWNvbnRhY3QgYTpob3ZlciB7IGNvbG9yOiB2YXIoLS1nb2xkKTsgfQouZm9vdGVyLWNvbC10aXRsZSB7CiAgZm9udC1zaXplOiAwLjU4cmVtOwogIGZvbnQtd2VpZ2h0OiA0MDA7CiAgbGV0dGVyLXNwYWNpbmc6IDAuMjhlbTsKICB0ZXh0LXRyYW5zZm9ybTogdXBwZXJjYXNlOwogIGNvbG9yOiB2YXIoLS1nb2xkKTsKICBtYXJnaW4tYm90dG9tOiAyMHB4Owp9Ci5mb290ZXItY29sIGEgewogIGRpc3BsYXk6IGJsb2NrOwogIGZvbnQtc2l6ZTogMC43MnJlbTsKICBmb250LXdlaWdodDogNzAwOwogIGNvbG9yOiAjRkZGRkZGOwogIHRleHQtZGVjb3JhdGlvbjogbm9uZTsKICBtYXJnaW4tYm90dG9tOiAxMHB4OwogIHRyYW5zaXRpb246IGNvbG9yIDAuM3M7CiAgY3Vyc29yOiBub25lOwp9Ci5mb290ZXItY29sIGE6aG92ZXIgeyBjb2xvcjogdmFyKC0tZ29sZC1saWdodCk7IH0KLmZvb3Rlci1ib3R0b20gewogIGJvcmRlci10b3A6IDFweCBzb2xpZCByZ2JhKDI1NSwyNTUsMjU1LDAuMDYpOwogIHBhZGRpbmctdG9wOiAyOHB4OwogIGRpc3BsYXk6IGZsZXg7CiAganVzdGlmeS1jb250ZW50OiBzcGFjZS1iZXR3ZWVuOwogIGFsaWduLWl0ZW1zOiBjZW50ZXI7Cn0KLmZvb3Rlci1jb3B5IHsKICBmb250LXNpemU6IDAuNjJyZW07CiAgZm9udC13ZWlnaHQ6IDMwMDsKICBjb2xvcjogcmdiYSgyNTUsMjU1LDI1NSwwLjIpOwogIGxldHRlci1zcGFjaW5nOiAwLjFlbTsKfQouZm9vdGVyLWxlZ2FsIHsKICBkaXNwbGF5OiBmbGV4OwogIGdhcDogMjhweDsKfQouZm9vdGVyLWxlZ2FsIGEgewogIGZvbnQtc2l6ZTogMC42MnJlbTsKICBmb250LXdlaWdodDogNzAwOwogIGNvbG9yOiAjRkZGRkZGOwogIHRleHQtZGVjb3JhdGlvbjogbm9uZTsKICBsZXR0ZXItc3BhY2luZzogMC4wOGVtOwogIHRyYW5zaXRpb246IGNvbG9yIDAuM3M7Cn0KLmZvb3Rlci1sZWdhbCBhOmhvdmVyIHsgY29sb3I6IHZhcigtLWdvbGQpOyB9CgovKiDilIDilIAgUkVWRUFMIEFOSU1BVElPTlMg4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSAICovCi5yZXZlYWwgewogIG9wYWNpdHk6IDA7CiAgdHJhbnNmb3JtOiB0cmFuc2xhdGVZKDMwcHgpOwogIHRyYW5zaXRpb246IG9wYWNpdHkgMC44cyBlYXNlLCB0cmFuc2Zvcm0gMC44cyBjdWJpYy1iZXppZXIoMC4xNiwxLDAuMywxKTsKfQoucmV2ZWFsLnZpc2libGUgewogIG9wYWNpdHk6IDE7CiAgdHJhbnNmb3JtOiB0cmFuc2xhdGVZKDApOwp9CgovKiDilIDilIAgUkVTUE9OU0lWRSDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIAgKi8KQG1lZGlhIChtYXgtd2lkdGg6IDEwMjRweCkgewogIG5hdiB7IHBhZGRpbmc6IDIwcHggMzBweDsgfQogIC5oZXJvLCBzZWN0aW9uLCAuaW50cm8sIC5haS1kZW1vLCAuZGVzdGluYXRpb25zLWhlYWRlciwgLmhvdyB7IHBhZGRpbmctbGVmdDogMzBweDsgcGFkZGluZy1yaWdodDogMzBweDsgfQogIC5kZXN0LWdyaWQgeyBwYWRkaW5nOiAwIDMwcHg7IH0KICAuaW50cm8sIC5ob3ctaGVhZGVyLCAuYWktZGVtbyB7IGdyaWQtdGVtcGxhdGUtY29sdW1uczogMWZyOyBnYXA6IDQwcHg7IH0KICAuaG93LXN0ZXBzIHsgZ3JpZC10ZW1wbGF0ZS1jb2x1bW5zOiAxZnIgMWZyOyB9CiAgLmRlc3QtZ3JpZCB7IGdyaWQtdGVtcGxhdGUtY29sdW1uczogMWZyIDFmcjsgZ3JpZC10ZW1wbGF0ZS1yb3dzOiBhdXRvOyB9CiAgLmRlc3QtY2FyZDpmaXJzdC1jaGlsZCB7IGdyaWQtcm93OiBhdXRvOyBncmlkLWNvbHVtbjogMSAvIC0xOyBoZWlnaHQ6IDM0MHB4OyB9CiAgLmZlYXR1cmVzLWdyaWQsIC50ZXN0aW1vbmlhbHMtZ3JpZCB7IGdyaWQtdGVtcGxhdGUtY29sdW1uczogMWZyOyB9CiAgLmZvb3Rlci10b3AgeyBncmlkLXRlbXBsYXRlLWNvbHVtbnM6IDFmciAxZnI7IGdhcDogNDBweDsgfQp9CkBtZWRpYSAobWF4LXdpZHRoOiA2NDBweCkgewogIG5hdiAubmF2LWxpbmtzIHsgZGlzcGxheTogbm9uZTsgfQogIC5oZXJvLWhlYWRsaW5lIHsgZm9udC1zaXplOiAyLjhyZW07IH0KICAuaG93LXN0ZXBzIHsgZ3JpZC10ZW1wbGF0ZS1jb2x1bW5zOiAxZnI7IH0KICAuZm9vdGVyLXRvcCB7IGdyaWQtdGVtcGxhdGUtY29sdW1uczogMWZyOyB9CiAgc2VjdGlvbiB7IHBhZGRpbmc6IDcycHggMjBweDsgfQp9Cjwvc3R5bGU+CjwvaGVhZD4KPGJvZHk+Cgo8IS0tIEN1c3RvbSBjdXJzb3IgLS0+CjxkaXYgY2xhc3M9ImN1cnNvciIgaWQ9ImN1cnNvciI+PC9kaXY+CjxkaXYgY2xhc3M9ImN1cnNvci1yaW5nIiBpZD0iY3Vyc29yUmluZyI+PC9kaXY+Cgo8IS0tIOKUgOKUgCBOQVYg4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSAIC0tPgo8bmF2PgogIDxhIGNsYXNzPSJuYXYtbG9nbyIgaHJlZj0iIyI+VGhlIFRyaXAgVGhlb3J5PC9hPgogIDx1bCBjbGFzcz0ibmF2LWxpbmtzIj4KICAgIDxsaT48YSBocmVmPSIjaG93Ij5Ib3cgSXQgV29ya3M8L2E+PC9saT4KICAgIDxsaT48YSBocmVmPSIjZGVzdGluYXRpb25zIj5EZXN0aW5hdGlvbnM8L2E+PC9saT4KICAgIDxsaT48YSBocmVmPSIjYWkiPlRoZSBBSTwvYT48L2xpPgogICAgPGxpPjxhIGhyZWY9IiNzdG9yaWVzIj5TdG9yaWVzPC9hPjwvbGk+CiAgPC91bD4KICA8YSBjbGFzcz0ibmF2LWN0YSIgaHJlZj0iamF2YXNjcmlwdDp2b2lkKDApIiBvbmNsaWNrPSJvcGVuVFRUU2lnbmluKCkiPlN0YXJ0IFBsYW5uaW5nPC9hPgo8L25hdj4KCjwhLS0g4pSA4pSAIEhFUk8g4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSAIC0tPgo8c2VjdGlvbiBjbGFzcz0iaGVybyIgaWQ9InRvcCI+CiAgPGRpdiBjbGFzcz0iaGVyby1iZyI+PC9kaXY+CiAgPGRpdiBjbGFzcz0iaGVyby1ncmFpbiI+PC9kaXY+CiAgPGRpdiBjbGFzcz0iaGVyby1jb250ZW50Ij4KICAgIDxkaXYgY2xhc3M9Imhlcm8tdGFnIj5JbmRpYSdzIEZpcnN0IEFJIFRyYXZlbCBDb25jaWVyZ2U8L2Rpdj4KICAgIDxoMSBjbGFzcz0iaGVyby1oZWFkbGluZSI+CiAgICAgIEV2ZXJ5IGpvdXJuZXksPGJyPgogICAgICA8ZW0+cGVyZmVjdGx5PC9lbT4gZGVzaWduZWQ8YnI+CiAgICAgIGJ5IGludGVsbGlnZW5jZS4KICAgIDwvaDE+CiAgICA8cCBjbGFzcz0iaGVyby1zdWIiPgogICAgICBXZSBkb24ndCBzZWFyY2guIFdlIHVuZGVyc3RhbmQuIFRUVCdzIEFJIGNvbmNpZXJnZSBsZWFybnMgaG93IHlvdSB0cmF2ZWwg4oCUIHlvdXIgcGFjZSwgeW91ciBwYWxldHRlLCB5b3VyIGJ1ZGdldCDigJQgYW5kIGJ1aWxkcyBqb3VybmV5cyB0aGF0IGZlZWwgaGFuZGNyYWZ0ZWQuCiAgICA8L3A+CiAgICA8ZGl2IGNsYXNzPSJoZXJvLWFjdGlvbnMiPgogICAgICA8YSBocmVmPSJqYXZhc2NyaXB0OnZvaWQoMCkiIG9uY2xpY2s9Im9wZW5UVFRTaWduaW4oKSIgY2xhc3M9ImJ0bi1wcmltYXJ5Ij48c3Bhbj5CZWdpbiBZb3VyIEpvdXJuZXk8L3NwYW4+PC9hPgogICAgICA8YSBocmVmPSIjaG93IiBjbGFzcz0iYnRuLWdob3N0Ij5TZWUgSG93IEl0IFdvcmtzPC9hPgogICAgPC9kaXY+CiAgPC9kaXY+CiAgPGRpdiBjbGFzcz0iaGVyby1zY3JvbGwiPlNjcm9sbDwvZGl2Pgo8L3NlY3Rpb24+Cgo8IS0tIOKUgOKUgCBNQVJRVUVFIOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgCAtLT4KPGRpdiBjbGFzcz0ibWFycXVlZS1zdHJpcCI+CiAgPGRpdiBjbGFzcz0ibWFycXVlZS1pbm5lciI+CiAgICA8c3BhbiBjbGFzcz0ibWFycXVlZS1pdGVtIj5BSS1DdXJhdGVkIEl0aW5lcmFyaWVzPC9zcGFuPgogICAgPHNwYW4gY2xhc3M9Im1hcnF1ZWUtaXRlbSI+THV4dXJ5IFN0YXlzPC9zcGFuPgogICAgPHNwYW4gY2xhc3M9Im1hcnF1ZWUtaXRlbSI+UGVyc29uYWxpc2VkIENvbmNpZXJnZTwvc3Bhbj4KICAgIDxzcGFuIGNsYXNzPSJtYXJxdWVlLWl0ZW0iPkdvYSDCtyBSYWphc3RoYW4gwrcgS2VyYWxhPC9zcGFuPgogICAgPHNwYW4gY2xhc3M9Im1hcnF1ZWUtaXRlbSI+SW50ZXJuYXRpb25hbCBFc2NhcGVzPC9zcGFuPgogICAgPHNwYW4gY2xhc3M9Im1hcnF1ZWUtaXRlbSI+MjQgLyA3IEFJIFN1cHBvcnQ8L3NwYW4+CiAgICA8c3BhbiBjbGFzcz0ibWFycXVlZS1pdGVtIj5IaWRkZW4gRXhwZXJpZW5jZXM8L3NwYW4+CiAgICA8c3BhbiBjbGFzcz0ibWFycXVlZS1pdGVtIj5SZWFsLVRpbWUgQm9va2luZ3M8L3NwYW4+CiAgICA8c3BhbiBjbGFzcz0ibWFycXVlZS1pdGVtIj5BSS1DdXJhdGVkIEl0aW5lcmFyaWVzPC9zcGFuPgogICAgPHNwYW4gY2xhc3M9Im1hcnF1ZWUtaXRlbSI+THV4dXJ5IFN0YXlzPC9zcGFuPgogICAgPHNwYW4gY2xhc3M9Im1hcnF1ZWUtaXRlbSI+UGVyc29uYWxpc2VkIENvbmNpZXJnZTwvc3Bhbj4KICAgIDxzcGFuIGNsYXNzPSJtYXJxdWVlLWl0ZW0iPkdvYSDCtyBSYWphc3RoYW4gwrcgS2VyYWxhPC9zcGFuPgogICAgPHNwYW4gY2xhc3M9Im1hcnF1ZWUtaXRlbSI+SW50ZXJuYXRpb25hbCBFc2NhcGVzPC9zcGFuPgogICAgPHNwYW4gY2xhc3M9Im1hcnF1ZWUtaXRlbSI+MjQgLyA3IEFJIFN1cHBvcnQ8L3NwYW4+CiAgICA8c3BhbiBjbGFzcz0ibWFycXVlZS1pdGVtIj5IaWRkZW4gRXhwZXJpZW5jZXM8L3NwYW4+CiAgICA8c3BhbiBjbGFzcz0ibWFycXVlZS1pdGVtIj5SZWFsLVRpbWUgQm9va2luZ3M8L3NwYW4+CiAgPC9kaXY+CjwvZGl2PgoKPCEtLSDilIDilIAgSU5UUk8gLyBNQU5JRkVTVE8g4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSAIC0tPgo8c2VjdGlvbiBjbGFzcz0iaW50cm8gcmV2ZWFsIj4KICA8ZGl2IGNsYXNzPSJpbnRyby1sZWZ0Ij4KICAgIDxkaXYgY2xhc3M9InNlY3Rpb24tbGFiZWwiPk91ciBQaGlsb3NvcGh5PC9kaXY+CiAgICA8aDIgY2xhc3M9InNlY3Rpb24taGVhZGxpbmUiIHN0eWxlPSJjb2xvcjp2YXIoLS1vZmYtYmxhY2spIj4KICAgICAgVHJhdmVsIGlzIG5vdDxicj5hIHRyYW5zYWN0aW9uLjxicj4KICAgICAgPGVtPkl0J3MgYSB0aGVvcnkuPC9lbT4KICAgIDwvaDI+CiAgPC9kaXY+CiAgPGRpdiBjbGFzcz0iaW50cm8tcmlnaHQiPgogICAgPHA+VHJhdmVsIGFnZW5jaWVzIGdpdmUgeW91IG9wdGlvbnMuIFNlYXJjaCBlbmdpbmVzIGdpdmUgeW91IGxpc3RzLiA8c3Ryb25nPlRUVCBnaXZlcyB5b3UgdW5kZXJzdGFuZGluZy48L3N0cm9uZz4gT3VyIEFJIGNvbmNpZXJnZSBkb2Vzbid0IGp1c3Qga25vdyB3aGVyZSB0byBnbyDigJQgaXQga25vd3MgaG93IDxlbT55b3U8L2VtPiB0cmF2ZWwuPC9wPgogICAgPHA+QnVpbHQgZm9yIGEgbmV3IGdlbmVyYXRpb24gb2YgSW5kaWFuIHRyYXZlbGxlcnMgd2hvIGRlbWFuZCBtb3JlIHRoYW4gdGVtcGxhdGVzLiBQb3dlcmVkIGJ5IENsYXVkZSBBSS4gQmFja2VkIGJ5IGEgbmV0d29yayBvZiAzMDArIGN1cmF0ZWQgcGFydG5lcnMgYWNyb3NzIEluZGlhIGFuZCB0aGUgd29ybGQuPC9wPgogICAgPHA+RnJvbSBhIOKCuTE1LDAwMCBzb2xvIHdlZWtlbmQgaW4gQ29vcmcgdG8gYSDigrk1IGxha2ggaG9uZXltb29uIGluIHRoZSBNYWxkaXZlcyDigJQgZXZlcnkgcGxhbiBpcyB1bmlxdWVseSB5b3Vycy48L3A+CiAgICA8ZGl2IGNsYXNzPSJpbnRyby1zdGF0Ij4KICAgICAgPGRpdiBjbGFzcz0iaW50cm8tc3RhdC1pdGVtIj4KICAgICAgICA8ZGl2IGNsYXNzPSJpbnRyby1zdGF0LW51bSI+MzAwPHNwYW4gc3R5bGU9ImZvbnQtc2l6ZToxLjVyZW0iPis8L3NwYW4+PC9kaXY+CiAgICAgICAgPGRpdiBjbGFzcz0iaW50cm8tc3RhdC1sYWJlbCI+Q3VyYXRlZCBwYXJ0bmVyczwvZGl2PgogICAgICA8L2Rpdj4KICAgICAgPGRpdiBjbGFzcz0iaW50cm8tc3RhdC1pdGVtIj4KICAgICAgICA8ZGl2IGNsYXNzPSJpbnRyby1zdGF0LW51bSI+NTA8c3BhbiBzdHlsZT0iZm9udC1zaXplOjEuNXJlbSI+Kzwvc3Bhbj48L2Rpdj4KICAgICAgICA8ZGl2IGNsYXNzPSJpbnRyby1zdGF0LWxhYmVsIj5EZXN0aW5hdGlvbnM8L2Rpdj4KICAgICAgPC9kaXY+CiAgICAgIDxkaXYgY2xhc3M9ImludHJvLXN0YXQtaXRlbSI+CiAgICAgICAgPGRpdiBjbGFzcz0iaW50cm8tc3RhdC1udW0iPuKInjwvZGl2PgogICAgICAgIDxkaXYgY2xhc3M9ImludHJvLXN0YXQtbGFiZWwiPlBvc3NpYmxlIGpvdXJuZXlzPC9kaXY+CiAgICAgIDwvZGl2PgogICAgPC9kaXY+CiAgPC9kaXY+Cjwvc2VjdGlvbj4KCjwhLS0g4pSA4pSAIEhPVyBJVCBXT1JLUyDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIAgLS0+CjxzZWN0aW9uIGNsYXNzPSJob3cgcmV2ZWFsIiBpZD0iaG93Ij4KICA8ZGl2IGNsYXNzPSJob3ctaGVhZGVyIj4KICAgIDxkaXY+CiAgICAgIDxkaXYgY2xhc3M9InNlY3Rpb24tbGFiZWwiPlRoZSBQcm9jZXNzPC9kaXY+CiAgICAgIDxoMiBjbGFzcz0ic2VjdGlvbi1oZWFkbGluZSI+CiAgICAgICAgTm90IGp1c3QgYSB0b29sLjxicj4KICAgICAgICA8ZW0+QSB0aGVvcnk8L2VtPjxicj4KICAgICAgICBvZiB0cmF2ZWwuCiAgICAgIDwvaDI+CiAgICA8L2Rpdj4KICAgIDxwIGNsYXNzPSJob3ctZGVzYyI+CiAgICAgIEZvdXIgc3RlcHMuIFplcm8gdGVtcGxhdGVzLiBPbmUgQUkgdGhhdCB0aGlua3MgdGhlIHdheSBncmVhdCB0cmF2ZWwgcGxhbm5lcnMgZG8g4oCUIGV4Y2VwdCBpdCBuZXZlciBzbGVlcHMsIG5ldmVyIGZvcmdldHMgeW91ciBwcmVmZXJlbmNlcywgYW5kIGxlYXJucyB3aXRoIGV2ZXJ5IGNvbnZlcnNhdGlvbi4KICAgIDwvcD4KICA8L2Rpdj4KICA8ZGl2IGNsYXNzPSJob3ctc3RlcHMiPgogICAgPGRpdiBjbGFzcz0ic3RlcCByZXZlYWwiPgogICAgICA8ZGl2IGNsYXNzPSJzdGVwLW51bSI+MDE8L2Rpdj4KICAgICAgPGRpdiBjbGFzcz0ic3RlcC10aXRsZSI+VGVsbCBVcyBXaG8gWW91IEFyZTwvZGl2PgogICAgICA8ZGl2IGNsYXNzPSJzdGVwLWRlc2MiPlNoYXJlIHlvdXIgdHJhdmVsIGlkZW50aXR5IOKAlCBzdHlsZSwgYnVkZ2V0LCBpbnRlcmVzdHMsIHBhY2UuIE91ciBBSSBidWlsZHMgYSBsaXZpbmcgcHJvZmlsZSB0aGF0IGdldHMgc2hhcnBlciB3aXRoIGV2ZXJ5IHRyaXAuPC9kaXY+CiAgICA8L2Rpdj4KICAgIDxkaXYgY2xhc3M9InN0ZXAgcmV2ZWFsIj4KICAgICAgPGRpdiBjbGFzcz0ic3RlcC1udW0iPjAyPC9kaXY+CiAgICAgIDxkaXYgY2xhc3M9InN0ZXAtdGl0bGUiPkRlc2NyaWJlIFlvdXIgRHJlYW08L2Rpdj4KICAgICAgPGRpdiBjbGFzcz0ic3RlcC1kZXNjIj5TcGVhayBuYXR1cmFsbHkuICIzIGRheXMgaW4gR29hLCBub3QgdG9vIHRvdXJpc3R5LCDigrkzMEsgYnVkZ2V0LiIgT3VyIEFJIHVuZGVyc3RhbmRzIG51YW5jZSwgbm90IGp1c3Qga2V5d29yZHMuPC9kaXY+CiAgICA8L2Rpdj4KICAgIDxkaXYgY2xhc3M9InN0ZXAgcmV2ZWFsIj4KICAgICAgPGRpdiBjbGFzcz0ic3RlcC1udW0iPjAzPC9kaXY+CiAgICAgIDxkaXYgY2xhc3M9InN0ZXAtdGl0bGUiPkFJIERlc2lnbnMgWW91ciBKb3VybmV5PC9kaXY+CiAgICAgIDxkaXYgY2xhc3M9InN0ZXAtZGVzYyI+SW4gc2Vjb25kcywgcmVjZWl2ZSBhIGZ1bGx5IHBlcnNvbmFsaXNlZCBpdGluZXJhcnkg4oCUIHN0YXlzLCBleHBlcmllbmNlcywgcm91dGVzIOKAlCBjdXJhdGVkIGZyb20gb3VyIHZlcmlmaWVkIHBhcnRuZXIgbmV0d29yay48L2Rpdj4KICAgIDwvZGl2PgogICAgPGRpdiBjbGFzcz0ic3RlcCByZXZlYWwiPgogICAgICA8ZGl2IGNsYXNzPSJzdGVwLW51bSI+MDQ8L2Rpdj4KICAgICAgPGRpdiBjbGFzcz0ic3RlcC10aXRsZSI+Qm9vayAmIEdvPC9kaXY+CiAgICAgIDxkaXYgY2xhc3M9InN0ZXAtZGVzYyI+Q29uZmlybSB5b3VyIHRyaXAgd2l0aCBvbmUgY2xpY2suIFlvdXIgQUkgY29uY2llcmdlIGhhbmRsZXMgZXZlcnkgZGV0YWlsIOKAlCBhbmQgc3RheXMgb24gY2FsbCBmb3IgdGhlIGVudGlyZSBqb3VybmV5LjwvZGl2PgogICAgPC9kaXY+CiAgPC9kaXY+Cjwvc2VjdGlvbj4KCjwhLS0g4pSA4pSAIERFU1RJTkFUSU9OUyDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIAgLS0+CjxzZWN0aW9uIGNsYXNzPSJkZXN0aW5hdGlvbnMgcmV2ZWFsIiBpZD0iZGVzdGluYXRpb25zIj4KICA8ZGl2IGNsYXNzPSJkZXN0aW5hdGlvbnMtaGVhZGVyIj4KICAgIDxkaXY+CiAgICAgIDxkaXYgY2xhc3M9InNlY3Rpb24tbGFiZWwiPkhhbmRwaWNrZWQgSm91cm5leXM8L2Rpdj4KICAgICAgPGgyIGNsYXNzPSJzZWN0aW9uLWhlYWRsaW5lIj4KICAgICAgICBQbGFjZXMgdGhhdDxicj4KICAgICAgICA8ZW0+bW92ZTwvZW0+IHBlb3BsZS4KICAgICAgPC9oMj4KICAgIDwvZGl2PgogICAgPGEgaHJlZj0iamF2YXNjcmlwdDp2b2lkKDApIiBvbmNsaWNrPSJvcGVuVFRUU2lnbmluKCkiIGNsYXNzPSJkZXN0aW5hdGlvbnMtbGluayI+RXhwbG9yZSBhbGwgZGVzdGluYXRpb25zIOKGkjwvYT4KICA8L2Rpdj4KICA8ZGl2IGNsYXNzPSJkZXN0LWdyaWQiPgogICAgPGRpdiBjbGFzcz0iZGVzdC1jYXJkIj4KICAgICAgPGltZyBjbGFzcz0iZGVzdC1pbWciIHNyYz0iaHR0cHM6Ly9pbWFnZXMudW5zcGxhc2guY29tL3Bob3RvLTE1MTg1MDk1NjI5MDQtZTdlZjk5Y2RjYzg2P3c9OTAwJnE9ODAiIGFsdD0iUmFqYXN0aGFuIj4KICAgICAgPGRpdiBjbGFzcz0iZGVzdC1vdmVybGF5Ij48L2Rpdj4KICAgICAgPGRpdiBjbGFzcz0iZGVzdC1haS10YWciPkFJIEN1cmF0ZWQ8L2Rpdj4KICAgICAgPGRpdiBjbGFzcz0iZGVzdC1pbmZvIj4KICAgICAgICA8ZGl2IGNsYXNzPSJkZXN0LW5hbWUiPlJhamFzdGhhbjwvZGl2PgogICAgICAgIDxkaXYgY2xhc3M9ImRlc3Qtc3ViIj5QYWxhY2VzIMK3IERlc2VydCDCtyBIZXJpdGFnZSDCtyBDdWx0dXJlPC9kaXY+CiAgICAgIDwvZGl2PgogICAgPC9kaXY+CiAgICA8ZGl2IGNsYXNzPSJkZXN0LWNhcmQiPgogICAgICA8aW1nIGNsYXNzPSJkZXN0LWltZyIgc3JjPSJodHRwczovL2ltYWdlcy51bnNwbGFzaC5jb20vcGhvdG8tMTYwMjIxNjA1NjA5Ni0zYjQwY2MwYzk5NDQ/dz02MDAmcT04MCIgYWx0PSJLZXJhbGEiPgogICAgICA8ZGl2IGNsYXNzPSJkZXN0LW92ZXJsYXkiPjwvZGl2PgogICAgICA8ZGl2IGNsYXNzPSJkZXN0LWFpLXRhZyI+QUkgQ3VyYXRlZDwvZGl2PgogICAgICA8ZGl2IGNsYXNzPSJkZXN0LWluZm8iPgogICAgICAgIDxkaXYgY2xhc3M9ImRlc3QtbmFtZSI+S2VyYWxhPC9kaXY+CiAgICAgICAgPGRpdiBjbGFzcz0iZGVzdC1zdWIiPkJhY2t3YXRlcnMgwrcgV2VsbG5lc3MgwrcgTmF0dXJlPC9kaXY+CiAgICAgIDwvZGl2PgogICAgPC9kaXY+CiAgICA8ZGl2IGNsYXNzPSJkZXN0LWNhcmQiPgogICAgICA8aW1nIGNsYXNzPSJkZXN0LWltZyIgc3JjPSJodHRwczovL2ltYWdlcy51bnNwbGFzaC5jb20vcGhvdG8tMTUxMjM0Mzg3OTc4NC1hOTYwYmY0MGU3ZjI/dz02MDAmcT04MCIgYWx0PSJHb2EiPgogICAgICA8ZGl2IGNsYXNzPSJkZXN0LW92ZXJsYXkiPjwvZGl2PgogICAgICA8ZGl2IGNsYXNzPSJkZXN0LWFpLXRhZyI+QUkgQ3VyYXRlZDwvZGl2PgogICAgICA8ZGl2IGNsYXNzPSJkZXN0LWluZm8iPgogICAgICAgIDxkaXYgY2xhc3M9ImRlc3QtbmFtZSI+R29hPC9kaXY+CiAgICAgICAgPGRpdiBjbGFzcz0iZGVzdC1zdWIiPkJlYWNoZXMgwrcgSGlkZGVuIEdlbXMgwrcgTmlnaHRsaWZlPC9kaXY+CiAgICAgIDwvZGl2PgogICAgPC9kaXY+CiAgICA8ZGl2IGNsYXNzPSJkZXN0LWNhcmQiPgogICAgICA8aW1nIGNsYXNzPSJkZXN0LWltZyIgc3JjPSJodHRwczovL2ltYWdlcy51bnNwbGFzaC5jb20vcGhvdG8tMTQ5NDU0ODE2MjQ5NC0zODRiYmE0YWI5OTk/dz02MDAmcT04MCIgYWx0PSJTcGl0aSBWYWxsZXkiPgogICAgICA8ZGl2IGNsYXNzPSJkZXN0LW92ZXJsYXkiPjwvZGl2PgogICAgICA8ZGl2IGNsYXNzPSJkZXN0LWFpLXRhZyI+QUkgQ3VyYXRlZDwvZGl2PgogICAgICA8ZGl2IGNsYXNzPSJkZXN0LWluZm8iPgogICAgICAgIDxkaXYgY2xhc3M9ImRlc3QtbmFtZSI+U3BpdGkgVmFsbGV5PC9kaXY+CiAgICAgICAgPGRpdiBjbGFzcz0iZGVzdC1zdWIiPk1vdW50YWlucyDCtyBPZmYtYmVhdCDCtyBBZHZlbnR1cmU8L2Rpdj4KICAgICAgPC9kaXY+CiAgICA8L2Rpdj4KICAgIDxkaXYgY2xhc3M9ImRlc3QtY2FyZCI+CiAgICAgIDxpbWcgY2xhc3M9ImRlc3QtaW1nIiBzcmM9Imh0dHBzOi8vaW1hZ2VzLnVuc3BsYXNoLmNvbS9waG90by0xNDM3NzE5NDE3MDMyLTg1OTVmZDllOWRjNj93PTYwMCZxPTgwIiBhbHQ9Ik1hbGRpdmVzIj4KICAgICAgPGRpdiBjbGFzcz0iZGVzdC1vdmVybGF5Ij48L2Rpdj4KICAgICAgPGRpdiBjbGFzcz0iZGVzdC1haS10YWciPkFJIEN1cmF0ZWQ8L2Rpdj4KICAgICAgPGRpdiBjbGFzcz0iZGVzdC1pbmZvIj4KICAgICAgICA8ZGl2IGNsYXNzPSJkZXN0LW5hbWUiPk1hbGRpdmVzPC9kaXY+CiAgICAgICAgPGRpdiBjbGFzcz0iZGVzdC1zdWIiPkx1eHVyeSDCtyBPdmVyd2F0ZXIgwrcgUm9tYW5jZTwvZGl2PgogICAgICA8L2Rpdj4KICAgIDwvZGl2PgogIDwvZGl2Pgo8L3NlY3Rpb24+Cgo8IS0tIOKUgOKUgCBBSSBERU1PIOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgCAtLT4KPHNlY3Rpb24gY2xhc3M9ImFpLWRlbW8gcmV2ZWFsIiBpZD0iYWkiPgogIDxkaXYgY2xhc3M9ImFpLWRlbW8tdGV4dCI+CiAgICA8ZGl2IGNsYXNzPSJzZWN0aW9uLWxhYmVsIj5UaGUgSW50ZWxsaWdlbmNlPC9kaXY+CiAgICA8aDIgY2xhc3M9InNlY3Rpb24taGVhZGxpbmUiPgogICAgICBUaGUgb25seTxicj4KICAgICAgY29uY2llcmdlIHRoYXQ8YnI+CiAgICAgIDxlbT5uZXZlciBzbGVlcHMuPC9lbT4KICAgIDwvaDI+CiAgICA8cD4KICAgICAgUG93ZXJlZCBieSBDbGF1ZGUgQUkg4oCUIG9uZSBvZiB0aGUgd29ybGQncyBtb3N0IGNhcGFibGUgbGFuZ3VhZ2UgbW9kZWxzIOKAlCBUVFQncyBjb25jaWVyZ2UgdW5kZXJzdGFuZHMgY29udGV4dCwgbnVhbmNlLCBhbmQgYnVkZ2V0LiBJdCdzIG5vdCBhIGNoYXRib3QuIEl0J3MgYSB0cmF2ZWwgbWluZC4KICAgIDwvcD4KICAgIDxhIGhyZWY9ImphdmFzY3JpcHQ6dm9pZCgwKSIgb25jbGljaz0ib3BlblRUVFNpZ25pbigpIiBjbGFzcz0iYnRuLXByaW1hcnkiPjxzcGFuPlRhbGsgdG8gdGhlIEFJPC9zcGFuPjwvYT4KICA8L2Rpdj4KICA8ZGl2IGNsYXNzPSJhaS13aW5kb3ciPgogICAgPGRpdiBjbGFzcz0iYWktd2luZG93LWhlYWRlciI+CiAgICAgIDxkaXYgY2xhc3M9ImFpLXdpbmRvdy1kb3QiPjwvZGl2PgogICAgICBUVFQgQUkgQ29uY2llcmdlIOKAlCBMaXZlCiAgICA8L2Rpdj4KICAgIDxkaXYgY2xhc3M9ImFpLW1lc3NhZ2VzIj4KICAgICAgPGRpdiBjbGFzcz0iYWktbXNnIHVzZXIiPgogICAgICAgIDxkaXYgY2xhc3M9ImFpLW1zZy1hdmF0YXIgdXNlci1hdmF0YXIiPkE8L2Rpdj4KICAgICAgICA8ZGl2IGNsYXNzPSJhaS1tc2ctYnViYmxlIj5QbGFuIGEgNS1kYXkgaG9uZXltb29uIGluIE1hbGRpdmVzIHVuZGVyIOKCuTIgbGFraHMgZm9yIHR3by48L2Rpdj4KICAgICAgPC9kaXY+CiAgICAgIDxkaXYgY2xhc3M9ImFpLW1zZyI+CiAgICAgICAgPGRpdiBjbGFzcz0iYWktbXNnLWF2YXRhciBhaS1hdmF0YXIiPlRUVDwvZGl2PgogICAgICAgIDxkaXYgY2xhc3M9ImFpLW1zZy1idWJibGUiPgogICAgICAgICAgUGVyZmVjdCB0aW1pbmcg4oCUIEknbGwgZGVzaWduIHNvbWV0aGluZyB0cnVseSBzcGVjaWFsLjxicj48YnI+CiAgICAgICAgICBJJ20gdGhpbmtpbmcgPHN0cm9uZz5GdXNoaWZhcnUgb3IgTWVlcnUgSXNsYW5kPC9zdHJvbmc+IOKAlCBib3RoIG9mZmVyIG92ZXJ3YXRlciB2aWxsYXMgd2l0aCBwcml2YXRlIHBvb2xzIGF0IOKCuTE4LTIyay9uaWdodC4gRm9yIDUgbmlnaHRzLCB0aGF0J3Mgcm91Z2hseSDigrk5NS0xMTBrIG9uIHN0YXlzLCBsZWF2aW5nIOKCuTkwayBmb3IgZmxpZ2h0cywgZXhwZXJpZW5jZXMsIGFuZCBkaW5pbmcuPGJyPjxicj4KICAgICAgICAgIFNoYWxsIEkgYnVpbGQgdGhlIGZ1bGwgaXRpbmVyYXJ5IHdpdGggYSBzdW5zZXQgZG9scGhpbiBjcnVpc2Ugb24gRGF5IDI/CiAgICAgICAgPC9kaXY+CiAgICAgIDwvZGl2PgogICAgICA8ZGl2IGNsYXNzPSJhaS1tc2cgdXNlciI+CiAgICAgICAgPGRpdiBjbGFzcz0iYWktbXNnLWF2YXRhciB1c2VyLWF2YXRhciI+QTwvZGl2PgogICAgICAgIDxkaXYgY2xhc3M9ImFpLW1zZy1idWJibGUiPlllcyEgQW5kIHdlIGxvdmUgc25vcmtlbGxpbmcuIEFkZCB0aGF0IHRvby48L2Rpdj4KICAgICAgPC9kaXY+CiAgICAgIDxkaXYgY2xhc3M9ImFpLW1zZyI+CiAgICAgICAgPGRpdiBjbGFzcz0iYWktbXNnLWF2YXRhciBhaS1hdmF0YXIiPlRUVDwvZGl2PgogICAgICAgIDxkaXYgY2xhc3M9ImFpLW1zZy1idWJibGUiPgogICAgICAgICAgPHN0cm9uZz5Zb3VyIDUtRGF5IE1hbGRpdmVzIEl0aW5lcmFyeSDinKY8L3N0cm9uZz48YnI+CiAgICAgICAgICBEYXkgMSDCtyBBcnJpdmFsICsgc3Vuc2V0IHdlbGNvbWUgZGlubmVyIG9uIHRoZSB3YXRlcjxicj4KICAgICAgICAgIERheSAyIMK3IERvbHBoaW4gY3J1aXNlIGF0IGR1c2sgKyBzdGFyZ2F6aW5nPGJyPgogICAgICAgICAgRGF5IDMgwrcgRnVsbC1kYXkgc25vcmtlbGxpbmcg4oCUIE1hbnRhIFBvaW50ICsgcmVlZjxicj4KICAgICAgICAgIERheSA0IMK3IFNwYSBkYXkgKyBwcml2YXRlIGJlYWNoIHBpY25pYzxicj4KICAgICAgICAgIERheSA1IMK3IERlcGFydHVyZSB3aXRoIGEgbW9ybmluZyBrYXlhazxicj48YnI+CiAgICAgICAgICBUb3RhbCBlc3RpbWF0ZTogPHN0cm9uZz7igrkxLDg4LDAwMDwvc3Ryb25nPiDCtyBGbGlnaHRzIGluY2x1ZGVkIOKckwogICAgICAgIDwvZGl2PgogICAgICA8L2Rpdj4KICAgIDwvZGl2PgogIDwvZGl2Pgo8L3NlY3Rpb24+Cgo8IS0tIOKUgOKUgCBBSSBGRUFUVVJFUyDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIAgLS0+CjxzZWN0aW9uIGNsYXNzPSJmZWF0dXJlcyByZXZlYWwiPgogIDxkaXYgY2xhc3M9InNlY3Rpb24tbGFiZWwiPldoYXQgTWFrZXMgVFRUIERpZmZlcmVudDwvZGl2PgogIDxoMiBjbGFzcz0ic2VjdGlvbi1oZWFkbGluZSIgc3R5bGU9ImNvbG9yOnZhcigtLW9mZi1ibGFjaykiPgogICAgSW50ZWxsaWdlbmNlIHRoYXQ8YnI+CiAgICA8ZW0+dHJhdmVscyB3aXRoIHlvdS48L2VtPgogIDwvaDI+CiAgPGRpdiBjbGFzcz0iZmVhdHVyZXMtZ3JpZCI+CiAgICA8ZGl2IGNsYXNzPSJmZWF0dXJlLWNhcmQgcmV2ZWFsIj4KICAgICAgPHNwYW4gY2xhc3M9ImZlYXR1cmUtaWNvbiI+8J+noDwvc3Bhbj4KICAgICAgPGRpdiBjbGFzcz0iZmVhdHVyZS10aXRsZSI+QUkgVGhhdCBLbm93cyBZb3U8L2Rpdj4KICAgICAgPGRpdiBjbGFzcz0iZmVhdHVyZS1kZXNjIj5FdmVyeSBjb252ZXJzYXRpb24gdGVhY2hlcyB0aGUgQUkgbW9yZSBhYm91dCBob3cgeW91IHRyYXZlbC4gWW91ciBzZWNvbmQgdHJpcCBwbGFuIGlzIHNoYXJwZXIgdGhhbiB5b3VyIGZpcnN0LiBZb3VyIHRlbnRoIGlzIHBlcmZlY3QuPC9kaXY+CiAgICA8L2Rpdj4KICAgIDxkaXYgY2xhc3M9ImZlYXR1cmUtY2FyZCByZXZlYWwiPgogICAgICA8c3BhbiBjbGFzcz0iZmVhdHVyZS1pY29uIj7wn5KOPC9zcGFuPgogICAgICA8ZGl2IGNsYXNzPSJmZWF0dXJlLXRpdGxlIj5DdXJhdGVkLCBOb3QgQWdncmVnYXRlZDwvZGl2PgogICAgICA8ZGl2IGNsYXNzPSJmZWF0dXJlLWRlc2MiPjMwMCsgaGFuZC12ZXJpZmllZCBwYXJ0bmVycy4gTm8gcGFpZCByYW5raW5ncy4gTm8gc3BvbnNvcmVkIGxpc3RpbmdzLiBFdmVyeSByZWNvbW1lbmRhdGlvbiBpcyBlYXJuZWQsIG5vdCBib3VnaHQuPC9kaXY+CiAgICA8L2Rpdj4KICAgIDxkaXYgY2xhc3M9ImZlYXR1cmUtY2FyZCByZXZlYWwiPgogICAgICA8c3BhbiBjbGFzcz0iZmVhdHVyZS1pY29uIj7imqE8L3NwYW4+CiAgICAgIDxkaXYgY2xhc3M9ImZlYXR1cmUtdGl0bGUiPkluc3RhbnQuIEFsd2F5cy48L2Rpdj4KICAgICAgPGRpdiBjbGFzcz0iZmVhdHVyZS1kZXNjIj5GdWxsIGl0aW5lcmFyeSBpbiBzZWNvbmRzLiBCb29raW5nIGNvbmZpcm1hdGlvbiBpbiBtaW51dGVzLiBTdXBwb3J0IGF2YWlsYWJsZSAyNC83LiBCZWNhdXNlIGdyZWF0IHRyYXZlbCBkb2Vzbid0IHdhaXQgZm9yIG9mZmljZSBob3Vycy48L2Rpdj4KICAgIDwvZGl2PgogICAgPGRpdiBjbGFzcz0iZmVhdHVyZS1jYXJkIHJldmVhbCI+CiAgICAgIDxzcGFuIGNsYXNzPSJmZWF0dXJlLWljb24iPvCfl7rvuI88L3NwYW4+CiAgICAgIDxkaXYgY2xhc3M9ImZlYXR1cmUtdGl0bGUiPkhpZGRlbiBJbmRpYSwgVW5sb2NrZWQ8L2Rpdj4KICAgICAgPGRpdiBjbGFzcz0iZmVhdHVyZS1kZXNjIj5Mb2NhbCBrbm93bGVkZ2UgYmFrZWQgaW50byB0aGUgQUkuIFRoZSB2aWxsYWdlIGNhZsOpIG9ubHkgbG9jYWxzIGtub3cuIFRoZSB0cmFpbCB0aGF0IGRvZXNuJ3QgYXBwZWFyIG9uIEdvb2dsZS4gVGhlIHJlc29ydCB3aXRob3V0IGEgYmlsbGJvYXJkLjwvZGl2PgogICAgPC9kaXY+CiAgICA8ZGl2IGNsYXNzPSJmZWF0dXJlLWNhcmQgcmV2ZWFsIj4KICAgICAgPHNwYW4gY2xhc3M9ImZlYXR1cmUtaWNvbiI+4oK5PC9zcGFuPgogICAgICA8ZGl2IGNsYXNzPSJmZWF0dXJlLXRpdGxlIj5FdmVyeSBCdWRnZXQsIEVsZXZhdGVkPC9kaXY+CiAgICAgIDxkaXYgY2xhc3M9ImZlYXR1cmUtZGVzYyI+RnJvbSDigrkxNUsgd2Vla2VuZGVycyB0byDigrk1TCBsdXh1cnkgZXNjYXBlcyDigJQgdGhlIEFJIG9wdGltaXNlcyBldmVyeSBydXBlZSBzbyB5b3UgbmV2ZXIgb3ZlcnBheSBmb3IgYXZlcmFnZSBvciB1bmRlcnNwZW5kIG9uIGV4dHJhb3JkaW5hcnkuPC9kaXY+CiAgICA8L2Rpdj4KICAgIDxkaXYgY2xhc3M9ImZlYXR1cmUtY2FyZCByZXZlYWwiPgogICAgICA8c3BhbiBjbGFzcz0iZmVhdHVyZS1pY29uIj7wn6SdPC9zcGFuPgogICAgICA8ZGl2IGNsYXNzPSJmZWF0dXJlLXRpdGxlIj5IdW1hbiBXaGVuIEl0IE1hdHRlcnM8L2Rpdj4KICAgICAgPGRpdiBjbGFzcz0iZmVhdHVyZS1kZXNjIj5PdXIgY29uY2llcmdlIHRlYW0gYmFja3MgZXZlcnkgQUkgcGxhbi4gRm9yIGNvbXBsZXggYm9va2luZ3MsIHNwZWNpYWwgb2NjYXNpb25zLCBvciBwZWFjZSBvZiBtaW5kIOKAlCBhIHJlYWwgVFRUIGV4cGVydCBpcyBhbHdheXMgb25lIG1lc3NhZ2UgYXdheS48L2Rpdj4KICAgIDwvZGl2PgogIDwvZGl2Pgo8L3NlY3Rpb24+Cgo8IS0tIOKUgOKUgCBURVNUSU1PTklBTFMg4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSA4pSAIC0tPgo8c2VjdGlvbiBjbGFzcz0idGVzdGltb25pYWxzIHJldmVhbCIgaWQ9InN0b3JpZXMiPgogIDxkaXYgY2xhc3M9InNlY3Rpb24tbGFiZWwiPlRyYXZlbGxlciBTdG9yaWVzPC9kaXY+CiAgPGgyIGNsYXNzPSJzZWN0aW9uLWhlYWRsaW5lIj4KICAgIEJ1aWx0IG9uPGJyPgogICAgPGVtPnJlYWwgam91cm5leXMuPC9lbT4KICA8L2gyPgogIDxkaXYgY2xhc3M9InRlc3RpbW9uaWFscy1ncmlkIj4KICAgIDxkaXYgY2xhc3M9InRjYXJkIHJldmVhbCI+CiAgICAgIDxkaXYgY2xhc3M9InRjYXJkLXN0YXJzIj7imIXimIXimIXimIXimIU8L2Rpdj4KICAgICAgPGRpdiBjbGFzcz0idGNhcmQtdGV4dCI+IkkgZGVzY3JpYmVkIG15IGRyZWFtIHRyaXAgaW4gb25lIG1lc3NhZ2UuIFRUVCBidWlsdCBhbiBpdGluZXJhcnkgdGhhdCBmZWx0IGxpa2UgaXQgd2FzIGRlc2lnbmVkIGJ5IHNvbWVvbmUgd2hvJ2Qga25vd24gbWUgZm9yIHllYXJzLiBVdHRhcmFraGFuZCB3YXMgZmxhd2xlc3MuIjwvZGl2PgogICAgICA8ZGl2IGNsYXNzPSJ0Y2FyZC1hdXRob3IiPgogICAgICAgIDxkaXYgY2xhc3M9InRjYXJkLWF2YXRhciI+UDwvZGl2PgogICAgICAgIDxkaXY+CiAgICAgICAgICA8ZGl2IGNsYXNzPSJ0Y2FyZC1uYW1lIj5Qcml5YSBNZWh0YTwvZGl2PgogICAgICAgICAgPGRpdiBjbGFzcz0idGNhcmQtbWV0YSI+RGVsaGkgwrcgVXR0YXJha2hhbmQgU29sbyBUcmVrPC9kaXY+CiAgICAgICAgPC9kaXY+CiAgICAgIDwvZGl2PgogICAgPC9kaXY+CiAgICA8ZGl2IGNsYXNzPSJ0Y2FyZCByZXZlYWwiPgogICAgICA8ZGl2IGNsYXNzPSJ0Y2FyZC1zdGFycyI+4piF4piF4piF4piF4piFPC9kaXY+CiAgICAgIDxkaXYgY2xhc3M9InRjYXJkLXRleHQiPiJPdXIgUmFqYXN0aGFuIGhvbmV5bW9vbiB3YXMgYmV5b25kIHdoYXQgd2UgaW1hZ2luZWQuIFRoZSBBSSBmb3VuZCBhIGhlcml0YWdlIGhhdmVsaSBpbiBKb2RocHVyIHRoYXQgd2Fzbid0IG9uIGFueSBhcHAuIE91ciBmcmllbmRzIGFyZSBzdGlsbCBhc2tpbmcgaG93IHdlIGZvdW5kIGl0LiI8L2Rpdj4KICAgICAgPGRpdiBjbGFzcz0idGNhcmQtYXV0aG9yIj4KICAgICAgICA8ZGl2IGNsYXNzPSJ0Y2FyZC1hdmF0YXIiPlI8L2Rpdj4KICAgICAgICA8ZGl2PgogICAgICAgICAgPGRpdiBjbGFzcz0idGNhcmQtbmFtZSI+Um9oYW4gJiBBbmFueWE8L2Rpdj4KICAgICAgICAgIDxkaXYgY2xhc3M9InRjYXJkLW1ldGEiPk11bWJhaSDCtyBSYWphc3RoYW4gSG9uZXltb29uPC9kaXY+CiAgICAgICAgPC9kaXY+CiAgICAgIDwvZGl2PgogICAgPC9kaXY+CiAgICA8ZGl2IGNsYXNzPSJ0Y2FyZCByZXZlYWwiPgogICAgICA8ZGl2IGNsYXNzPSJ0Y2FyZC1zdGFycyI+4piF4piF4piF4piF4piFPC9kaXY+CiAgICAgIDxkaXYgY2xhc3M9InRjYXJkLXRleHQiPiJBcyBhIGZyZXF1ZW50IGJ1c2luZXNzIHRyYXZlbGxlciwgSSBuZWVkZWQgZWZmaWNpZW5jeS4gVFRUIHBsYW5uZWQgbXkgR29hIGxvbmcgd2Vla2VuZCBpbiB1bmRlciAyIG1pbnV0ZXMuIFRoZSBoaWRkZW4gYmVhY2ggaXQgcmVjb21tZW5kZWQgaGFkIG1heWJlIDEwIHBlb3BsZSBvbiBpdC4iPC9kaXY+CiAgICAgIDxkaXYgY2xhc3M9InRjYXJkLWF1dGhvciI+CiAgICAgICAgPGRpdiBjbGFzcz0idGNhcmQtYXZhdGFyIj5WPC9kaXY+CiAgICAgICAgPGRpdj4KICAgICAgICAgIDxkaXYgY2xhc3M9InRjYXJkLW5hbWUiPlZpa3JhbSBTaW5naDwvZGl2PgogICAgICAgICAgPGRpdiBjbGFzcz0idGNhcmQtbWV0YSI+QmVuZ2FsdXJ1IMK3IEdvYSBXZWVrZW5kPC9kaXY+CiAgICAgICAgPC9kaXY+CiAgICAgIDwvZGl2PgogICAgPC9kaXY+CiAgPC9kaXY+Cjwvc2VjdGlvbj4KCjwhLS0g4pSA4pSAIENUQSDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIDilIAgLS0+CjxkaXYgY2xhc3M9ImN0YS1mdWxsIj4KICA8ZGl2IGNsYXNzPSJjdGEtYmciPjwvZGl2PgogIDxkaXYgY2xhc3M9ImN0YS1jb250ZW50IHJldmVhbCI+CiAgICA8aDIgY2xhc3M9ImN0YS1oZWFkbGluZSI+CiAgICAgIFlvdXIgbmV4dCBqb3VybmV5PGJyPgogICAgICBiZWdpbnMgd2l0aCBhPGJyPgogICAgICA8ZW0+Y29udmVyc2F0aW9uLjwvZW0+CiAgICA8L2gyPgogICAgPHAgY2xhc3M9ImN0YS1zdWIiPk5vIGZvcm1zLiBObyB3YWl0aW5nLiBKdXN0IHRlbGwgdGhlIEFJIHdoZXJlIHlvdSB3YW50IHRvIGdvIOKAlCBhbmQgbGV0IEluZGlhJ3MgbW9zdCBpbnRlbGxpZ2VudCB0cmF2ZWwgY29uY2llcmdlIHRha2UgaXQgZnJvbSB0aGVyZS48L3A+CiAgICA8ZGl2IGNsYXNzPSJjdGEtYWN0aW9ucyI+CiAgICAgIDxhIGhyZWY9ImphdmFzY3JpcHQ6dm9pZCgwKSIgb25jbGljaz0ib3BlblRUVFNpZ25pbigpIiBjbGFzcz0iYnRuLXByaW1hcnkiPjxzcGFuPlN0YXJ0IGZvciBGcmVlPC9zcGFuPjwvYT4KICAgICAgPGEgaHJlZj0iamF2YXNjcmlwdDp2b2lkKDApIiBvbmNsaWNrPSJvcGVuVFRUU2lnbmluKCkiIGNsYXNzPSJidG4tZ2hvc3QiIHN0eWxlPSJjb2xvcjpyZ2JhKDI1NSwyNTUsMjU1LDAuNSkiPkxlYXJuIG1vcmUg4oaSPC9hPgogICAgPC9kaXY+CiAgPC9kaXY+CjwvZGl2PgoKPCEtLSDilIDilIAgRk9PVEVSIOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgOKUgCAtLT4KPGZvb3Rlcj4KICA8ZGl2IGNsYXNzPSJmb290ZXItdG9wIj4KICAgIDxkaXY+CiAgICAgIDxkaXYgY2xhc3M9ImZvb3Rlci1icmFuZC1uYW1lIj5UVFQ8L2Rpdj4KICAgICAgPGRpdiBjbGFzcz0iZm9vdGVyLWJyYW5kLXRhZ2xpbmUiPkluZGlhJ3MgZmlyc3QgQWdlbnRpYyBBSSBsdXh1cnkgdHJhdmVsIGNvbmNpZXJnZS4gQnVpbHQgZm9yIGV4cGxvcmVycyB3aG8gcmVmdXNlIG9yZGluYXJ5IGpvdXJuZXlzLjwvZGl2PgogICAgICA8ZGl2IGNsYXNzPSJmb290ZXItY29udGFjdCI+CiAgICAgICAgPGEgaHJlZj0ibWFpbHRvOmNvbmNpZXJnZUB0aGV0cmlwdGhlb3J5LmNvbSI+4pyJIGNvbmNpZXJnZUB0aGV0cmlwdGhlb3J5LmNvbTwvYT4KICAgICAgICA8YSBocmVmPSIjIj7wn5ONIEd1cnVncmFtLCBIYXJ5YW5hLCBJbmRpYTwvYT4KICAgICAgPC9kaXY+CiAgICA8L2Rpdj4KICAgIDxkaXY+CiAgICAgIDxkaXYgY2xhc3M9ImZvb3Rlci1jb2wtdGl0bGUiPlByb2R1Y3Q8L2Rpdj4KICAgICAgPGRpdiBjbGFzcz0iZm9vdGVyLWNvbCI+CiAgICAgICAgPGEgaHJlZj0iamF2YXNjcmlwdDp2b2lkKDApIiBvbmNsaWNrPSJvcGVuVFRUU2lnbmluKCkiPlBsYW4gYSBUcmlwPC9hPgogICAgICAgIDxhIGhyZWY9ImphdmFzY3JpcHQ6dm9pZCgwKSIgb25jbGljaz0ib3BlblRUVFNpZ25pbigpIj5BSSBDb25jaWVyZ2U8L2E+CiAgICAgICAgPGEgaHJlZj0iamF2YXNjcmlwdDp2b2lkKDApIiBvbmNsaWNrPSJvcGVuVFRUU2lnbmluKCkiPk1hcmtldHBsYWNlPC9hPgogICAgICAgIDxhIGhyZWY9ImphdmFzY3JpcHQ6dm9pZCgwKSIgb25jbGljaz0ib3BlblRUVFNpZ25pbigpIj5Gb3IgUGFydG5lcnM8L2E+CiAgICAgIDwvZGl2PgogICAgPC9kaXY+CiAgICA8ZGl2PgogICAgICA8ZGl2IGNsYXNzPSJmb290ZXItY29sLXRpdGxlIj5Db21wYW55PC9kaXY+CiAgICAgIDxkaXYgY2xhc3M9ImZvb3Rlci1jb2wiPgogICAgICAgIDxhIGhyZWY9IiMiPk91ciBTdG9yeTwvYT4KICAgICAgICA8YSBocmVmPSIjIj5UaGUgQUk8L2E+CiAgICAgICAgPGEgaHJlZj0iIyI+UGFydG5lciBQb3J0YWw8L2E+CiAgICAgICAgPGEgaHJlZj0iIyI+Q2FyZWVyczwvYT4KICAgICAgPC9kaXY+CiAgICA8L2Rpdj4KICAgIDxkaXY+CiAgICAgIDxkaXYgY2xhc3M9ImZvb3Rlci1jb2wtdGl0bGUiPkxlZ2FsPC9kaXY+CiAgICAgIDxkaXYgY2xhc3M9ImZvb3Rlci1jb2wiPgogICAgICAgIDxhIGhyZWY9IiMiIG9uY2xpY2s9ImRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCd0dHQtcHJpdmFjeS1tb2RhbCcpJiYoZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3R0dC1wcml2YWN5LW1vZGFsJykuc3R5bGUuZGlzcGxheT0nZmxleCcpIj5Qcml2YWN5IFBvbGljeTwvYT4KICAgICAgICA8YSBocmVmPSIjIiBvbmNsaWNrPSJkb2N1bWVudC5nZXRFbGVtZW50QnlJZCgndHR0LXRlcm1zLW1vZGFsJykmJihkb2N1bWVudC5nZXRFbGVtZW50QnlJZCgndHR0LXRlcm1zLW1vZGFsJykuc3R5bGUuZGlzcGxheT0nZmxleCcpIj5UZXJtcyBvZiBTZXJ2aWNlPC9hPgogICAgICAgIDxhIGhyZWY9IiMiIG9uY2xpY2s9ImRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCd0dHQtcmVmdW5kLW1vZGFsJykmJihkb2N1bWVudC5nZXRFbGVtZW50QnlJZCgndHR0LXJlZnVuZC1tb2RhbCcpLnN0eWxlLmRpc3BsYXk9J2ZsZXgnKSI+Q2FuY2VsbGF0aW9uICYgUmVmdW5kPC9hPgogICAgICA8L2Rpdj4KICAgIDwvZGl2PgogIDwvZGl2PgogIDxkaXYgY2xhc3M9ImZvb3Rlci1ib3R0b20iPgogICAgPGRpdiBjbGFzcz0iZm9vdGVyLWNvcHkiPsKpIDIwMjYgVGhlIFRyaXAgVGhlb3J5IFB2dC4gTHRkLiDCtyBHdXJ1Z3JhbSwgSGFyeWFuYSwgSW5kaWEgwrcgUG93ZXJlZCBieSBDbGF1ZGUgQUk8L2Rpdj4KICAgIDxkaXYgY2xhc3M9ImZvb3Rlci1sZWdhbCI+CiAgICAgIDxhIGhyZWY9IiMiPlByaXZhY3k8L2E+CiAgICAgIDxhIGhyZWY9IiMiPlRlcm1zPC9hPgogICAgICA8YSBocmVmPSIjIj5SZWZ1bmRzPC9hPgogICAgPC9kaXY+CiAgPC9kaXY+CjwvZm9vdGVyPgoKPHNjcmlwdD4KLy8gQ3VzdG9tIGN1cnNvcgpjb25zdCBjdXJzb3IgPSBkb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnY3Vyc29yJyk7CmNvbnN0IHJpbmcgPSBkb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnY3Vyc29yUmluZycpOwpsZXQgcnggPSAwLCByeSA9IDAsIGN4ID0gMCwgY3kgPSAwOwoKZG9jdW1lbnQuYWRkRXZlbnRMaXN0ZW5lcignbW91c2Vtb3ZlJywgZSA9PiB7CiAgY3ggPSBlLmNsaWVudFg7IGN5ID0gZS5jbGllbnRZOwogIGN1cnNvci5zdHlsZS5sZWZ0ID0gY3ggLSA0ICsgJ3B4JzsKICBjdXJzb3Iuc3R5bGUudG9wID0gY3kgLSA0ICsgJ3B4JzsKfSk7CgooZnVuY3Rpb24gYW5pbVJpbmcoKSB7CiAgcnggKz0gKGN4IC0gcngpICogMC4xMjsKICByeSArPSAoY3kgLSByeSkgKiAwLjEyOwogIHJpbmcuc3R5bGUubGVmdCA9IHJ4IC0gMTggKyAncHgnOwogIHJpbmcuc3R5bGUudG9wID0gcnkgLSAxOCArICdweCc7CiAgcmVxdWVzdEFuaW1hdGlvbkZyYW1lKGFuaW1SaW5nKTsKfSkoKTsKCmRvY3VtZW50LnF1ZXJ5U2VsZWN0b3JBbGwoJ2EsIGJ1dHRvbicpLmZvckVhY2goZWwgPT4gewogIGVsLmFkZEV2ZW50TGlzdGVuZXIoJ21vdXNlZW50ZXInLCAoKSA9PiB7CiAgICBjdXJzb3Iuc3R5bGUudHJhbnNmb3JtID0gJ3NjYWxlKDIuNSknOwogICAgcmluZy5zdHlsZS50cmFuc2Zvcm0gPSAnc2NhbGUoMS41KSc7CiAgICByaW5nLnN0eWxlLmJvcmRlckNvbG9yID0gJ3JnYmEoMjAxLDE1MCw1OCwwLjgpJzsKICB9KTsKICBlbC5hZGRFdmVudExpc3RlbmVyKCdtb3VzZWxlYXZlJywgKCkgPT4gewogICAgY3Vyc29yLnN0eWxlLnRyYW5zZm9ybSA9ICdzY2FsZSgxKSc7CiAgICByaW5nLnN0eWxlLnRyYW5zZm9ybSA9ICdzY2FsZSgxKSc7CiAgICByaW5nLnN0eWxlLmJvcmRlckNvbG9yID0gJ3JnYmEoMjAxLDE1MCw1OCwwLjUpJzsKICB9KTsKfSk7CgovLyBTY3JvbGwgcmV2ZWFsCmNvbnN0IG9ic2VydmVyID0gbmV3IEludGVyc2VjdGlvbk9ic2VydmVyKChlbnRyaWVzKSA9PiB7CiAgZW50cmllcy5mb3JFYWNoKChlbnRyeSwgaSkgPT4gewogICAgaWYgKGVudHJ5LmlzSW50ZXJzZWN0aW5nKSB7CiAgICAgIHNldFRpbWVvdXQoKCkgPT4gZW50cnkudGFyZ2V0LmNsYXNzTGlzdC5hZGQoJ3Zpc2libGUnKSwgaSAqIDgwKTsKICAgIH0KICB9KTsKfSwgeyB0aHJlc2hvbGQ6IDAuMDgsIHJvb3RNYXJnaW46ICcwcHggMHB4IC02MHB4IDBweCcgfSk7Cgpkb2N1bWVudC5xdWVyeVNlbGVjdG9yQWxsKCcucmV2ZWFsJykuZm9yRWFjaChlbCA9PiBvYnNlcnZlci5vYnNlcnZlKGVsKSk7CgovLyBOYXYgc2Nyb2xsIGVmZmVjdAp3aW5kb3cuYWRkRXZlbnRMaXN0ZW5lcignc2Nyb2xsJywgKCkgPT4gewogIGNvbnN0IG5hdiA9IGRvY3VtZW50LnF1ZXJ5U2VsZWN0b3IoJ25hdicpOwogIGlmICh3aW5kb3cuc2Nyb2xsWSA+IDgwKSB7CiAgICBuYXYuc3R5bGUuYmFja2dyb3VuZCA9ICdyZ2JhKDEwLDgsNSwwLjk2KSc7CiAgICBuYXYuc3R5bGUuYmFja2Ryb3BGaWx0ZXIgPSAnYmx1cigxMnB4KSc7CiAgICBuYXYuc3R5bGUuYm9yZGVyQm90dG9tID0gJzFweCBzb2xpZCByZ2JhKDIwMSwxNTAsNTgsMC4wOCknOwogIH0gZWxzZSB7CiAgICBuYXYuc3R5bGUuYmFja2dyb3VuZCA9ICdsaW5lYXItZ3JhZGllbnQodG8gYm90dG9tLCByZ2JhKDEwLDgsNSwwLjk1KSAwJSwgdHJhbnNwYXJlbnQgMTAwJSknOwogICAgbmF2LnN0eWxlLmJhY2tkcm9wRmlsdGVyID0gJ2JsdXIoMnB4KSc7CiAgICBuYXYuc3R5bGUuYm9yZGVyQm90dG9tID0gJ25vbmUnOwogIH0KfSk7Cjwvc2NyaXB0PgoKPCEtLSDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZDilZAgLS0+CjwhLS0gVFRUIFNJR04tSU4gTU9EQUwgLS0+CjwhLS0g4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQ4pWQIC0tPgo8c3R5bGU+CiN0dHQtc2lnbmluLW92ZXJsYXl7CiAgcG9zaXRpb246Zml4ZWQ7aW5zZXQ6MDtiYWNrZ3JvdW5kOnJnYmEoMTAsOCw1LDAuODgpO3otaW5kZXg6MTAwMDsKICBkaXNwbGF5Om5vbmU7YWxpZ24taXRlbXM6Y2VudGVyO2p1c3RpZnktY29udGVudDpjZW50ZXI7cGFkZGluZzoyMHB4OwogIGJhY2tkcm9wLWZpbHRlcjpibHVyKDhweCk7Cn0KI3R0dC1zaWduaW4tb3ZlcmxheS5vcGVue2Rpc3BsYXk6ZmxleDt9Ci50dHQtc2lnbmluLWNhcmR7CiAgYmFja2dyb3VuZDojMTUxMDA4O2JvcmRlcjoxcHggc29saWQgcmdiYSgyMDEsMTUwLDU4LDAuMik7CiAgd2lkdGg6MTAwJTttYXgtd2lkdGg6NDYwcHg7Ym9yZGVyLXJhZGl1czoycHg7CiAgb3ZlcmZsb3c6aGlkZGVuO3Bvc2l0aW9uOnJlbGF0aXZlOwogIGFuaW1hdGlvbjpzaWduaW5JbiAuMzVzIGN1YmljLWJlemllcigwLjE2LDEsMC4zLDEpOwp9CkBrZXlmcmFtZXMgc2lnbmluSW57ZnJvbXtvcGFjaXR5OjA7dHJhbnNmb3JtOnRyYW5zbGF0ZVkoMjRweCl9dG97b3BhY2l0eToxO3RyYW5zZm9ybTp0cmFuc2xhdGVZKDApfX0KLnR0dC1zaWduaW4tdG9wewogIHBvc2l0aW9uOnJlbGF0aXZlO2hlaWdodDoxODBweDtvdmVyZmxvdzpoaWRkZW47CiAgYmFja2dyb3VuZDpsaW5lYXItZ3JhZGllbnQoMTM1ZGVnLCMxQTEyMDggMCUsIzBBMDgwNSAxMDAlKTsKfQoudHR0LXNpZ25pbi10b3AtaW1newogIHBvc2l0aW9uOmFic29sdXRlO2luc2V0OjA7CiAgYmFja2dyb3VuZDp1cmwoJ2h0dHBzOi8vaW1hZ2VzLnVuc3BsYXNoLmNvbS9waG90by0xNTA2OTA1OTI1MzQ2LTIxYmRhNGQzMmRmND93PTgwMCZxPTcwJykgY2VudGVyL2NvdmVyOwogIG9wYWNpdHk6MC4zOwp9Ci50dHQtc2lnbmluLXRvcC1jb250ZW50ewogIHBvc2l0aW9uOnJlbGF0aXZlO3otaW5kZXg6MjsKICBoZWlnaHQ6MTAwJTtkaXNwbGF5OmZsZXg7ZmxleC1kaXJlY3Rpb246Y29sdW1uOwogIGFsaWduLWl0ZW1zOmNlbnRlcjtqdXN0aWZ5LWNvbnRlbnQ6Y2VudGVyOwogIHBhZGRpbmc6MjBweDt0ZXh0LWFsaWduOmNlbnRlcjsKfQoudHR0LXNpZ25pbi1sb2dvewogIGZvbnQtZmFtaWx5OidDb3Jtb3JhbnQgR2FyYW1vbmQnLEdlb3JnaWEsc2VyaWY7CiAgZm9udC1zaXplOjIuMnJlbTtmb250LXdlaWdodDozMDA7CiAgY29sb3I6I0M5OTYzQTtsZXR0ZXItc3BhY2luZzowLjI1ZW07CiAgbWFyZ2luLWJvdHRvbTo2cHg7Cn0KLnR0dC1zaWduaW4tdGFnbGluZXsKICBmb250LXNpemU6MC41OHJlbTtmb250LXdlaWdodDozMDA7CiAgbGV0dGVyLXNwYWNpbmc6MC4yNWVtO3RleHQtdHJhbnNmb3JtOnVwcGVyY2FzZTsKICBjb2xvcjpyZ2JhKDI1NSwyNTUsMjU1LDAuNCk7Cn0KLnR0dC1zaWduaW4tY2xvc2V7CiAgcG9zaXRpb246YWJzb2x1dGU7dG9wOjE0cHg7cmlnaHQ6MTZweDsKICBiYWNrZ3JvdW5kOm5vbmU7Ym9yZGVyOm5vbmU7Y3Vyc29yOnBvaW50ZXI7CiAgY29sb3I6cmdiYSgyNTUsMjU1LDI1NSwwLjMpO2ZvbnQtc2l6ZToyMHB4OwogIGxpbmUtaGVpZ2h0OjE7ei1pbmRleDoxMDsKICB0cmFuc2l0aW9uOmNvbG9yIC4yczsKfQoudHR0LXNpZ25pbi1jbG9zZTpob3Zlcntjb2xvcjpyZ2JhKDI1NSwyNTUsMjU1LDAuNyk7fQoudHR0LXNpZ25pbi1ib2R5e3BhZGRpbmc6MzJweCAzNnB4IDM2cHg7fQoudHR0LXNpZ25pbi1zdGVwe2Rpc3BsYXk6bm9uZTt9Ci50dHQtc2lnbmluLXN0ZXAuYWN0aXZle2Rpc3BsYXk6YmxvY2s7fQoudHR0LXNpZ25pbi1zdGVwLWxhYmVsewogIGZvbnQtc2l6ZTowLjU4cmVtO2ZvbnQtd2VpZ2h0OjQwMDsKICBsZXR0ZXItc3BhY2luZzowLjI4ZW07dGV4dC10cmFuc2Zvcm06dXBwZXJjYXNlOwogIGNvbG9yOiNDOTk2M0E7bWFyZ2luLWJvdHRvbToxMHB4OwogIGRpc3BsYXk6ZmxleDthbGlnbi1pdGVtczpjZW50ZXI7Z2FwOjEwcHg7Cn0KLnR0dC1zaWduaW4tc3RlcC1sYWJlbDo6YmVmb3JlewogIGNvbnRlbnQ6Jyc7ZGlzcGxheTpibG9jazt3aWR0aDoyMHB4O2hlaWdodDoxcHg7YmFja2dyb3VuZDojQzk5NjNBOwp9Ci50dHQtc2lnbmluLXRpdGxlewogIGZvbnQtZmFtaWx5OidDb3Jtb3JhbnQgR2FyYW1vbmQnLEdlb3JnaWEsc2VyaWY7CiAgZm9udC1zaXplOjEuNnJlbTtmb250LXdlaWdodDozMDA7CiAgY29sb3I6I0ZFRkNGODttYXJnaW4tYm90dG9tOjZweDtsaW5lLWhlaWdodDoxLjI7Cn0KLnR0dC1zaWduaW4tdGl0bGUgZW17Zm9udC1zdHlsZTppdGFsaWM7Y29sb3I6I0U4Qzg3ODt9Ci50dHQtc2lnbmluLXN1YnsKICBmb250LXNpemU6MC43MnJlbTtmb250LXdlaWdodDozMDA7CiAgbGluZS1oZWlnaHQ6MS44O2NvbG9yOnJnYmEoMjU1LDI1NSwyNTUsMC40KTsKICBtYXJnaW4tYm90dG9tOjI0cHg7Cn0KLnR0dC1zaWduaW4taW5wdXQtd3JhcHsKICBwb3NpdGlvbjpyZWxhdGl2ZTttYXJnaW4tYm90dG9tOjEycHg7Cn0KLnR0dC1zaWduaW4taW5wdXR7CiAgd2lkdGg6MTAwJTtwYWRkaW5nOjEzcHggMTZweDsKICBiYWNrZ3JvdW5kOnJnYmEoMjU1LDI1NSwyNTUsMC4wNCk7CiAgYm9yZGVyOjFweCBzb2xpZCByZ2JhKDIwMSwxNTAsNTgsMC4yKTsKICBib3JkZXItcmFkaXVzOjFweDsKICBmb250LWZhbWlseTonTW9udHNlcnJhdCcsc2Fucy1zZXJpZjsKICBmb250LXNpemU6MC44MnJlbTtmb250LXdlaWdodDozMDA7CiAgY29sb3I6I0ZFRkNGODtvdXRsaW5lOm5vbmU7CiAgdHJhbnNpdGlvbjpib3JkZXIgLjJzOwogIGxldHRlci1zcGFjaW5nOjAuMDVlbTsKfQoudHR0LXNpZ25pbi1pbnB1dDo6cGxhY2Vob2xkZXJ7Y29sb3I6cmdiYSgyNTUsMjU1LDI1NSwwLjI1KTt9Ci50dHQtc2lnbmluLWlucHV0OmZvY3Vze2JvcmRlci1jb2xvcjpyZ2JhKDIwMSwxNTAsNTgsMC42KTt9Ci50dHQtc2lnbmluLWJ0bnsKICB3aWR0aDoxMDAlO3BhZGRpbmc6MTRweDsKICBiYWNrZ3JvdW5kOiNDOTk2M0E7Ym9yZGVyOm5vbmU7CiAgZm9udC1mYW1pbHk6J01vbnRzZXJyYXQnLHNhbnMtc2VyaWY7CiAgZm9udC1zaXplOjAuNjJyZW07Zm9udC13ZWlnaHQ6NDAwOwogIGxldHRlci1zcGFjaW5nOjAuMjJlbTt0ZXh0LXRyYW5zZm9ybTp1cHBlcmNhc2U7CiAgY29sb3I6IzBBMDgwNTtjdXJzb3I6cG9pbnRlcjsKICBtYXJnaW4tdG9wOjhweDsKICB0cmFuc2l0aW9uOmJhY2tncm91bmQgLjJzOwogIHBvc2l0aW9uOnJlbGF0aXZlO292ZXJmbG93OmhpZGRlbjsKfQoudHR0LXNpZ25pbi1idG46aG92ZXJ7YmFja2dyb3VuZDojRThDODc4O30KLnR0dC1zaWduaW4tbm90ZXsKICBmb250LXNpemU6MC42MnJlbTtmb250LXdlaWdodDozMDA7CiAgY29sb3I6cmdiYSgyNTUsMjU1LDI1NSwwLjI1KTsKICB0ZXh0LWFsaWduOmNlbnRlcjttYXJnaW4tdG9wOjE0cHg7CiAgbGluZS1oZWlnaHQ6MS43Owp9Ci50dHQtc2lnbmluLW5vdGUgYXtjb2xvcjpyZ2JhKDIwMSwxNTAsNTgsMC43KTt0ZXh0LWRlY29yYXRpb246bm9uZTt9Ci50dHQtb3RwLWJveGVzewogIGRpc3BsYXk6ZmxleDtnYXA6MTBweDtqdXN0aWZ5LWNvbnRlbnQ6Y2VudGVyOwogIG1hcmdpbi1ib3R0b206MTJweDsKfQoudHR0LW90cC1ib3h7CiAgd2lkdGg6NDZweDtoZWlnaHQ6NTJweDsKICBiYWNrZ3JvdW5kOnJnYmEoMjU1LDI1NSwyNTUsMC4wNCk7CiAgYm9yZGVyOjFweCBzb2xpZCByZ2JhKDIwMSwxNTAsNTgsMC4yKTsKICBib3JkZXItcmFkaXVzOjFweDsKICBmb250LWZhbWlseTonQ29ybW9yYW50IEdhcmFtb25kJyxHZW9yZ2lhLHNlcmlmOwogIGZvbnQtc2l6ZToxLjRyZW07Zm9udC13ZWlnaHQ6MzAwOwogIGNvbG9yOiNGRUZDRjg7dGV4dC1hbGlnbjpjZW50ZXI7CiAgb3V0bGluZTpub25lO3RyYW5zaXRpb246Ym9yZGVyIC4yczsKfQoudHR0LW90cC1ib3g6Zm9jdXN7Ym9yZGVyLWNvbG9yOnJnYmEoMjAxLDE1MCw1OCwwLjcpO30KLnR0dC1iYWNrLWJ0bnsKICBiYWNrZ3JvdW5kOm5vbmU7Ym9yZGVyOm5vbmU7Y3Vyc29yOnBvaW50ZXI7CiAgZm9udC1mYW1pbHk6J01vbnRzZXJyYXQnLHNhbnMtc2VyaWY7CiAgZm9udC1zaXplOjAuNnJlbTtmb250LXdlaWdodDozMDA7CiAgbGV0dGVyLXNwYWNpbmc6MC4xNWVtO3RleHQtdHJhbnNmb3JtOnVwcGVyY2FzZTsKICBjb2xvcjpyZ2JhKDI1NSwyNTUsMjU1LDAuMyk7CiAgZGlzcGxheTpmbGV4O2FsaWduLWl0ZW1zOmNlbnRlcjtnYXA6NnB4OwogIHBhZGRpbmc6MDttYXJnaW4tYm90dG9tOjE2cHg7CiAgdHJhbnNpdGlvbjpjb2xvciAuMnM7Cn0KLnR0dC1iYWNrLWJ0bjpob3Zlcntjb2xvcjpyZ2JhKDIwMSwxNTAsNTgsMC44KTt9Ci50dHQtZGl2aWRlcnsKICBkaXNwbGF5OmZsZXg7YWxpZ24taXRlbXM6Y2VudGVyO2dhcDoxMnB4OwogIG1hcmdpbjoxNnB4IDA7Cn0KLnR0dC1kaXZpZGVyOjpiZWZvcmUsLnR0dC1kaXZpZGVyOjphZnRlcnsKICBjb250ZW50OicnO2ZsZXg6MTtoZWlnaHQ6MXB4OwogIGJhY2tncm91bmQ6cmdiYSgyNTUsMjU1LDI1NSwwLjA4KTsKfQoudHR0LWRpdmlkZXIgc3BhbnsKICBmb250LXNpemU6MC42cmVtO2NvbG9yOnJnYmEoMjU1LDI1NSwyNTUsMC4yNSk7CiAgbGV0dGVyLXNwYWNpbmc6MC4xZW07Cn0KLnR0dC1ndWVzdC1idG57CiAgd2lkdGg6MTAwJTtwYWRkaW5nOjEycHg7CiAgYmFja2dyb3VuZDp0cmFuc3BhcmVudDsKICBib3JkZXI6MXB4IHNvbGlkIHJnYmEoMjU1LDI1NSwyNTUsMC4xKTsKICBmb250LWZhbWlseTonTW9udHNlcnJhdCcsc2Fucy1zZXJpZjsKICBmb250LXNpemU6MC42MnJlbTtmb250LXdlaWdodDozMDA7CiAgbGV0dGVyLXNwYWNpbmc6MC4xOGVtO3RleHQtdHJhbnNmb3JtOnVwcGVyY2FzZTsKICBjb2xvcjpyZ2JhKDI1NSwyNTUsMjU1LDAuNCk7Y3Vyc29yOnBvaW50ZXI7CiAgdHJhbnNpdGlvbjphbGwgLjJzOwp9Ci50dHQtZ3Vlc3QtYnRuOmhvdmVyewogIGJvcmRlci1jb2xvcjpyZ2JhKDIwMSwxNTAsNTgsMC4zKTsKICBjb2xvcjpyZ2JhKDIwMSwxNTAsNTgsMC43KTsKfQo8L3N0eWxlPgoKPGRpdiBpZD0idHR0LXNpZ25pbi1vdmVybGF5IiBvbmNsaWNrPSJpZihldmVudC50YXJnZXQ9PT10aGlzKWNsb3NlVFRUU2lnbmluKCkiPgogIDxkaXYgY2xhc3M9InR0dC1zaWduaW4tY2FyZCI+CiAgICA8ZGl2IGNsYXNzPSJ0dHQtc2lnbmluLXRvcCI+CiAgICAgIDxkaXYgY2xhc3M9InR0dC1zaWduaW4tdG9wLWltZyI+PC9kaXY+CiAgICAgIDxkaXYgY2xhc3M9InR0dC1zaWduaW4tdG9wLWNvbnRlbnQiPgogICAgICAgIDxkaXYgY2xhc3M9InR0dC1zaWduaW4tbG9nbyI+VFRUPC9kaXY+CiAgICAgICAgPGRpdiBjbGFzcz0idHR0LXNpZ25pbi10YWdsaW5lIj5JbmRpYSdzIEZpcnN0IEFJIFRyYXZlbCBDb25jaWVyZ2U8L2Rpdj4KICAgICAgPC9kaXY+CiAgICAgIDxidXR0b24gY2xhc3M9InR0dC1zaWduaW4tY2xvc2UiIG9uY2xpY2s9ImNsb3NlVFRUU2lnbmluKCkiPsOXPC9idXR0b24+CiAgICA8L2Rpdj4KCiAgICA8ZGl2IGNsYXNzPSJ0dHQtc2lnbmluLWJvZHkiPgoKICAgICAgPCEtLSBTdGVwIDE6IEVudGVyIHBob25lL2VtYWlsIC0tPgogICAgICA8ZGl2IGNsYXNzPSJ0dHQtc2lnbmluLXN0ZXAgYWN0aXZlIiBpZD0idHR0LXN0ZXAtMSI+CiAgICAgICAgPGRpdiBjbGFzcz0idHR0LXNpZ25pbi1zdGVwLWxhYmVsIj5CZWdpbjwvZGl2PgogICAgICAgIDxkaXYgY2xhc3M9InR0dC1zaWduaW4tdGl0bGUiPllvdXIgam91cm5leTxicj48ZW0+c3RhcnRzIGhlcmUuPC9lbT48L2Rpdj4KICAgICAgICA8ZGl2IGNsYXNzPSJ0dHQtc2lnbmluLXN1YiI+RW50ZXIgeW91ciBwaG9uZSBudW1iZXIgb3IgZW1haWwgdG8gcmVjZWl2ZSBhIG9uZS10aW1lIGNvZGUuPC9kaXY+CiAgICAgICAgPGRpdiBjbGFzcz0idHR0LXNpZ25pbi1pbnB1dC13cmFwIj4KICAgICAgICAgIDxpbnB1dCBjbGFzcz0idHR0LXNpZ25pbi1pbnB1dCIgaWQ9InR0dC1jb250YWN0LWlucHV0IgogICAgICAgICAgICB0eXBlPSJ0ZXh0IiBwbGFjZWhvbGRlcj0iKzkxIFhYWFhYIFhYWFhYIG9yIGVtYWlsQGV4YW1wbGUuY29tIgogICAgICAgICAgICBhdXRvY29tcGxldGU9Im9mZiI+CiAgICAgICAgPC9kaXY+CiAgICAgICAgPGJ1dHRvbiBjbGFzcz0idHR0LXNpZ25pbi1idG4iIG9uY2xpY2s9InR0dFNlbmRPVFAoKSI+Q29udGludWUg4oaSPC9idXR0b24+CiAgICAgICAgPGRpdiBjbGFzcz0idHR0LWRpdmlkZXIiPjxzcGFuPm9yPC9zcGFuPjwvZGl2PgogICAgICAgIDxidXR0b24gY2xhc3M9InR0dC1ndWVzdC1idG4iIG9uY2xpY2s9InR0dENvbnRpbnVlQXNHdWVzdCgpIj5Db250aW51ZSBhcyBndWVzdDwvYnV0dG9uPgogICAgICAgIDxkaXYgY2xhc3M9InR0dC1zaWduaW4tbm90ZSI+CiAgICAgICAgICBCeSBjb250aW51aW5nIHlvdSBhZ3JlZSB0byBvdXIKICAgICAgICAgIDxhIGhyZWY9IiMiPlRlcm1zIG9mIFNlcnZpY2U8L2E+ICZhbXA7IDxhIGhyZWY9IiMiPlByaXZhY3kgUG9saWN5PC9hPgogICAgICAgIDwvZGl2PgogICAgICA8L2Rpdj4KCiAgICAgIDwhLS0gU3RlcCAyOiBPVFAgLS0+CiAgICAgIDxkaXYgY2xhc3M9InR0dC1zaWduaW4tc3RlcCIgaWQ9InR0dC1zdGVwLTIiPgogICAgICAgIDxidXR0b24gY2xhc3M9InR0dC1iYWNrLWJ0biIgb25jbGljaz0idHR0R29CYWNrKCkiPuKGkCBCYWNrPC9idXR0b24+CiAgICAgICAgPGRpdiBjbGFzcz0idHR0LXNpZ25pbi1zdGVwLWxhYmVsIj5WZXJpZnk8L2Rpdj4KICAgICAgICA8ZGl2IGNsYXNzPSJ0dHQtc2lnbmluLXRpdGxlIj5PbmUtdGltZTxicj48ZW0+Y29kZSBzZW50LjwvZW0+PC9kaXY+CiAgICAgICAgPGRpdiBjbGFzcz0idHR0LXNpZ25pbi1zdWIiIGlkPSJ0dHQtb3RwLXN1YiI+RW50ZXIgdGhlIDYtZGlnaXQgY29kZSB3ZSBzZW50IHlvdS48L2Rpdj4KICAgICAgICA8ZGl2IGNsYXNzPSJ0dHQtb3RwLWJveGVzIj4KICAgICAgICAgIDxpbnB1dCBjbGFzcz0idHR0LW90cC1ib3giIG1heGxlbmd0aD0iMSIgdHlwZT0idGV4dCIgaW5wdXRtb2RlPSJudW1lcmljIiBpZD0ib3RwMCIgb25pbnB1dD0idHR0T1RQTmV4dCgwKSI+CiAgICAgICAgICA8aW5wdXQgY2xhc3M9InR0dC1vdHAtYm94IiBtYXhsZW5ndGg9IjEiIHR5cGU9InRleHQiIGlucHV0bW9kZT0ibnVtZXJpYyIgaWQ9Im90cDEiIG9uaW5wdXQ9InR0dE9UUE5leHQoMSkiPgogICAgICAgICAgPGlucHV0IGNsYXNzPSJ0dHQtb3RwLWJveCIgbWF4bGVuZ3RoPSIxIiB0eXBlPSJ0ZXh0IiBpbnB1dG1vZGU9Im51bWVyaWMiIGlkPSJvdHAyIiBvbmlucHV0PSJ0dHRPVFBOZXh0KDIpIj4KICAgICAgICAgIDxpbnB1dCBjbGFzcz0idHR0LW90cC1ib3giIG1heGxlbmd0aD0iMSIgdHlwZT0idGV4dCIgaW5wdXRtb2RlPSJudW1lcmljIiBpZD0ib3RwMyIgb25pbnB1dD0idHR0T1RQTmV4dCgzKSI+CiAgICAgICAgICA8aW5wdXQgY2xhc3M9InR0dC1vdHAtYm94IiBtYXhsZW5ndGg9IjEiIHR5cGU9InRleHQiIGlucHV0bW9kZT0ibnVtZXJpYyIgaWQ9Im90cDQiIG9uaW5wdXQ9InR0dE9UUE5leHQoNCkiPgogICAgICAgICAgPGlucHV0IGNsYXNzPSJ0dHQtb3RwLWJveCIgbWF4bGVuZ3RoPSIxIiB0eXBlPSJ0ZXh0IiBpbnB1dG1vZGU9Im51bWVyaWMiIGlkPSJvdHA1IiBvbmlucHV0PSJ0dHRPVFBOZXh0KDUpIj4KICAgICAgICA8L2Rpdj4KICAgICAgICA8YnV0dG9uIGNsYXNzPSJ0dHQtc2lnbmluLWJ0biIgb25jbGljaz0idHR0VmVyaWZ5T1RQKCkiPlZlcmlmeSAmYW1wOyBFbnRlciDihpI8L2J1dHRvbj4KICAgICAgICA8ZGl2IGNsYXNzPSJ0dHQtc2lnbmluLW5vdGUiPgogICAgICAgICAgRGlkbid0IHJlY2VpdmUgaXQ/IDxhIGhyZWY9IiMiIG9uY2xpY2s9InR0dEdvQmFjaygpO3JldHVybiBmYWxzZTsiPlRyeSBhZ2FpbjwvYT4KICAgICAgICA8L2Rpdj4KICAgICAgPC9kaXY+CgogICAgPC9kaXY+CiAgPC9kaXY+CjwvZGl2PgoKPHNjcmlwdD4KZnVuY3Rpb24gb3BlblRUVFNpZ25pbigpewogIGRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCd0dHQtc2lnbmluLW92ZXJsYXknKS5jbGFzc0xpc3QuYWRkKCdvcGVuJyk7CiAgZG9jdW1lbnQuYm9keS5zdHlsZS5vdmVyZmxvdz0naGlkZGVuJzsKfQpmdW5jdGlvbiBjbG9zZVRUVFNpZ25pbigpewogIGRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCd0dHQtc2lnbmluLW92ZXJsYXknKS5jbGFzc0xpc3QucmVtb3ZlKCdvcGVuJyk7CiAgZG9jdW1lbnQuYm9keS5zdHlsZS5vdmVyZmxvdz0nJzsKfQpmdW5jdGlvbiB0dHRHb0JhY2soKXsKICBkb2N1bWVudC5nZXRFbGVtZW50QnlJZCgndHR0LXN0ZXAtMScpLmNsYXNzTGlzdC5hZGQoJ2FjdGl2ZScpOwogIGRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCd0dHQtc3RlcC0yJykuY2xhc3NMaXN0LnJlbW92ZSgnYWN0aXZlJyk7Cn0KZnVuY3Rpb24gdHR0U2VuZE9UUCgpewogIGNvbnN0IHZhbCA9IGRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCd0dHQtY29udGFjdC1pbnB1dCcpLnZhbHVlLnRyaW0oKTsKICBpZighdmFsKXsgZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3R0dC1jb250YWN0LWlucHV0JykuZm9jdXMoKTsgcmV0dXJuOyB9CiAgZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3R0dC1vdHAtc3ViJykudGV4dENvbnRlbnQgPSAnRW50ZXIgdGhlIDYtZGlnaXQgY29kZSBzZW50IHRvICcgKyB2YWw7CiAgZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3R0dC1zdGVwLTEnKS5jbGFzc0xpc3QucmVtb3ZlKCdhY3RpdmUnKTsKICBkb2N1bWVudC5nZXRFbGVtZW50QnlJZCgndHR0LXN0ZXAtMicpLmNsYXNzTGlzdC5hZGQoJ2FjdGl2ZScpOwogIHNldFRpbWVvdXQoKCk9PmRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdvdHAwJykuZm9jdXMoKSwgMTAwKTsKICAvLyBDYWxsIHRoZSBleGlzdGluZyBUVFQgYmFja2VuZAogIGZldGNoKCcvYXBpL2F1dGgvc2VuZC1vdHAnLCB7CiAgICBtZXRob2Q6J1BPU1QnLCBoZWFkZXJzOnsnQ29udGVudC1UeXBlJzonYXBwbGljYXRpb24vanNvbid9LAogICAgYm9keTpKU09OLnN0cmluZ2lmeSh7Y29udGFjdDogdmFsfSkKICB9KS5jYXRjaCgoKT0+e30pOwp9CmZ1bmN0aW9uIHR0dE9UUE5leHQoaWR4KXsKICBjb25zdCB2ID0gZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ290cCcraWR4KS52YWx1ZTsKICBpZih2ICYmIGlkeCA8IDUpIGRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdvdHAnKyhpZHgrMSkpLmZvY3VzKCk7CiAgLy8gQXV0by1zdWJtaXQgd2hlbiBhbGwgZmlsbGVkCiAgY29uc3QgYWxsID0gWzAsMSwyLDMsNCw1XS5tYXAoaT0+ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ290cCcraSkudmFsdWUpLmpvaW4oJycpOwogIGlmKGFsbC5sZW5ndGg9PT02KSB0dHRWZXJpZnlPVFAoKTsKfQpmdW5jdGlvbiB0dHRWZXJpZnlPVFAoKXsKICBjb25zdCBvdHAgPSBbMCwxLDIsMyw0LDVdLm1hcChpPT5kb2N1bWVudC5nZXRFbGVtZW50QnlJZCgnb3RwJytpKS52YWx1ZSkuam9pbignJyk7CiAgY29uc3QgY29udGFjdCA9IGRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCd0dHQtY29udGFjdC1pbnB1dCcpLnZhbHVlLnRyaW0oKTsKICBpZihvdHAubGVuZ3RoPDYpIHJldHVybjsKICAvLyBWZXJpZnkgd2l0aCBiYWNrZW5kIHRoZW4gcmVkaXJlY3QgdG8gYXBwCiAgZmV0Y2goJy9hcGkvYXV0aC92ZXJpZnktb3RwJywgewogICAgbWV0aG9kOidQT1NUJywgaGVhZGVyczp7J0NvbnRlbnQtVHlwZSc6J2FwcGxpY2F0aW9uL2pzb24nfSwKICAgIGJvZHk6SlNPTi5zdHJpbmdpZnkoe2NvbnRhY3Q6IGNvbnRhY3QsIG90cDogb3RwfSkKICB9KQogIC50aGVuKHI9PnIuanNvbigpKQogIC50aGVuKGQ9PnsKICAgIGlmKGQudG9rZW4gfHwgZC5zdWNjZXNzKXsKICAgICAgaWYoZC50b2tlbikgc2Vzc2lvblN0b3JhZ2Uuc2V0SXRlbSgndHR0X2F1dGgnLCBKU09OLnN0cmluZ2lmeShkKSk7CiAgICAgIHdpbmRvdy5sb2NhdGlvbi5ocmVmID0gJy8nOwogICAgfQogIH0pCiAgLmNhdGNoKCgpPT57CiAgICAvLyBJZiBBUEkgZmFpbHMsIHN0aWxsIGdvIHRvIHRoZSBhcHAgKGRldiBtb2RlKQogICAgd2luZG93LmxvY2F0aW9uLmhyZWYgPSAnLyc7CiAgfSk7Cn0KZnVuY3Rpb24gdHR0Q29udGludWVBc0d1ZXN0KCl7CiAgY2xvc2VUVFRTaWduaW4oKTsKICB3aW5kb3cubG9jYXRpb24uaHJlZiA9ICcvI2d1ZXN0JzsKfQoKLy8gRW50ZXIga2V5IG9uIGNvbnRhY3QgaW5wdXQKZG9jdW1lbnQuYWRkRXZlbnRMaXN0ZW5lcignRE9NQ29udGVudExvYWRlZCcsICgpPT57CiAgY29uc3QgaW5wID0gZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoJ3R0dC1jb250YWN0LWlucHV0Jyk7CiAgaWYoaW5wKSBpbnAuYWRkRXZlbnRMaXN0ZW5lcigna2V5ZG93bicsIGU9PnsgaWYoZS5rZXk9PT0nRW50ZXInKSB0dHRTZW5kT1RQKCk7IH0pOwp9KTsKPC9zY3JpcHQ+Cgo8L2JvZHk+CjwvaHRtbD4K"
    _html = _b64.b64decode(_b64_html).decode("utf-8")
    return HTMLResponse(content=_html, headers={"Cache-Control": "no-store"})


@app.get("/crm")
async def crm_dashboard():
    # Try file first, fallback to inline HTML
    attempts = [
        os.path.join(frontend_dir, "crm.html"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend", "crm.html"),
        os.path.join(os.getcwd(), "frontend", "crm.html"),
        "/app/frontend/crm.html",
    ]
    for crm_file in attempts:
        if os.path.exists(crm_file):
            return FileResponse(crm_file, headers={"Cache-Control": "no-store"})
    return HTMLResponse("<h1>CRM Loading...</h1><script>setTimeout(()=>location.reload(),2000)</script>", 200)

# ══════════════════════════════════════════════════════════════════════════════
# END CRM  [landing v2 - signin modal]
# ══════════════════════════════════════════════════════════════════════════════
