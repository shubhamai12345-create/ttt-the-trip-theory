# TTT Build: 2026-05-28 11:48:34 UTC
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


@app.get("/partner")
async def partner_portal():
    """TTT Partner Portal — MYT-style business management dashboard."""
    for fp in [
        os.path.join(frontend_dir, "partner.html"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend", "partner.html"),
        os.path.join(os.getcwd(), "frontend", "partner.html"),
    ]:
        if os.path.exists(fp):
            return FileResponse(fp, headers={"Cache-Control": "no-store"})
    return HTMLResponse("<h2>Partner portal not found</h2>", 404)

@app.get("/app")
async def app_page():
    """TTT AI Concierge app — served directly without old welcome popup."""
    # Try all path variations for index.html
    for fp in [
        os.path.join(frontend_dir, "index.html"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend", "index.html"),
        os.path.join(os.getcwd(), "frontend", "index.html"),
    ]:
        if os.path.exists(fp):
            return FileResponse(fp, headers={"Cache-Control": "no-store"})
    return HTMLResponse("<h2>App not found</h2>", 404)


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


_last_email_error = ""

def _send_otp_email(to_email: str, code: str) -> bool:
    global _last_email_error
    _last_email_error = ""
    _resend_key = os.getenv("RESEND_API_KEY", "")
    if not _resend_key:
        _last_email_error = "RESEND_API_KEY env var is empty or not set"
        print(f"[TTT Resend] {_last_email_error}")
    else:
        try:
            import urllib.request as _ur, json as _json
            _payload = _json.dumps({
                "from":    "TTT Concierge <noreply@thetriptheory.com>",
                "to":      [to_email],
                "subject": "Your TTT verification code",
                "html":    f'''<div style="font-family:Georgia,serif;max-width:480px;margin:auto;padding:32px;background:#0A0805;color:#FEFCF8;border:1px solid rgba(201,150,58,0.2)"><div style="font-size:1.8rem;color:#C9963A;letter-spacing:0.2em;margin-bottom:16px">TTT</div><p style="font-size:0.9rem;color:rgba(255,255,255,0.6);margin-bottom:24px">Your verification code:</p><div style="font-size:2.5rem;letter-spacing:0.3em;color:#E8C878;text-align:center;padding:24px;border:1px solid rgba(201,150,58,0.3);margin:0 0 24px">{code}</div><p style="font-size:0.75rem;color:rgba(255,255,255,0.3)">Expires in 10 minutes. The Trip Theory.</p></div>'''
            }).encode()
            _req = _ur.Request(
                "https://api.resend.com/emails",
                data=_payload,
                headers={"Authorization": f"Bearer {_resend_key}", "Content-Type": "application/json", "User-Agent": "TTT-Backend/1.0"},
                method="POST"
            )
            with _ur.urlopen(_req, timeout=10) as _resp:
                _body = _resp.read().decode()
                print(f"[TTT Resend] Success! Status={_resp.status} Body={_body}")
                return _resp.status in (200, 201)
        except Exception as _e:
            _last_email_error = str(_e)
            print(f"[TTT Resend] Error: {_e}")
            # Try to read error body
            if hasattr(_e, "read"):
                try:
                    _last_email_error += " | " + _e.read().decode()
                except: pass
    # SMTP fallback
    if not (SMTP_USER and SMTP_PASS):
        print(f"[TTT OTP] Demo mode — OTP for {to_email}: {code}")
        return False
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

    # Mark OTP record as demo if email wasn't delivered
    if not sent and contact in _otp_store:
        _otp_store[contact]["demo"] = True

    return {
        "success": True,
        "message": f"OTP sent to {contact}",
        "demo":    not sent,
        "sent":    sent,
        "debug_error": _last_email_error if not sent else "",
        "resend_configured": bool(os.getenv("RESEND_API_KEY", "")),
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

    # In demo mode (email couldn't be delivered), accept any 6-digit code
    is_demo = record.get("demo", False)
    if not is_demo and code != record["code"]:
        remaining = OTP_MAX_ATTEMPTS - record["attempts"]
        raise HTTPException(400, f"Incorrect OTP. {remaining} attempt(s) remaining.")
    if is_demo:
        print(f"[TTT OTP] Demo bypass: accepted any code for {contact}")

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
# Use Railway persistent volume if available, else local (volume survives redeploys)
_DB_DIR = os.getenv("DATA_DIR", "/data") if os.path.isdir(os.getenv("DATA_DIR", "/data")) else os.path.dirname(__file__)
try:
    os.makedirs(_DB_DIR, exist_ok=True)
    # Test writability
    _test = os.path.join(_DB_DIR, ".write_test")
    with open(_test, "w") as _f: _f.write("ok")
    os.remove(_test)
except Exception:
    _DB_DIR = os.path.dirname(__file__)
DB_PATH = os.path.join(_DB_DIR, "ttt_data.json")
print(f"[TTT] Database path: {DB_PATH}")

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
            "invoices":        globals().get("_invoices", {}),
            "partner_wallets_v2": globals().get("_partner_wallets_v2", {}),
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

# ═══ SEED PARTNER DATA (always present on startup) ═══
_SEED_PARTNERS = {
    "ttt-hp-001": {
        "id": "ttt-hp-001", "name": "Rajesh Kumar", "business_name": "Usha River Side Resort",
        "email": "info@ushariverside.com", "phone": "+91 98765 43210",
        "listing_type": "property", "location": "Tirthan Valley, Himachal Pradesh",
        "category": "Boutique Resort", "type": "stay",
        "description": "A serene riverside retreat in the heart of Tirthan Valley, offering eco-friendly luxury with breathtaking Himalayan views.",
        "created_at": "2026-01-15T10:00:00"
    },
    "ttt-hp-002": {
        "id": "ttt-hp-002", "name": "Tenzin Norbu", "business_name": "Bímil Society",
        "email": "hello@bimilsociety.com", "phone": "+91 98765 43211",
        "listing_type": "property", "location": "McLeod Ganj, Himachal Pradesh",
        "category": "Boutique Homestay", "type": "stay",
        "description": "A curated community living space in McLeod Ganj with cultural immersion, meditation, and mountain experiences.",
        "created_at": "2026-02-01T10:00:00"
    }
}
for _spid, _spdata in _SEED_PARTNERS.items():
    if _spid not in _partners:
        _partners[_spid] = _spdata
        print(f"[TTT] Seeded partner: {_spid} ({_spdata['business_name']})")
_save_db()

_SEED_LISTINGS = {
    "listing-usha-001": {
        "id": "listing-usha-001", "partner_id": "ttt-hp-001", "listing_type": "property",
        "title": "Usha River Side Resort — Riverside Room", "name": "Usha River Side Resort — Riverside Room",
        "description": "Wake up to the sound of the Tirthan river. Eco-friendly rooms with mountain views, organic meals, and guided treks.",
        "price_per_night": 5000, "price": 5000, "max_guests": 4, "location": "Tirthan Valley, HP",
        "amenities": "WiFi, Organic Meals, Bonfire, River Access, Trekking, Bird Watching",
        "type": "stay", "status": "active", "created_at": "2026-01-15T10:00:00"
    },
    "listing-usha-002": {
        "id": "listing-usha-002", "partner_id": "ttt-hp-001", "listing_type": "property",
        "title": "Mountain View Cottage", "name": "Mountain View Cottage",
        "description": "Private cottage with panoramic Himalayan views, perfect for couples and families seeking tranquility.",
        "price_per_night": 3500, "price": 3500, "max_guests": 3, "location": "Tirthan Valley, HP",
        "amenities": "WiFi, Meals, Private Balcony, Garden, Parking",
        "type": "stay", "status": "active", "created_at": "2026-01-20T10:00:00"
    },
    "listing-usha-003": {
        "id": "listing-usha-003", "partner_id": "ttt-hp-001", "listing_type": "tour",
        "title": "Tirthan Valley Trek — GHNP", "name": "Tirthan Valley Trek — GHNP",
        "description": "Guided trek through the Great Himalayan National Park. Spot rare birds, medicinal plants, and camp under the stars.",
        "price_per_night": 2500, "price": 2500, "max_guests": 8, "location": "GHNP, HP",
        "amenities": "Guide, Camping Gear, Meals, Permits",
        "type": "tour", "status": "active", "created_at": "2026-02-10T10:00:00"
    },
    "listing-bimil-001": {
        "id": "listing-bimil-001", "partner_id": "ttt-hp-002", "listing_type": "property",
        "title": "Bímil Society — Community Room", "name": "Bímil Society — Community Room",
        "description": "Live like a local in McLeod Ganj. Community kitchen, meditation sessions, Tibetan culture immersion.",
        "price_per_night": 4500, "price": 4500, "max_guests": 2, "location": "McLeod Ganj, HP",
        "amenities": "WiFi, Community Kitchen, Meditation, Library, Rooftop",
        "type": "stay", "status": "active", "created_at": "2026-02-01T10:00:00"
    }
}
for _slid, _sldata in _SEED_LISTINGS.items():
    if _slid not in _listings:
        _listings[_slid] = _sldata
        print(f"[TTT] Seeded listing: {_slid}")
_save_db()

# ═══ INVOICE ENGINE ═══
_invoices: Dict[str, dict] = _db.get("invoices", {})



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



# ═══════════════════════════════════════════════════════
# PARTNER AUTHENTICATION — Email + Password + OTP signup
# ═══════════════════════════════════════════════════════
import hashlib as _hashlib

def _hash_pw(password: str) -> str:
    return _hashlib.sha256(("ttt_salt_2026_" + password).encode()).hexdigest()

class PartnerSignupStart(BaseModel):
    name: str
    business_name: str
    email: str
    phone: str = ""
    listing_type: str = "property"

class PartnerSignupVerify(BaseModel):
    email: str
    code: str
    password: str

class PartnerLogin(BaseModel):
    email: str
    password: str

# Temp store for pending signups (email -> partner data, until OTP verified)
_pending_partner_signups: Dict[str, dict] = {}


@app.post("/api/partner/signup/start")
async def partner_signup_start(req: PartnerSignupStart):
    """Step 1: Partner enters details + email, receives OTP."""
    email = req.email.strip().lower()
    if "@" not in email:
        raise HTTPException(400, "Invalid email address")
    # Check if email already registered
    for p in _partners.values():
        if (p.get("email") or "").strip().lower() == email and p.get("password_hash"):
            raise HTTPException(400, "An account with this email already exists. Please log in.")
    # Generate + send OTP
    code = str(random.randint(100000, 999999))
    _otp_store[email] = {"code": code, "expires_at": _time.time() + OTP_EXPIRY_SECONDS, "attempts": 0, "mode": "email"}
    # Store pending signup
    _pending_partner_signups[email] = {
        "name": req.name, "business_name": req.business_name,
        "email": email, "phone": req.phone, "listing_type": req.listing_type
    }
    sent = _send_otp_email(email, code)
    if not sent:
        _otp_store[email]["demo"] = True
    return {"success": True, "email": email, "sent": sent,
            "message": f"Verification code sent to {email}" if sent else "Email delivery pending — demo mode active (use any 6 digits)"}

@app.post("/api/partner/signup/verify")
async def partner_signup_verify(req: PartnerSignupVerify):
    """Step 2: Verify OTP + set password → create partner account."""
    email = req.email.strip().lower()
    record = _otp_store.get(email)
    if not record:
        raise HTTPException(400, "No verification code requested. Start over.")
    if _time.time() > record["expires_at"]:
        _otp_store.pop(email, None)
        raise HTTPException(400, "Code expired. Please request a new one.")
    is_demo = record.get("demo", False)
    if not is_demo and req.code.strip() != record["code"]:
        record["attempts"] += 1
        raise HTTPException(400, "Incorrect code. Try again.")
    if len(req.password) < 6:
        raise HTTPException(400, "Password must be at least 6 characters")
    pending = _pending_partner_signups.get(email)
    if not pending:
        raise HTTPException(400, "Signup session expired. Start over.")
    # Create partner
    pid = "partner-" + str(uuid.uuid4())[:8]
    record_data = {
        "id": pid, "name": pending["name"], "business_name": pending["business_name"],
        "email": email, "phone": pending["phone"], "listing_type": pending["listing_type"],
        "password_hash": _hash_pw(req.password),
        "created_at": datetime.now().isoformat(),
    }
    _partners[pid] = record_data
    _save_db()
    _otp_store.pop(email, None)
    _pending_partner_signups.pop(email, None)
    safe = {k: v for k, v in record_data.items() if k != "password_hash"}
    return {"success": True, "partner_id": pid, "partner": safe}

@app.post("/api/partner/login")
async def partner_login(req: PartnerLogin):
    """Log in with email + password."""
    email = req.email.strip().lower()
    for p in _partners.values():
        if (p.get("email") or "").strip().lower() == email:
            if not p.get("password_hash"):
                raise HTTPException(400, "This account has no password set. Please sign up again.")
            if p["password_hash"] != _hash_pw(req.password):
                raise HTTPException(401, "Incorrect password")
            safe = {k: v for k, v in p.items() if k != "password_hash"}
            return {"success": True, "partner": safe}
    raise HTTPException(404, "No account found with this email")

@app.post("/api/partner/forgot-password")
async def partner_forgot_password(req: OTPSendRequest):
    """Send OTP to reset password."""
    email = req.contact.strip().lower()
    found = None
    for p in _partners.values():
        if (p.get("email") or "").strip().lower() == email:
            found = p; break
    if not found:
        raise HTTPException(404, "No account found with this email")
    code = str(random.randint(100000, 999999))
    _otp_store[email] = {"code": code, "expires_at": _time.time() + OTP_EXPIRY_SECONDS, "attempts": 0, "mode": "email", "reset": True}
    sent = _send_otp_email(email, code)
    if not sent:
        _otp_store[email]["demo"] = True
    return {"success": True, "sent": sent}

class PartnerResetPassword(BaseModel):
    email: str
    code: str
    password: str

@app.post("/api/partner/reset-password")
async def partner_reset_password(req: PartnerResetPassword):
    """Reset password with OTP."""
    email = req.email.strip().lower()
    record = _otp_store.get(email)
    if not record:
        raise HTTPException(400, "No reset code requested")
    if _time.time() > record["expires_at"]:
        _otp_store.pop(email, None)
        raise HTTPException(400, "Code expired")
    is_demo = record.get("demo", False)
    if not is_demo and req.code.strip() != record["code"]:
        raise HTTPException(400, "Incorrect code")
    if len(req.password) < 6:
        raise HTTPException(400, "Password must be at least 6 characters")
    for p in _partners.values():
        if (p.get("email") or "").strip().lower() == email:
            p["password_hash"] = _hash_pw(req.password)
            _save_db()
            _otp_store.pop(email, None)
            return {"success": True}
    raise HTTPException(404, "Account not found")


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
    if partner_id in _partners:
        return {k: v for k, v in _partners[partner_id].items() if k != "password_hash"}
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
    raw = [b for b in _bookings.values() if b.get("listing_id") in partner_listing_ids or b.get("partner_id") == partner_id]
    result = []
    for b in raw:
        result.append({**b,
            "guest_name": b.get("customer_name", ""),
            "guest_email": b.get("customer_email", ""),
            "listing_name": b.get("listing_title", ""),
            "check_in": b.get("checkin", ""),
            "amount": b.get("total_price", 0),
        })
    result.sort(key=lambda x: x.get("created_at",""), reverse=True)
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
    # Credit partner wallet
    try:
        _credit_partner_wallet(listing["partner_id"], req.total_price, bid, f"Booking — {req.customer_name} — {listing['title']}")
    except Exception as e:
        print(f"[TTT] Wallet credit error: {e}")
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


# ── Google OAuth + Resend Email ───────────────────────────────────────────────
GOOGLE_CLIENT_ID  = os.getenv("GOOGLE_CLIENT_ID", "")
RESEND_API_KEY    = os.getenv("RESEND_API_KEY", "")
FROM_EMAIL        = os.getenv("FROM_EMAIL", "TTT <onboarding@resend.dev>")

def _send_otp_resend(to_email: str, code: str) -> bool:
    """Send OTP via Resend (free 3k/month). Uses onboarding domain — no verification needed."""
    if not RESEND_API_KEY:
        return False
    try:
        resp = requests.post(
            "https://api.resend.com/emails",
            headers={"Authorization": f"Bearer {RESEND_API_KEY}", "Content-Type": "application/json"},
            json={
                "from":    "TTT Concierge <noreply@thetriptheory.com>",
                "to":      [to_email],
                "subject": f"{code} — Your TTT verification code",
                "html":    f"""
                <div style="font-family:Georgia,serif;max-width:480px;margin:auto;padding:32px;background:#0A0805;color:#FEFCF8;border:1px solid rgba(201,150,58,0.2)">
                  <div style="font-size:1.8rem;color:#C9963A;letter-spacing:0.2em;margin-bottom:16px">TTT</div>
                  <p style="font-size:0.9rem;color:rgba(255,255,255,0.6);margin-bottom:24px">Your one-time verification code:</p>
                  <div style="font-size:2.5rem;letter-spacing:0.3em;color:#E8C878;text-align:center;padding:24px;border:1px solid rgba(201,150,58,0.3);margin:0 0 24px">{code}</div>
                  <p style="font-size:0.75rem;color:rgba(255,255,255,0.3)">Expires in 10 minutes. The Trip Theory — India's First AI Concierge.</p>
                </div>"""
            },
            timeout=10
        )
        return resp.status_code in (200, 201)
    except Exception:
        return False

async def _verify_google_token(id_token: str):
    """Verify Google ID token using Google's tokeninfo endpoint (free)."""
    try:
        import urllib.request, json as _json
        url = f"https://oauth2.googleapis.com/tokeninfo?id_token={id_token}"
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = _json.loads(resp.read())
        if GOOGLE_CLIENT_ID and GOOGLE_CLIENT_ID != "PENDING" and data.get("aud") != GOOGLE_CLIENT_ID:
            return None
        if data.get("email_verified") != "true":
            return None
        return data
    except Exception:
        return None




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



@app.get("/api/admin/resend-setup")
async def resend_domain_setup(key: str = ""):
    """One-time setup: create domain in Resend and return DNS records."""
    if key != os.getenv("ADMIN_KEY", "ttt-admin-2024"):
        raise HTTPException(403, "Invalid admin key")
    import urllib.request as _ur
    _resend_key = os.getenv("RESEND_API_KEY", "")
    if not _resend_key:
        return {"error": "RESEND_API_KEY not set"}
    
    # First try to list existing domains
    try:
        _req = _ur.Request(
            "https://api.resend.com/domains",
            headers={"Authorization": f"Bearer {_resend_key}", "Content-Type": "application/json", "User-Agent": "TTT-Backend/1.0"},
            method="GET"
        )
        with _ur.urlopen(_req, timeout=10) as _resp:
            existing = json.loads(_resp.read().decode())
            # Check if domain already exists
            for d in existing.get("data", []):
                if d.get("name") == "thetriptheory.com":
                    # Get full details
                    _req2 = _ur.Request(
                        f"https://api.resend.com/domains/{d['id']}",
                        headers={"Authorization": f"Bearer {_resend_key}", "Content-Type": "application/json", "User-Agent": "TTT-Backend/1.0"},
                        method="GET"
                    )
                    with _ur.urlopen(_req2, timeout=10) as _resp2:
                        return {"status": "already_exists", "domain": json.loads(_resp2.read().decode())}
    except Exception as e:
        pass
    
    # Create domain
    try:
        _payload = json.dumps({"name": "thetriptheory.com"}).encode()
        _req = _ur.Request(
            "https://api.resend.com/domains",
            data=_payload,
            headers={"Authorization": f"Bearer {_resend_key}", "Content-Type": "application/json", "User-Agent": "TTT-Backend/1.0"},
            method="POST"
        )
        with _ur.urlopen(_req, timeout=10) as _resp:
            result = json.loads(_resp.read().decode())
            return {"status": "created", "domain": result}
    except Exception as e:
        err_body = ""
        if hasattr(e, "read"):
            try: err_body = e.read().decode()
            except: pass
        return {"error": str(e), "body": err_body}


@app.get("/api/admin/resend-verify")
async def resend_domain_verify(key: str = ""):
    """Trigger domain verification check in Resend."""
    if key != os.getenv("ADMIN_KEY", "ttt-admin-2024"):
        raise HTTPException(403, "Invalid admin key")
    import urllib.request as _ur
    _resend_key = os.getenv("RESEND_API_KEY", "")
    
    # List domains to find ours
    try:
        _req = _ur.Request(
            "https://api.resend.com/domains",
            headers={"Authorization": f"Bearer {_resend_key}", "Content-Type": "application/json", "User-Agent": "TTT-Backend/1.0"},
            method="GET"
        )
        with _ur.urlopen(_req, timeout=10) as _resp:
            domains = json.loads(_resp.read().decode())
            for d in domains.get("data", []):
                if d.get("name") == "thetriptheory.com":
                    # Trigger verify
                    _req2 = _ur.Request(
                        f"https://api.resend.com/domains/{d['id']}/verify",
                        data=b"{}",
                        headers={"Authorization": f"Bearer {_resend_key}", "Content-Type": "application/json", "User-Agent": "TTT-Backend/1.0"},
                        method="POST"
                    )
                    with _ur.urlopen(_req2, timeout=10) as _resp2:
                        return {"status": "verification_triggered", "response": json.loads(_resp2.read().decode())}
            return {"error": "Domain not found in Resend"}
    except Exception as e:
        err_body = ""
        if hasattr(e, "read"):
            try: err_body = e.read().decode()
            except: pass
        return {"error": str(e), "body": err_body}








# ═══════════════════════════════════════════════════════
# ADMIN RESET — Clear all test data
# ═══════════════════════════════════════════════════════
@app.post("/api/admin/reset-all")
async def reset_all_data(key: str = ""):
    """Wipe all partners, listings, bookings, invoices for fresh testing."""
    if key != os.getenv("ADMIN_KEY", "ttt-admin-2024"):
        raise HTTPException(403, "Invalid admin key")
    _partners.clear()
    _listings.clear()
    _bookings.clear()
    _invoices.clear()
    try:
        PARTNER_WALLETS.clear()
    except: pass
    _save_db()
    return {"success": True, "message": "All data cleared"}

# ═══════════════════════════════════════════════════════
# PARTNER WALLET v2 — Real earnings from bookings
# ═══════════════════════════════════════════════════════
_partner_wallets_v2: Dict[str, dict] = _db.get("partner_wallets_v2", {})

def _credit_partner_wallet(partner_id: str, amount: float, booking_id: str, desc: str):
    """Credit partner wallet when a booking is confirmed."""
    if partner_id not in _partner_wallets_v2:
        _partner_wallets_v2[partner_id] = {"balance": 0, "total_earned": 0, "pending": 0, "transactions": []}
    w = _partner_wallets_v2[partner_id]
    commission = round(amount * 0.10, 2)  # 10% TTT commission
    net = round(amount - commission, 2)
    w["total_earned"] = round(w.get("total_earned", 0) + net, 2)
    w["pending"] = round(w.get("pending", 0) + net, 2)
    w["transactions"].insert(0, {
        "date": datetime.now().strftime("%b %d"),
        "description": desc,
        "booking": booking_id,
        "gross": amount,
        "commission": commission,
        "amount": net,
        "status": "pending"
    })
    _db["partner_wallets_v2"] = _partner_wallets_v2
    _save_db()

@app.get("/api/partner/wallet-v2/{partner_id}")
async def get_partner_wallet_v2(partner_id: str):
    """Get real wallet with earnings from bookings."""
    w = _partner_wallets_v2.get(partner_id, {"balance": 0, "total_earned": 0, "pending": 0, "transactions": []})
    return w




# ═══════════════════════════════════════════════════════
# LEADS + AI DEMAND FEED (Phase 1 — blueprint secret weapon)
# ═══════════════════════════════════════════════════════
_leads: Dict[str, list] = _db.get("leads", {})
_search_log: list = _db.get("search_log", [])

@app.get("/api/partner/{partner_id}/leads")
async def get_partner_leads(partner_id: str):
    """Qualified traveler leads matched to this partner."""
    return {"leads": _leads.get(partner_id, [])}

class LeadCreate(BaseModel):
    partner_id: str = ""
    location: str = ""
    name: str = "Traveler"
    query: str = ""
    dates: str = ""
    guests: int = 2
    budget: float = 0

@app.post("/api/leads")
async def create_lead(req: LeadCreate):
    """Create a lead — routed to partners matching the location."""
    lead = {"name": req.name, "query": req.query, "dates": req.dates,
            "guests": req.guests, "budget": req.budget, "responded": False,
            "created_at": datetime.now().isoformat()}
    # Route to partners in matching location
    targets = []
    if req.partner_id:
        targets = [req.partner_id]
    else:
        for pid, p in _partners.items():
            if req.location.lower() in (p.get("location","") or "").lower() or not req.location:
                targets.append(pid)
    for pid in targets:
        _leads.setdefault(pid, []).insert(0, lead)
    _db["leads"] = _leads
    _save_db()
    return {"success": True, "routed_to": len(targets)}

@app.post("/api/log-search")
async def log_search(q: dict):
    """Log a traveler search term for the demand feed."""
    _search_log.insert(0, {"term": q.get("term",""), "location": q.get("location","Goa"), "ts": datetime.now().isoformat()})
    del _search_log[500:]  # keep last 500
    _db["search_log"] = _search_log
    _save_db()
    return {"success": True}

@app.get("/api/demand-feed")
async def demand_feed(location: str = ""):
    """Aggregate recent searches into a demand feed."""
    from collections import Counter
    recent = _search_log[:200]
    if location:
        recent = [s for s in recent if location.lower() in (s.get("location","") or "").lower()] or recent
    counts = Counter(s.get("term","").strip() for s in recent if s.get("term","").strip())
    demand = [{"term": t, "searches": c} for t, c in counts.most_common(8)]
    return {"demand": demand}




# ═══════════════════════════════════════════════════════
# FOUNDER COMMAND CENTER — Phase 1 metrics in one call
# ═══════════════════════════════════════════════════════
@app.get("/api/founder/overview")
async def founder_overview(key: str = ""):
    """Single endpoint: traveller footfall + partner activity + Phase-1 goals."""
    if key != os.getenv("ADMIN_KEY", "ttt-admin-2024"):
        raise HTTPException(403, "Invalid key")
    from collections import Counter
    
    # Partners
    partners = list(_partners.values())
    partner_count = len(partners)
    
    # Listings
    listing_count = len(_listings)
    
    # Bookings
    bookings = list(_bookings.values())
    booking_count = len(bookings)
    confirmed = [b for b in bookings if b.get("status") == "confirmed"]
    gmv = sum(b.get("total_price", 0) for b in bookings)
    
    # Revenue (TTT commission @ 10%)
    ttt_revenue = round(gmv * 0.10, 2)
    
    # Traveller footfall (searches + unique)
    search_count = len(_search_log)
    
    # Leads
    total_leads = sum(len(v) for v in _leads.values())
    
    # Top demand
    counts = Counter(s.get("term","").strip()[:40] for s in _search_log if s.get("term","").strip())
    top_demand = [{"term": t, "searches": c} for t, c in counts.most_common(6)]
    
    # Per-partner activity
    partner_activity = []
    for pid, p in _partners.items():
        p_listings = [l for l in _listings.values() if l.get("partner_id") == pid]
        p_bookings = [b for b in bookings if b.get("partner_id") == pid]
        p_rev = sum(b.get("total_price", 0) for b in p_bookings)
        partner_activity.append({
            "id": pid,
            "name": p.get("business_name") or p.get("name"),
            "location": p.get("location", ""),
            "email": p.get("email", ""),
            "listings": len(p_listings),
            "bookings": len(p_bookings),
            "gmv": p_rev,
            "joined": (p.get("created_at") or "")[:10],
            "verified": bool(p.get("password_hash")),
        })
    partner_activity.sort(key=lambda x: x["gmv"], reverse=True)
    
    # Recent bookings
    recent_bookings = sorted(bookings, key=lambda x: x.get("created_at",""), reverse=True)[:10]
    recent = [{
        "guest": b.get("customer_name",""),
        "listing": b.get("listing_title",""),
        "partner": _partners.get(b.get("partner_id",""), {}).get("business_name",""),
        "amount": b.get("total_price",0),
        "status": b.get("status",""),
        "date": (b.get("created_at") or "")[:10],
    } for b in recent_bookings]
    
    # Phase 1 goals (from blueprint)
    goals = {
        "partners":   {"current": partner_count,  "target": 10,     "label": "Active Partners"},
        "listings":   {"current": listing_count,  "target": 50,     "label": "Live Listings"},
        "searches":   {"current": search_count,   "target": 500,    "label": "Traveler Conversations"},
        "bookings":   {"current": booking_count,  "target": 50,     "label": "Paid Bookings"},
        "gmv":        {"current": gmv,             "target": 500000, "label": "GMV (₹)"},
    }
    
    return {
        "summary": {
            "partners": partner_count, "listings": listing_count,
            "bookings": booking_count, "confirmed": len(confirmed),
            "gmv": gmv, "ttt_revenue": ttt_revenue,
            "searches": search_count, "leads": total_leads,
        },
        "goals": goals,
        "top_demand": top_demand,
        "partner_activity": partner_activity,
        "recent_bookings": recent,
    }

@app.get("/founder")
async def founder_page():
    for fp in [os.path.join(frontend_dir, "founder.html"),
               os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend", "founder.html"),
               os.path.join(os.getcwd(), "frontend", "founder.html")]:
        if os.path.exists(fp):
            return FileResponse(fp, headers={"Cache-Control": "no-store"})
    return HTMLResponse("<h2>Founder dashboard not found</h2>", 404)


# ═══════════════════════════════════════════════════════
# INVOICE ENGINE — Full CRUD + PDF + Email
# ═══════════════════════════════════════════════════════

class InvoiceCreate(BaseModel):
    partner_id: str
    guest_name: str
    guest_email: str = ""
    booking_ref: str = ""
    service: str = ""
    amount: float
    tax_percent: float = 18.0
    notes: str = ""
    status: str = "pending"  # pending, paid, overdue, cancelled
    due_date: str = ""

@app.post("/api/invoices")
async def create_invoice(req: InvoiceCreate):
    """Create a new invoice."""
    inv_num = f"INV-{len(_invoices)+1:04d}"
    tax_amount = round(req.amount * req.tax_percent / 100, 2)
    total = round(req.amount + tax_amount, 2)
    
    partner = _partners.get(req.partner_id, {})
    
    invoice = {
        "id": inv_num,
        "partner_id": req.partner_id,
        "partner_name": partner.get("business_name", partner.get("name", "")),
        "guest_name": req.guest_name,
        "guest_email": req.guest_email,
        "booking_ref": req.booking_ref,
        "service": req.service,
        "subtotal": req.amount,
        "tax_percent": req.tax_percent,
        "tax_amount": tax_amount,
        "total": total,
        "notes": req.notes,
        "status": req.status,
        "due_date": req.due_date or (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d"),
        "created_at": datetime.now().isoformat(),
        "paid_at": None,
        "emailed": False
    }
    _invoices[inv_num] = invoice
    _save_db()
    return {"success": True, "invoice": invoice}

@app.get("/api/invoices/{partner_id}")
async def get_partner_invoices(partner_id: str):
    """Get all invoices for a partner."""
    invs = [v for v in _invoices.values() if v.get("partner_id") == partner_id]
    invs.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    total_revenue = sum(i.get("total", 0) for i in invs if i.get("status") == "paid")
    total_pending = sum(i.get("total", 0) for i in invs if i.get("status") == "pending")
    total_overdue = sum(i.get("total", 0) for i in invs if i.get("status") == "overdue")
    return {
        "invoices": invs,
        "summary": {
            "count": len(invs),
            "paid": len([i for i in invs if i["status"] == "paid"]),
            "pending": len([i for i in invs if i["status"] == "pending"]),
            "overdue": len([i for i in invs if i["status"] == "overdue"]),
            "total_revenue": total_revenue,
            "total_pending": total_pending,
            "total_overdue": total_overdue
        }
    }

@app.get("/api/invoice/{invoice_id}")
async def get_invoice(invoice_id: str):
    """Get a single invoice."""
    if invoice_id not in _invoices:
        raise HTTPException(404, "Invoice not found")
    return _invoices[invoice_id]

@app.put("/api/invoice/{invoice_id}/status")
async def update_invoice_status(invoice_id: str, status: str = "paid"):
    """Update invoice status (paid, pending, overdue, cancelled)."""
    if invoice_id not in _invoices:
        raise HTTPException(404, "Invoice not found")
    _invoices[invoice_id]["status"] = status
    if status == "paid":
        _invoices[invoice_id]["paid_at"] = datetime.now().isoformat()
    _save_db()
    return {"success": True, "invoice": _invoices[invoice_id]}

@app.post("/api/invoice/{invoice_id}/email")
async def email_invoice(invoice_id: str):
    """Email invoice to guest via Resend."""
    if invoice_id not in _invoices:
        raise HTTPException(404, "Invoice not found")
    inv = _invoices[invoice_id]
    if not inv.get("guest_email"):
        return {"success": False, "error": "No guest email"}
    
    _resend_key = os.getenv("RESEND_API_KEY", "")
    if not _resend_key:
        return {"success": False, "error": "Email not configured"}
    
    html_body = f"""
    <div style="font-family:Georgia,serif;max-width:600px;margin:auto;padding:32px;background:#0A0805;color:#FEFCF8;border:1px solid rgba(201,150,58,0.2)">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:24px">
        <div><div style="font-size:1.8rem;color:#C9963A;letter-spacing:0.15em">TTT</div><div style="font-size:0.65rem;color:rgba(255,255,255,0.4)">The Trip Theory</div></div>
        <div style="text-align:right"><div style="font-size:1.1rem;font-weight:600;color:#F5EFE0">{inv['id']}</div><div style="font-size:0.75rem;color:rgba(255,255,255,0.4)">{inv['created_at'][:10]}</div></div>
      </div>
      <div style="border-top:1px solid rgba(201,150,58,0.2);padding-top:16px;margin-bottom:16px">
        <p style="color:rgba(255,255,255,0.5);font-size:0.8rem;margin-bottom:4px">Bill to</p>
        <p style="font-size:1rem;color:#F5EFE0;font-weight:600">{inv['guest_name']}</p>
        <p style="font-size:0.8rem;color:rgba(255,255,255,0.5)">{inv['guest_email']}</p>
      </div>
      <table style="width:100%;border-collapse:collapse;margin:20px 0">
        <tr style="border-bottom:1px solid rgba(201,150,58,0.15)"><td style="padding:10px 0;color:rgba(255,255,255,0.6)">Service</td><td style="padding:10px 0;text-align:right;color:#F5EFE0">{inv['service']}</td></tr>
        <tr style="border-bottom:1px solid rgba(201,150,58,0.15)"><td style="padding:10px 0;color:rgba(255,255,255,0.6)">Booking Ref</td><td style="padding:10px 0;text-align:right;color:#F5EFE0">{inv['booking_ref']}</td></tr>
        <tr style="border-bottom:1px solid rgba(201,150,58,0.15)"><td style="padding:10px 0;color:rgba(255,255,255,0.6)">Subtotal</td><td style="padding:10px 0;text-align:right;color:#F5EFE0">₹{inv['subtotal']:,.0f}</td></tr>
        <tr style="border-bottom:1px solid rgba(201,150,58,0.15)"><td style="padding:10px 0;color:rgba(255,255,255,0.6)">Tax ({inv['tax_percent']}%)</td><td style="padding:10px 0;text-align:right;color:#F5EFE0">₹{inv['tax_amount']:,.0f}</td></tr>
        <tr><td style="padding:14px 0;color:#C9963A;font-size:1.1rem;font-weight:600">Total</td><td style="padding:14px 0;text-align:right;color:#C9963A;font-size:1.3rem;font-weight:700">₹{inv['total']:,.0f}</td></tr>
      </table>
      <div style="background:rgba(201,150,58,0.08);border:1px solid rgba(201,150,58,0.15);border-radius:6px;padding:12px;margin:16px 0;text-align:center">
        <div style="font-size:0.7rem;color:rgba(255,255,255,0.4);text-transform:uppercase;letter-spacing:0.1em">Status</div>
        <div style="font-size:1rem;color:#C9963A;font-weight:600;text-transform:uppercase">{inv['status']}</div>
        <div style="font-size:0.75rem;color:rgba(255,255,255,0.4);margin-top:4px">Due: {inv['due_date']}</div>
      </div>
      {f'<p style="color:rgba(255,255,255,0.4);font-size:0.8rem">{inv["notes"]}</p>' if inv.get('notes') else ''}
      <p style="color:rgba(255,255,255,0.25);font-size:0.65rem;margin-top:24px;text-align:center">Generated by TTT Partner Portal · The Trip Theory Pvt. Ltd.</p>
    </div>
    """
    
    try:
        import urllib.request as _ur
        _payload = json.dumps({
            "from": "TTT Invoices <noreply@thetriptheory.com>",
            "to": [inv["guest_email"]],
            "subject": f"Invoice {inv['id']} — ₹{inv['total']:,.0f} — {inv.get('partner_name', 'TTT Partner')}",
            "html": html_body
        }).encode()
        _req = _ur.Request(
            "https://api.resend.com/emails",
            data=_payload,
            headers={"Authorization": f"Bearer {_resend_key}", "Content-Type": "application/json", "User-Agent": "TTT-Backend/1.0"},
            method="POST"
        )
        with _ur.urlopen(_req, timeout=10) as _resp:
            _invoices[invoice_id]["emailed"] = True
            _invoices[invoice_id]["emailed_at"] = datetime.now().isoformat()
            _save_db()
            return {"success": True, "message": f"Invoice emailed to {inv['guest_email']}"}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.delete("/api/invoice/{invoice_id}")
async def delete_invoice(invoice_id: str):
    """Delete an invoice."""
    if invoice_id in _invoices:
        del _invoices[invoice_id]
        _save_db()
    return {"success": True}


@app.get("/api/admin/resend-create-domain")
async def resend_create_domain(key: str = ""):
    """One-time: create domain in Resend using full-access key."""
    if key != os.getenv("ADMIN_KEY", "ttt-admin-2024"):
        raise HTTPException(403, "Invalid admin key")
    import urllib.request as _ur
    _full_key = "re_LgfxQnuc_3fkmApcrmNHq7Bc4jTsBpCQ2"
    try:
        _payload = json.dumps({"name": "thetriptheory.com"}).encode()
        _req = _ur.Request(
            "https://api.resend.com/domains",
            data=_payload,
            headers={"Authorization": f"Bearer {_full_key}", "Content-Type": "application/json", "User-Agent": "TTT-Backend/1.0"},
            method="POST"
        )
        with _ur.urlopen(_req, timeout=15) as _resp:
            return {"status": "created", "data": json.loads(_resp.read().decode())}
    except Exception as e:
        err_body = ""
        if hasattr(e, "read"):
            try: err_body = e.read().decode()
            except: pass
        return {"error": str(e), "body": err_body}


@app.get("/api/admin/resend-get-domain")
async def resend_get_domain(key: str = "", domain_id: str = ""):
    """Get domain DNS records from Resend."""
    if key != os.getenv("ADMIN_KEY", "ttt-admin-2024"):
        raise HTTPException(403, "Invalid admin key")
    import urllib.request as _ur
    _full_key = "re_LgfxQnuc_3fkmApcrmNHq7Bc4jTsBpCQ2"
    try:
        _req = _ur.Request(
            f"https://api.resend.com/domains/{domain_id}",
            headers={"Authorization": f"Bearer {_full_key}", "Content-Type": "application/json", "User-Agent": "TTT-Backend/1.0"},
            method="GET"
        )
        with _ur.urlopen(_req, timeout=15) as _resp:
            return json.loads(_resp.read().decode())
    except Exception as e:
        err_body = ""
        if hasattr(e, "read"):
            try: err_body = e.read().decode()
            except: pass
        return {"error": str(e), "body": err_body}


@app.get("/api/admin/resend-verify-domain")
async def resend_verify_domain_endpoint(key: str = "", domain_id: str = ""):
    """Trigger domain verification in Resend."""
    if key != os.getenv("ADMIN_KEY", "ttt-admin-2024"):
        raise HTTPException(403, "Invalid admin key")
    import urllib.request as _ur
    _full_key = "re_LgfxQnuc_3fkmApcrmNHq7Bc4jTsBpCQ2"
    try:
        _req = _ur.Request(
            f"https://api.resend.com/domains/{domain_id}/verify",
            data=b"{}",
            headers={"Authorization": f"Bearer {_full_key}", "Content-Type": "application/json", "User-Agent": "TTT-Backend/1.0"},
            method="POST"
        )
        with _ur.urlopen(_req, timeout=15) as _resp:
            return {"status": "verification_triggered", "data": json.loads(_resp.read().decode())}
    except Exception as e:
        err_body = ""
        if hasattr(e, "read"):
            try: err_body = e.read().decode()
            except: pass
        return {"error": str(e), "body": err_body}


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

@app.get("/")
async def landing_page():
    """Serve the main TTT app (landing + concierge combined)."""
    for fp in [
        os.path.join(frontend_dir, "index.html"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend", "index.html"),
        os.path.join(os.getcwd(), "frontend", "index.html"),
    ]:
        if os.path.exists(fp):
            return FileResponse(fp, headers={"Cache-Control": "no-store"})
    return HTMLResponse("<h2>App loading...</h2>", 404)


@app.get("/landing")
async def landing_alias():
    """Alias - /landing now redirects to /"""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/")



class GoogleAuthRequest(BaseModel):
    id_token: str

@app.post("/api/auth/google")
async def google_auth(req: GoogleAuthRequest):
    """Verify Google OAuth token and sign user in."""
    payload = await _verify_google_token(req.id_token)
    if not payload:
        raise HTTPException(401, "Invalid Google token")

    email = payload.get("email", "")
    name  = payload.get("name",  "")
    pic   = payload.get("picture", "")

    # Find or create user
    existing = next((u for u in _users.values() if u.get("email") == email), None)

    if existing:
        uid    = existing["id"]
        is_new = False
        # Update name/pic if missing
        if name and not existing.get("name"):
            _users[uid]["name"] = name
        log_activity(uid, "login", {"method": "google"})
    else:
        uid = "user-" + str(uuid.uuid4())[:8]
        _users[uid] = {
            "id":         uid,
            "email":      email,
            "phone":      None,
            "name":       name,
            "picture":    pic,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "auth_method": "google",
        }
        log_signup(uid, email, source="google_oauth")
        _save_db()
        log_activity(uid, "signup", {"method": "google", "email": email})
        is_new = True

    token = _create_jwt(uid)
    return {
        "success":     True,
        "is_new_user": is_new,
        "token":       token,
        "user": {
            "id":      uid,
            "email":   email,
            "name":    name,
            "picture": pic,
        },
        "message": "Welcome to TTT!" if is_new else "Welcome back!",
    }

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
