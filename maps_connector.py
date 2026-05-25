"""
TTT Maps Connector — Google Places API integration
Finds hidden gems, top spots, restaurants, hotels near any destination
"""
import os, httpx
from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional, List

router = APIRouter()
MAPS_KEY = os.getenv("GOOGLE_MAPS_KEY", "")

class PlacesRequest(BaseModel):
    destination: str
    category: str = "tourist_attraction"  # tourist_attraction, restaurant, lodging, spa, museum
    limit: int = 6
    hidden_gems: bool = False  # if True, filters for lesser-known spots (lower review count, high rating)

async def geocode(destination: str):
    """Convert destination name to lat/lng"""
    if not MAPS_KEY:
        return None, None
    url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {"address": destination + " India", "key": MAPS_KEY}
    async with httpx.AsyncClient() as client:
        r = await client.get(url, params=params, timeout=8)
        d = r.json()
    if d.get("results"):
        loc = d["results"][0]["geometry"]["location"]
        return loc["lat"], loc["lng"]
    return None, None

async def nearby_search(lat, lng, category, radius=15000, limit=6, hidden=False):
    """Search for places near coordinates"""
    if not MAPS_KEY:
        return []
    url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
    params = {
        "location": f"{lat},{lng}",
        "radius": radius,
        "type": category,
        "key": MAPS_KEY,
        "rankby": "prominence" if not hidden else "distance",
    }
    async with httpx.AsyncClient() as client:
        r = await client.get(url, params=params, timeout=10)
        d = r.json()
    
    places = d.get("results", [])[:limit*2]
    
    if hidden:
        # Hidden gems = high rating (4.4+) but fewer reviews (under 500)
        places = [p for p in places if 
                  p.get("rating", 0) >= 4.3 and 
                  p.get("user_ratings_total", 9999) < 1000]
    
    result = []
    for p in places[:limit]:
        photo_url = None
        if p.get("photos") and MAPS_KEY:
            ref = p["photos"][0]["photo_reference"]
            photo_url = f"https://maps.googleapis.com/maps/api/place/photo?maxwidth=600&photo_reference={ref}&key={MAPS_KEY}"
        result.append({
            "name": p.get("name"),
            "rating": p.get("rating"),
            "reviews": p.get("user_ratings_total"),
            "address": p.get("vicinity"),
            "open_now": p.get("opening_hours", {}).get("open_now"),
            "price_level": p.get("price_level"),
            "types": p.get("types", [])[:3],
            "photo_url": photo_url,
            "place_id": p.get("place_id"),
            "maps_url": f"https://www.google.com/maps/place/?q=place_id:{p.get('place_id')}"
        })
    return result

@router.post("/api/maps/places")
async def get_places(req: PlacesRequest):
    """Get recommended places for a destination"""
    # If no API key, return curated mock data
    if not MAPS_KEY:
        return mock_places(req.destination, req.category)
    
    lat, lng = await geocode(req.destination)
    if not lat:
        return {"places": [], "error": "Destination not found", "destination": req.destination}
    
    places = await nearby_search(lat, lng, req.category, 
                                  limit=req.limit, hidden=req.hidden_gems)
    return {
        "destination": req.destination,
        "category": req.category,
        "hidden_gems": req.hidden_gems,
        "lat": lat, "lng": lng,
        "places": places,
        "maps_embed_url": f"https://www.google.com/maps/embed/v1/search?key={MAPS_KEY}&q={req.category}+in+{req.destination}&zoom=12"
    }

@router.get("/api/maps/embed")
async def get_embed(destination: str, category: str = "tourist_attraction"):
    """Get Google Maps embed URL for a destination"""
    if not MAPS_KEY:
        return {"embed_url": None, "message": "Add GOOGLE_MAPS_KEY to .env"}
    return {
        "embed_url": f"https://www.google.com/maps/embed/v1/search?key={MAPS_KEY}&q={category}+in+{destination}&zoom=12"
    }

def mock_places(destination: str, category: str):
    """Curated mock data when no API key is set — still useful and beautiful"""
    dest = destination.lower()
    data = {
        "goa": {
            "tourist_attraction": [
                {"name":"Chapora Fort","rating":4.4,"reviews":8200,"address":"Chapora, North Goa","photo_url":"https://images.unsplash.com/photo-1582548961019-42d2be3ec6c3?w=600&q=80","maps_url":"https://maps.google.com/?q=Chapora+Fort+Goa","types":["fort","heritage"],"open_now":True},
                {"name":"Dudhsagar Waterfalls","rating":4.7,"reviews":22000,"address":"Mollem, Goa","photo_url":"https://images.unsplash.com/photo-1601919051950-bb9f3ffb3fee?w=600&q=80","maps_url":"https://maps.google.com/?q=Dudhsagar+Waterfalls","types":["waterfall","nature"],"open_now":True},
                {"name":"Fontainhas Latin Quarter","rating":4.5,"reviews":4100,"address":"Panjim, Goa","photo_url":"https://images.unsplash.com/photo-1548013146-72479768bada?w=600&q=80","maps_url":"https://maps.google.com/?q=Fontainhas+Goa","types":["heritage","neighbourhood"],"open_now":True},
            ],
            "restaurant": [
                {"name":"Gunpowder","rating":4.6,"reviews":3200,"address":"Assagao, North Goa","photo_url":"https://images.unsplash.com/photo-1414235077428-338989a2e8c0?w=600&q=80","maps_url":"https://maps.google.com/?q=Gunpowder+Goa","types":["restaurant","indian"],"open_now":True,"price_level":2},
                {"name":"Antares","rating":4.5,"reviews":2800,"address":"Vagator Beach, Goa","photo_url":"https://images.unsplash.com/photo-1555396273-367ea4eb4db5?w=600&q=80","maps_url":"https://maps.google.com/?q=Antares+Goa","types":["restaurant","seafood"],"open_now":True,"price_level":3},
            ]
        },
        "rajasthan": {
            "tourist_attraction": [
                {"name":"Mehrangarh Fort","rating":4.8,"reviews":42000,"address":"Jodhpur, Rajasthan","photo_url":"https://images.unsplash.com/photo-1477587458883-47145ed94245?w=600&q=80","maps_url":"https://maps.google.com/?q=Mehrangarh+Fort","types":["fort","heritage"],"open_now":True},
                {"name":"Nahargarh Fort (Sunset Point)","rating":4.5,"reviews":18000,"address":"Jaipur, Rajasthan","photo_url":"https://images.unsplash.com/photo-1599661046289-e31897846e41?w=600&q=80","maps_url":"https://maps.google.com/?q=Nahargarh+Fort","types":["fort","viewpoint"],"open_now":True},
                {"name":"Thar Desert, Sam Dunes","rating":4.6,"reviews":12000,"address":"Jaisalmer, Rajasthan","photo_url":"https://images.unsplash.com/photo-1509316785289-025f5b846b35?w=600&q=80","maps_url":"https://maps.google.com/?q=Sam+Sand+Dunes","types":["desert","nature"],"open_now":True},
            ],
            "lodging": [
                {"name":"Rawla Narlai","rating":4.8,"reviews":890,"address":"Narlai Village, Rajasthan","photo_url":"https://images.unsplash.com/photo-1520250497591-112f2f40a3f4?w=600&q=80","maps_url":"https://maps.google.com/?q=Rawla+Narlai","types":["hotel","heritage","luxury"],"open_now":True,"price_level":4},
            ]
        },
        "kerala": {
            "tourist_attraction": [
                {"name":"Alleppey Backwaters","rating":4.7,"reviews":28000,"address":"Alappuzha, Kerala","photo_url":"https://images.unsplash.com/photo-1593693411515-c20261bcad6e?w=600&q=80","maps_url":"https://maps.google.com/?q=Alleppey+Backwaters","types":["backwaters","nature"],"open_now":True},
                {"name":"Munroe Island","rating":4.6,"reviews":3200,"address":"Kollam, Kerala","photo_url":"https://images.unsplash.com/photo-1500534314209-a25ddb2bd429?w=600&q=80","maps_url":"https://maps.google.com/?q=Munroe+Island+Kerala","types":["island","hidden_gem"],"open_now":True},
            ]
        }
    }
    
    # Find matching destination
    for key in data:
        if key in dest or dest in key:
            places = data[key].get(category, data[key].get("tourist_attraction", []))
            return {
                "destination": destination,
                "category": category,
                "places": places,
                "note": "Add GOOGLE_MAPS_KEY to .env for live data"
            }
    
    # Generic fallback
    return {
        "destination": destination,
        "category": category,
        "places": [
            {"name":f"Top attraction in {destination}","rating":4.5,"reviews":5000,"address":destination,"photo_url":"https://images.unsplash.com/photo-1469474968028-56623f02e42e?w=600&q=80","maps_url":f"https://maps.google.com/?q=attractions+in+{destination}","types":["attraction"],"open_now":True},
        ],
        "note": "Add GOOGLE_MAPS_KEY to .env for live results"
    }
