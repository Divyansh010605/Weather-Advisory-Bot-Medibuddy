from typing import Any, Dict, Optional, Tuple
import httpx

from src.http_utils import TTLCache, get_with_retry

geocoding_api = "https://geocoding-api.open-meteo.com/v1/search"

# City coordinates basically never change, so cache them for a week.
_geocode_cache = TTLCache(ttl_seconds=7 * 24 * 3600, max_items=1024)


# Resolve a city or place name into geographic coordinates via Open-Meteo (cached, with backoff retry)
def resolve_location(query: str, client: Optional[httpx.Client] = None) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    cleaned = (query or "").strip()
    if not cleaned:
        return None, "No location was specified in the request or conversation history."

    # An injected client (tests) bypasses the cache so mocks stay deterministic.
    cache_key = cleaned.lower()
    if client is None:
        hit = _geocode_cache.get(cache_key)
        if hit:
            return dict(hit[0]), None

    params = {
        "name": cleaned,
        "count": 1,
        "language": "en",
        "format": "json",
    }

    try:
        resp, net_err = get_with_retry(geocoding_api, params, client=client)
        if resp is None:
            return None, f"Network error connecting to geocoding service: {net_err}"
        if resp.status_code != 200:
            return None, f"Location service error: Open-Meteo returned HTTP {resp.status_code}."

        results = resp.json().get("results")
        if not results:
            return None, f"Could not find coordinates for location '{cleaned}'."

        top = results[0]
        location = {
            "name": top.get("name"),
            "latitude": top.get("latitude"),
            "longitude": top.get("longitude"),
            "country": top.get("country"),
            "admin1": top.get("admin1"),
            "timezone": top.get("timezone", "UTC"),
        }
    except Exception as exc:
        return None, f"Unexpected error during location resolution: {str(exc)}"

    if client is None:
        _geocode_cache.set(cache_key, location)
    return dict(location), None