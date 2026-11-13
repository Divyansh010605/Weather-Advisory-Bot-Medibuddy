import time
from typing import Any, Dict, Optional, Tuple
import httpx


geocoding_api = "https://geocoding-api.open-meteo.com/v1/search"

# Resolve a city or place name into geographic coordinates via Open-Meteo with retry
def resolve_location(query: str, client: Optional[httpx.Client] = None ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    cleaned = (query or "").strip()
    if not cleaned:
        return None, "No location was specified in the request or conversation history."

    params = {
        "name": cleaned,
        "count": 1,
        "language": "en",
        "format": "json",
    }

    # Retry up to 2 times for transient network/rate-limit glitches
    for attempt in range(2):
        try:
            if client:
                resp = client.get(geocoding_api, params=params, timeout=12.0)
            else:
                with httpx.Client() as c:
                    resp = c.get(geocoding_api, params=params, timeout=12.0)

            if resp.status_code == 200:
                data = resp.json()
                results = data.get("results")
                if not results:
                    return None, f"Could not find coordinates for location '{cleaned}'."

                top = results[0]
                return {
                    "name": top.get("name"),
                    "latitude": top.get("latitude"),
                    "longitude": top.get("longitude"),
                    "country": top.get("country"),
                    "admin1": top.get("admin1"),
                    "timezone": top.get("timezone", "UTC"),
                }, None

            if attempt == 0 and resp.status_code in [429, 500, 502, 503, 504]:
                time.sleep(1.0)
                continue

            return None, f"Location service error: Open-Meteo returned HTTP {resp.status_code}."

        except httpx.RequestError as exc:
            if attempt == 0:
                time.sleep(1.0)
                continue
            return None, f"Network error connecting to geocoding service: {str(exc)}"
        except Exception as exc:
            return None, f"Unexpected error during location resolution: {str(exc)}"

    return None, f"Could not resolve location '{cleaned}' after retry."
