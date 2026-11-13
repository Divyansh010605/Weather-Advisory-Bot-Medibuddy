import time 
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
import httpx

forcast_api_url = "https://api.open-meteo.com/v1/forecast"

wmo_codes = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    71: "Slight snow fall",
    73: "Moderate snow fall",
    75: "Heavy snow fall",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}

def get_weather_desc(code: Optional[int]) -> str:
    if code is None:
        return "Unknown"
    return wmo_codes.get(code, f"Weather code {code}")

def target_hour(hourly_times: list, time_target: Optional[str]) -> Optional[int]:
    if not hourly_times or not time_target:
        return None

    target = time_target.lower().strip()
    if target in ["now", "current"]:
        return None  # Prefer 'current' endpoint

    # Target hour offsets
    target_hour = None
    if "evening" in target:
        target_hour = 18
    elif "night" in target:
        target_hour = 21
    elif "morning" in target:
        target_hour = 9
    elif "afternoon" in target:
        target_hour = 14

    if target_hour is None:
        return None

    # Find the first forecast entry matching the target hour
    for idx, t_str in enumerate(hourly_times):
        try:
            dt = datetime.fromisoformat(t_str)
            if dt.hour == target_hour:
                return idx
        except Exception:
            continue
    return None

# Fetch live weather or forecast from Open-Meteo for specified coordinates with retry
def fetch_weather(
    latitude: float,
    longitude: float,
    time_target: Optional[str] = None,
    client: Optional[httpx.Client] = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[str], str]:
    

    fields = [
        "temperature_2m",
        "relative_humidity_2m",
        "apparent_temperature",
        "precipitation",
        "weather_code",
        "wind_speed_10m",
        "wind_gusts_10m",
        "visibility",
        "uv_index",
    ]
    fields_str = ",".join(fields)

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": fields_str,
        "hourly": fields_str,
        "timezone": "auto",
    }

    # Retry up to 2 times for transient network/rate-limit glitches
    for attempt in range(2):
        try:
            if client:
                resp = client.get(forcast_api_url, params=params, timeout=12.0)
            else:
                with httpx.Client() as c:
                    resp = c.get(forcast_api_url, params=params, timeout=12.0)

            if resp.status_code == 200:
                data = resp.json()
                current = data.get("current") or {}
                hourly = data.get("hourly") or {}

                # Check if user asked for a specific future period (e.g., "this evening")
                target_idx = target_hour(hourly.get("time", []), time_target)

                if target_idx is not None and hourly:
                    # Extract measurements from the targeted hourly slot
                    raw_time = hourly.get("time", [])[target_idx]
                    time_label = f"Forecast for {time_target} ({raw_time})"
                    source = {k: hourly.get(k, [])[target_idx] for k in fields if k in hourly}
                else:
                    raw_time = current.get("time", "now")
                    time_label = f"Live conditions ({raw_time})"
                    source = current

                weather_data: Dict[str, Any] = {
                    "timestamp": raw_time,
                    "temperature_2m": source.get("temperature_2m"),
                    "apparent_temperature": source.get("apparent_temperature"),
                    "relative_humidity_2m": source.get("relative_humidity_2m"),
                    "precipitation": source.get("precipitation"),
                    "weather_code": source.get("weather_code"),
                    "weather_description": get_weather_desc(source.get("weather_code")),
                    "wind_speed_10m": source.get("wind_speed_10m"),
                    "wind_gusts_10m": source.get("wind_gusts_10m"),
                    "visibility": source.get("visibility"),
                    "uv_index": source.get("uv_index"),
                }

                return weather_data, None, time_label

            if attempt == 0 and resp.status_code in [429, 500, 502, 503, 504]:
                time.sleep(0.5)
                continue

            return None, f"Open-Meteo forecast API returned HTTP {resp.status_code}.", ""

        except httpx.RequestError as exc:
            if attempt == 0:
                time.sleep(0.5)
                continue
            return None, f"Network connection error to weather service: {str(exc)}", ""
        except Exception as exc:
            return None, f"Unexpected error during weather data processing: {str(exc)}", ""

    return None, "Unable to retrieve weather data from Open-Meteo after retry.", ""
