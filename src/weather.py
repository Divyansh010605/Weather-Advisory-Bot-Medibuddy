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

current_fields = [
    "temperature_2m", "relative_humidity_2m", "apparent_temperature",
    "precipitation", "weather_code", "wind_speed_10m", "wind_gusts_10m",
    "visibility", "uv_index",
]
hourly_fields = current_fields + ["precipitation_probability"]

def get_weather_desc(code: Optional[int]) -> str:
    if code is None:
        return "Unknown"
    return wmo_codes.get(code, f"Weather code {code}")

def _select_hour(hourly_times: list, time_target: Optional[str], current_time: Optional[str] = None) -> Optional[int]:
    if not hourly_times:
        return None

    target = (time_target or "today").strip().lower()
    parsed = []
    for index, value in enumerate(hourly_times):
        try:
            parsed.append((index, datetime.fromisoformat(value)))
        except (TypeError, ValueError):
            continue
    if not parsed:
        return None

    if target in {"now", "current"}:
        if current_time:
            try:
                current_dt = datetime.fromisoformat(current_time)
                return min(parsed, key=lambda item: abs((item[1] - current_dt).total_seconds()))[0]
            except ValueError:
                pass
        return parsed[0][0]

    desired_hour = None
    if "evening" in target:
        desired_hour = 18
    elif "night" in target or "tonight" in target:
        desired_hour = 21
    elif "morning" in target:
        desired_hour = 9
    elif "afternoon" in target:
        desired_hour = 14

    if target in {"today", "this day"}:
        if current_time:
            try:
                now = datetime.fromisoformat(current_time)
                today_slots = [(i, dt) for i, dt in parsed if dt.date() == now.date() and dt >= now.replace(minute=0, second=0, microsecond=0)]
                if today_slots:
                    return today_slots[0][0]
            except ValueError:
                pass
        return parsed[0][0]

    if "tomorrow" in target:
        try:
            base = datetime.fromisoformat(current_time) if current_time else parsed[0][1]
            target_date = base.date().fromordinal(base.date().toordinal() + 1)
            candidates = [(i, dt) for i, dt in parsed if dt.date() == target_date]
            if candidates:
                if desired_hour is not None:
                    return min(candidates, key=lambda item: abs(item[1].hour - desired_hour))[0]
                return min(candidates, key=lambda item: abs(item[1].hour - 12))[0]
        except ValueError:
            return None

    if desired_hour is not None:
        if current_time:
            try:
                base_date = datetime.fromisoformat(current_time).date()
                same_day = [(i, dt) for i, dt in parsed if dt.date() == base_date and dt.hour == desired_hour]
                if same_day:
                    return same_day[0][0]
            except ValueError:
                pass
        candidates = [(i, dt) for i, dt in parsed if dt.hour == desired_hour]
        return candidates[0][0] if candidates else None

    return None

def fetch_weather(
    latitude: float,
    longitude: float,
    time_target: Optional[str] = None,
    client: Optional[httpx.Client] = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[str], str]:
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": ",".join(current_fields),
        "hourly": ",".join(hourly_fields),
        "timezone": "auto",
        "forecast_days": 2,
    }

    for attempt in range(2):
        try:
            if client is not None:
                response = client.get(forcast_api_url, params=params, timeout=12.0)
            else:
                with httpx.Client() as http_client:
                    response = http_client.get(forcast_api_url, params=params, timeout=12.0)

            if response.status_code != 200:
                if attempt == 0 and response.status_code in {429, 500, 502, 503, 504}:
                    time.sleep(0.2)
                    continue
                return None, f"Open-Meteo forecast API returned HTTP {response.status_code}.", ""

            payload = response.json()
            current = payload.get("current") or {}
            hourly = payload.get("hourly") or {}
            hourly_times = hourly.get("time") or []
            selected_index = _select_hour(hourly_times, time_target or "today", current.get("time"))

            target = (time_target or "today").strip().lower()
            use_current = target in {"now", "current"}
            if selected_index is not None and not use_current:
                raw_time = hourly_times[selected_index]
                source = {
                    field: (hourly.get(field) or [None] * len(hourly_times))[selected_index]
                    for field in hourly_fields
                }
                time_label = f"Forecast for {time_target or 'today'} ({raw_time})"
            else:
                raw_time = current.get("time", "unknown")
                source = dict(current)
                source["precipitation_probability"] = None
                if selected_index is not None:
                    probability_values = hourly.get("precipitation_probability") or []
                    if selected_index < len(probability_values):
                        source["precipitation_probability"] = probability_values[selected_index]
                time_label = f"Live conditions ({raw_time})"

            weather_data = {
                "timestamp": raw_time,
                "temperature_2m": source.get("temperature_2m"),
                "apparent_temperature": source.get("apparent_temperature"),
                "relative_humidity_2m": source.get("relative_humidity_2m"),
                "precipitation": source.get("precipitation"),
                "precipitation_probability": source.get("precipitation_probability"),
                "weather_code": source.get("weather_code"),
                "weather_description": get_weather_desc(source.get("weather_code")),
                "wind_speed_10m": source.get("wind_speed_10m"),
                "wind_gusts_10m": source.get("wind_gusts_10m"),
                "visibility": source.get("visibility"),
                "uv_index": source.get("uv_index"),
            }

            # Malformed or incomplete API responses must not be treated as valid observations.
            if weather_data["temperature_2m"] is None or weather_data["weather_code"] is None:
                return None, "Open-Meteo returned an incomplete weather payload.", ""
            return weather_data, None, time_label

        except httpx.RequestError as exc:
            if attempt == 0:
                time.sleep(0.2)
                continue
            return None, f"Network connection error to weather service: {exc}", ""
        except (ValueError, KeyError, TypeError, IndexError) as exc:
            return None, f"Invalid weather response from Open-Meteo: {exc}", ""
        except Exception as exc:
            return None, f"Unexpected error during weather data processing: {exc}", ""

    return None, "Unable to retrieve weather data from Open-Meteo after retry.", ""
