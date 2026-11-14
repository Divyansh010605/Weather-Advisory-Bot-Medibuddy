# Response formatting and grounding for Weather Advisory Bot

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv

env_path = Path(__file__).resolve().parent.parent / ".env"
parent_env_path = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(env_path)
load_dotenv(parent_env_path)
load_dotenv()


def format_weather_summary(weather: Dict[str, Any]) -> str:
    lines = []

    def val_or_unknown(val, unit=""):
        return f"{val}{unit}" if val is not None else "Unknown"

    lines.append(f"- **Condition:** {weather.get('weather_description', 'Unknown')}")
    lines.append(f"- **Temperature:** {val_or_unknown(weather.get('temperature_2m'), '°C')}")
    lines.append(f"- **Feels Like:** {val_or_unknown(weather.get('apparent_temperature'), '°C')}")
    lines.append(f"- **Precipitation:** {val_or_unknown(weather.get('precipitation'), ' mm')}")
    lines.append(f"- **Precipitation Probability:** {val_or_unknown(weather.get('precipitation_probability'), '%')}")
    lines.append(f"- **Wind Speed:** {val_or_unknown(weather.get('wind_speed_10m'), ' km/h')}")
    if weather.get("wind_gusts_10m") is not None:
        lines.append(f"- **Wind Gusts:** {weather.get('wind_gusts_10m')} km/h")
    if weather.get("visibility") is not None:
        lines.append(f"- **Visibility:** {weather.get('visibility')} m")
    if weather.get("uv_index") is not None:
        lines.append(f"- **UV Index:** {weather.get('uv_index')}")
    if weather.get("relative_humidity_2m") is not None:
        lines.append(f"- **Relative Humidity:** {weather.get('relative_humidity_2m')}%")

    return "\n".join(lines)


def _generate_natural_guidance(
    activity: str,
    location: str,
    severity: str,
    sop_guidance: str,
    weather_data: Dict[str, Any],) -> str:

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return sop_guidance  # fallback to raw SOP text

    from google import genai
    from google.genai import types

    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    client = genai.Client(api_key=api_key)

    # Build a concise weather snapshot for the prompt
    weather_facts = {k: v for k, v in weather_data.items() if v is not None and k != "timestamp"}

    prompt = (
        "You are a friendly weather advisory assistant. Rewrite the safety guidance below "
        "into a short, natural, conversational reply (2-4 sentences) that speaks directly "
        "about the user's specific activity. Do NOT list other activities or generic advice.\n\n"
        f"Activity: {activity}\n"
        f"Location: {location}\n"
        f"Severity: {severity}\n"
        f"Weather snapshot: {json.dumps(weather_facts)}\n"
        f"Official SOP guidance: {sop_guidance}\n\n"
        "Rules:\n"
        "- Talk ONLY about the requested activity, not other activities.\n"
        "- Be concise and warm, like a helpful friend.\n"
        "- If severity is Normal, be encouraging. If Moderate, note the concern. "
        "If High or Critical, be firm about the risk.\n"
        "- Ground your response in the actual weather numbers provided.\n"
        "- Do NOT invent facts, alerts, or policies not in the SOP guidance.\n"
        "- Return ONLY the rewritten guidance text, no markdown headers or labels."
    )

    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.7),
            )
            text = (response.text or "").strip()
            if text:
                return text
        except Exception as exc:
            err_str = str(exc)
            if ("429" in err_str or "503" in err_str or "RESOURCE_EXHAUSTED" in err_str) and "PER DAY" not in err_str.upper() and attempt < 2:
                time.sleep(3.0 * (attempt + 1))
                continue
            break

    return sop_guidance  # fallback to raw SOP text on failure


def build_advice_response(
    sop: Dict[str, Any],
    weather_data: Dict[str, Any],
    location_info: Dict[str, Any],
    time_label: str,
    activity: Optional[str] = None,) -> str:
    loc_name = location_info.get("name", "the requested location")
    country = location_info.get("country", "")
    full_loc = f"{loc_name}, {country}" if country else loc_name
    act_name = (activity or "your outdoor activity").title()

    severity = sop.get("severity", "Normal")
    sop_id = sop.get("id", "SOP-UNKNOWN")
    sop_title = sop.get("title", "")
    raw_guidance = sop.get("guidance", "")

    weather_text = format_weather_summary(weather_data)

    # Generate natural, activity-specific guidance via Gemini
    guidance = _generate_natural_guidance(
        activity=activity or "outdoor activity",
        location=full_loc,
        severity=severity,
        sop_guidance=raw_guidance,
        weather_data=weather_data,
    )

    return (
        f"### Activity Advisory: {act_name} in {full_loc}\n\n"
        f"**Applicable Safety Policy:** `[{sop_id}]` {sop_title} (Severity: **{severity}**)\n\n"
        f"#### Weather Observations ({time_label})\n"
        f"{weather_text}\n\n"
        f"#### Guidance & Recommendation\n"
        f"{guidance}\n\n"
        f"> *Note: This advisory is based on live Open-Meteo measurements evaluated against written safety policies. "
        f"Measurements alone do not constitute an official meteorological alert.*"
    )


def build_no_matching_sop_response(
    weather_data: Dict[str, Any],
    location_info: Dict[str, Any],
    time_label: str,
    activity: Optional[str] = None,) -> str:
    loc_name = location_info.get("name", "the requested location")
    country = location_info.get("country", "")
    full_loc = f"{loc_name}, {country}" if country else loc_name
    act_name = (activity or "the requested activity").title()

    weather_text = format_weather_summary(weather_data)

    return (
        f"### Weather Report: {act_name} in {full_loc}\n\n"
        f"I checked the current weather for {full_loc}, but I don't have a specific safety policy "
        f"that covers **{act_name.lower()}** under these conditions.\n\n"
        f"#### Current Weather\n"
        f"{weather_text}\n\n"
        f"Since no SOP applies, I can't make a safety recommendation for this activity. "
        f"You may want to check local advisories or use your own judgment."
    )


def build_location_failure_response(error_detail: Optional[str]) -> str:
    err = error_detail or "The specified location could not be identified."
    return (
        f"### Location Lookup Notice\n\n"
        f"Unable to resolve the requested location: {err}\n\n"
        f"Because geographic coordinates could not be retrieved, no live weather data could be fetched "
        f"and no weather-based activity conclusion can be made. Please specify a valid city or location name."
    )


def build_weather_failure_response(
    location_info: Optional[Dict[str, Any]], error_detail: Optional[str]) -> str:
    loc_name = location_info.get("name", "the requested location") if location_info else "the requested location"
    err = error_detail or "Live weather measurements could not be fetched from Open-Meteo."
    return (
        f"### Weather Service Notice for {loc_name}\n\n"
        f"Could not retrieve live weather observations: {err}\n\n"
        f"In accordance with safety grounding protocols, no weather-based activity decision or recommendation "
        f"can be provided without verified measurement data. Please try again in a few moments."
    )
