# LangGraph nodes for Weather Advisory Bot workflow

import json
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load .env file from both local and parent directory
env_path = Path(__file__).resolve().parent.parent / ".env"
parent_env_path = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(env_path)
load_dotenv(parent_env_path)
load_dotenv()

from src.location import resolve_location
from src.policy_matcher import match_sops
from src.responses import (
    build_advice_response,
    build_location_failure_response,
    build_no_matching_sop_response,
    build_weather_failure_response,
)
from src.state import WeatherBotState
from src.weather import fetch_weather


class ExtractedSlots(BaseModel):
    activity: Optional[str] = Field(
        description="Canonical outdoor activity (e.g. 'cycling', 'picnic', 'construction', 'running', 'hiking', 'drone flying', 'skydiving') or null",
    )
    location: Optional[str] = Field(
        description="City or place name (e.g. 'Bhopal', 'London', 'Miami', 'Sydney', 'Tokyo', 'Jaipur'). Carry forward from history if follow-up, or null.",
    )
    time_target: Optional[str] = Field(
        description="Time period of interest (e.g. 'today', 'this evening', 'tomorrow', 'now', 'tonight') or null.",
    )


def extract_with_gemini(
    message: str, history: List[Dict[str, str]]) -> Dict[str, Optional[str]]:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY environment variable is mandatory for language-model intent parsing. "
            "Please configure GEMINI_API_KEY in your .env file or environment."
        )

    from google import genai
    from google.genai import types

    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    client = genai.Client(api_key=api_key)

    history_summary = []
    for msg in history[-6:]:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        history_summary.append(f"{role.capitalize()}: {content}")
    hist_text = "\n".join(history_summary) if history_summary else "None"

    prompt = (
        "You are an intent and slot extractor for a weather advisory support system.\n"
        "Extract the following three parameters from the user's current message and conversational history:\n"
        "1. activity: The outdoor activity being asked about (e.g., cycling, picnic, construction, running, walking, drone flying, skydiving). "
        "Resolve colloquial phrases (such as 'peddle on two wheels' -> 'cycling', 'dining al fresco' or 'laying down a blanket' -> 'picnic').\n"
        "2. location: The targeted city or location name (e.g., Bhopal, London, Miami, Sydney, Tokyo). "
        "If the user message is a follow-up inquiry (such as 'What about this evening?'), carry forward the prior location from conversation history unless changed.\n"
        "3. time_target: Timeframe of interest (e.g., 'today', 'this evening', 'tomorrow', 'now'). Default to 'today' if not specified.\n\n"
        f"Conversation History:\n{hist_text}\n\n"
        f"Current User Message: {message}"
    )

    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=ExtractedSlots,
                    temperature=0.0,
                ),
            )

            raw_text = (response.text or "").strip()
            data = json.loads(raw_text)

            return {
                "activity": data.get("activity"),
                "location": data.get("location"),
                "time_target": data.get("time_target") or "today",
            }
        except Exception as exc:
            err_str = str(exc)
            if ("503" in err_str or "UNAVAILABLE" in err_str or "429" in err_str or "RESOURCE_EXHAUSTED" in err_str) and attempt < 2:
                time.sleep(3.0 * (attempt + 1))
                continue
            raise


def parse_request_node(state: WeatherBotState) -> Dict[str, Any]:
    user_msg = state.get("user_message", "")
    history = state.get("history", [])

    extracted = extract_with_gemini(user_msg, history)

    return {
        "extracted_activity": extracted.get("activity"),
        "extracted_location": extracted.get("location"),
        "extracted_time": extracted.get("time_target") or "today",
    }


def resolve_location_node(state: WeatherBotState) -> Dict[str, Any]:
    query = state.get("extracted_location")
    if not query:
        return {
            "resolved_location": None,
            "location_error": "No city or location name was provided in the message or conversation history.",
        }

    resolved, err = resolve_location(query)
    return {
        "resolved_location": resolved,
        "location_error": err,
    }


def fetch_weather_node(state: WeatherBotState) -> Dict[str, Any]:
    loc = state.get("resolved_location")
    if not loc:
        return {
            "weather_data": None,
            "weather_fetch_error": "Coordinates unavailable for weather retrieval.",
        }

    lat = loc["latitude"]
    lon = loc["longitude"]
    time_target = state.get("extracted_time") or "today"

    weather_data, err, time_label = fetch_weather(lat, lon, time_target)
    return {
        "weather_data": weather_data,
        "weather_fetch_error": err,
        "weather_time_label": time_label,
    }


def match_policy_node(state: WeatherBotState) -> Dict[str, Any]:
    weather = state.get("weather_data")
    activity = state.get("extracted_activity")

    selected, matched = match_sops(activity, weather)
    return {
        "selected_sop": selected,
        "matched_sops": matched,
    }


def generate_advice_node(state: WeatherBotState) -> Dict[str, Any]:
    sop = state["selected_sop"]
    weather = state["weather_data"]
    loc = state["resolved_location"]
    time_label = state.get("weather_time_label", "Live conditions")
    activity = state.get("extracted_activity")

    resp_text = build_advice_response(sop, weather, loc, time_label, activity)

    citation = {
        "id": sop.get("id"),
        "title": sop.get("title"),
        "severity": sop.get("severity"),
        "category": sop.get("category"),
    }

    relevant_weather = {
        "timestamp": weather.get("timestamp"),
        "temperature_2m": weather.get("temperature_2m"),
        "apparent_temperature": weather.get("apparent_temperature"),
        "precipitation": weather.get("precipitation"),
        "weather_code": weather.get("weather_code"),
        "weather_description": weather.get("weather_description"),
        "wind_speed_10m": weather.get("wind_speed_10m"),
        "wind_gusts_10m": weather.get("wind_gusts_10m"),
        "visibility": weather.get("visibility"),
        "uv_index": weather.get("uv_index"),
    }

    return {
        "outcome": "advice",
        "final_response": resp_text,
        "sop_citation": citation,
        "relevant_weather": relevant_weather,
    }


def no_matching_sop_node(state: WeatherBotState) -> Dict[str, Any]:
    weather = state["weather_data"]
    loc = state["resolved_location"]
    time_label = state.get("weather_time_label", "Live conditions")
    activity = state.get("extracted_activity")

    resp_text = build_no_matching_sop_response(weather, loc, time_label, activity)

    relevant_weather = {
        "timestamp": weather.get("timestamp"),
        "temperature_2m": weather.get("temperature_2m"),
        "apparent_temperature": weather.get("apparent_temperature"),
        "precipitation": weather.get("precipitation"),
        "weather_code": weather.get("weather_code"),
        "weather_description": weather.get("weather_description"),
        "wind_speed_10m": weather.get("wind_speed_10m"),
        "wind_gusts_10m": weather.get("wind_gusts_10m"),
        "visibility": weather.get("visibility"),
        "uv_index": weather.get("uv_index"),
    }

    return {
        "outcome": "no_matching_sop",
        "final_response": resp_text,
        "sop_citation": None,
        "relevant_weather": relevant_weather,
    }


def location_failure_node(state: WeatherBotState) -> Dict[str, Any]:
    err = state.get("location_error")
    resp_text = build_location_failure_response(err)
    return {
        "outcome": "location_failure",
        "final_response": resp_text,
        "sop_citation": None,
        "relevant_weather": None,
    }


def weather_failure_node(state: WeatherBotState) -> Dict[str, Any]:
    loc = state.get("resolved_location")
    err = state.get("weather_fetch_error")
    resp_text = build_weather_failure_response(loc, err)
    return {
        "outcome": "weather_failure",
        "final_response": resp_text,
        "sop_citation": None,
        "relevant_weather": None,
    }
