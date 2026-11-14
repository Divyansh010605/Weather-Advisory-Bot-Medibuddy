# LangGraph nodes for Weather Advisory Bot workflow

import json
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional
import random
import logging

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
logger = logging.getLogger(__name__)


class LLMUnavailableError(Exception):
    pass

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

class GeminiUnavailableError(Exception):
    pass


class GeminiConfigurationError(Exception):
    pass


class GeminiResponseError(Exception):
    pass



def extract_with_gemini(
    message: str,
    history: List[Dict[str, str]]
) -> Dict[str, Optional[str]]:

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError("Gemini API key is not configured.")

    from google import genai
    from google.genai import types

    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            timeout=15000
        )
    )

    history_summary = []

    for msg in history[-6:]:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        history_summary.append(f"{role.capitalize()}: {content}")

    hist_text = (
        "\n".join(history_summary)
        if history_summary
        else "None"
    )

    prompt = (
        "You are an intent and slot extractor for a weather advisory support system.\n"
        "Extract three parameters:\n"
        "1. activity: Canonical outdoor activity or null.\n"
        "2. location: City or place name or null.\n"
        "3. time_target: Time period, default to today.\n"
        "Resolve colloquial activity descriptions.\n"
        "Carry forward previous activity and location for follow-up questions "
        "unless the user changes them.\n\n"
        f"Conversation History:\n{hist_text}\n\n"
        f"Current User Message: {message}"
    )

    last_error = None

    for attempt in range(2):
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

            if not raw_text:
                raise ValueError("Gemini returned an empty response.")

            data = json.loads(raw_text)

            slots = ExtractedSlots.model_validate({
                "activity": data.get("activity"),
                "location": data.get("location"),
                "time_target": data.get("time_target") or "today",
            })

            logger.info("Gemini extraction successful")

            return slots.model_dump()

        except Exception as exc:
            last_error = exc
            err_str = str(exc).upper()

            retryable = any(
                marker in err_str
                for marker in (
                    "503",
                    "UNAVAILABLE",
                    "429",
                    "RESOURCE_EXHAUSTED",
                    "500",
                    "502",
                    "504",
                    "INTERNAL",
                )
            )

            if not retryable:
                raise

            if attempt == 0:
                delay = 1.5 + random.uniform(0, 0.5)

                logger.warning(
                    "Gemini temporary failure. Retrying in %.2f seconds.",
                    delay,
                )

                time.sleep(delay)

    logger.warning("Gemini failed after 2 attempts: %r", last_error)

    raise RuntimeError(
        "Gemini is temporarily unavailable."
    ) from last_error



def extract_with_groq(
    message: str,
    history: List[Dict[str, str]]
) -> Dict[str, Optional[str]]:

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        logger.error("Groq API key is not configured.")
        raise LLMUnavailableError("Groq API key is not configured.")

    from groq import Groq

    model_name = os.getenv(
        "GROQ_MODEL",
        "openai/gpt-oss-120b"
    )

    client = Groq(
        api_key=api_key,
        timeout=20.0,
        max_retries=1,
    )

    history_summary = []

    for msg in history[-6:]:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if content:
            history_summary.append(
                f"{role.capitalize()}: {content}"
            )

    hist_text = (
        "\n".join(history_summary)
        if history_summary
        else "None"
    )

    system_prompt = (
        "You are an intent and slot extractor for a weather "
        "advisory system.\n"
        "Extract exactly three fields: activity, location, "
        "time_target.\n"
        "Resolve colloquial descriptions into canonical "
        "outdoor activities.\n"
        "For follow-up messages, carry forward the previous "
        "activity and location when the user has not changed them.\n"
        "Use null when a field cannot be determined.\n"
        "Use today as the default time_target if no timeframe "
        "is specified.\n"
        "Do not invent locations or activities.\n"
        "Return only a valid JSON object with exactly these keys:\n"
        '{"activity": null, "location": null, "time_target": "today"}'
    )

    user_prompt = (
        f"Conversation History:\n{hist_text}\n\n"
        f"Current User Message: {message}"
    )

    logger.info(
        "Groq request started | model=%s | history_messages=%d",
        model_name,
        len(history_summary),
    )

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=250,
        )

        logger.info("Groq response received")

        if not response.choices:
            raise ValueError("Groq returned no response choices.")

        raw_text = response.choices[0].message.content

        if not raw_text:
            raise ValueError("Groq returned an empty response.")

        logger.info("Groq response content received")

        data = json.loads(raw_text)

        if not isinstance(data, dict):
            raise ValueError("Groq response must be a JSON object.")

        slots = ExtractedSlots.model_validate({
            "activity": data.get("activity"),
            "location": data.get("location"),
            "time_target": data.get("time_target") or "today",
        })

        result = slots.model_dump()

        logger.info(
            "Groq intent extraction successful | activity=%s | location=%s | time_target=%s",
            result.get("activity"),
            result.get("location"),
            result.get("time_target"),
        )

        return result

    except Exception:
        logger.exception(
            "Groq intent extraction failed | model=%s",
            model_name,
        )
        raise



def parse_request_node(state: WeatherBotState) -> Dict[str, Any]:
    user_msg = state.get("user_message", "")
    history = state.get("history", [])

    try:
        extracted = extract_with_fallback(user_msg, history)

        return {
            "extracted_activity": extracted.get("activity"),
            "extracted_location": extracted.get("location"),
            "extracted_time": extracted.get("time_target") or "today",
            "parse_error": None,
        }

    except LLMUnavailableError as exc:
        return {
            "parse_error": str(exc),
        }

def parse_failure_node(state: WeatherBotState) -> Dict[str, Any]:
    return {
        "outcome": "llm_failure",
        "final_response": state.get(
            "parse_error",
            "The AI service is temporarily unavailable. Please try again."
        ),
        "sop_citation": None,
        "relevant_weather": None,
    }

def extract_with_fallback(
    message: str,
    history: List[Dict[str, str]]
) -> Dict[str, Optional[str]]:

    gemini_key = os.getenv("GEMINI_API_KEY")
    groq_key = os.getenv("GROQ_API_KEY")

    if gemini_key:
        try:
            logger.info("Attempting intent extraction with Gemini")

            result = extract_with_gemini(message, history)

            logger.info("Gemini intent extraction successful")
            return result

        except Exception as exc:
            logger.warning(
                "Gemini failed. Switching to Groq. Error type: %s",
                type(exc).__name__,
            )
    else:
        logger.warning("Gemini key missing. Switching directly to Groq.")

    if groq_key:
        try:
            logger.info("DEBUG: Entering extract_with_groq()")

            result = extract_with_groq(message, history)

            logger.info("DEBUG: Returned successfully from extract_with_groq()")
            return result

        except Exception as exc:
            logger.exception(
                "Groq fallback also failed. Error type: %s",
                type(exc).__name__,
            )

    raise LLMUnavailableError(
        "Both AI providers are currently unavailable. "
        "Please try again in a few moments."
    )

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
        "precipitation_probability": weather.get("precipitation_probability"),
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
        "precipitation_probability": weather.get("precipitation_probability"),
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
