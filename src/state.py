# LangGraph state schema for Weather Advisory Bot.""

from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict


class WeatherBotState(TypedDict, total=False):
    session_id: str
    user_message: str
    history: List[Dict[str, str]]

    # Extracted intent and slot parameters
    extracted_activity: Optional[str]
    extracted_location: Optional[str]
    extracted_time: Optional[str]

    # Geocoding result
    resolved_location: Optional[Dict[str, Any]]
    location_error: Optional[str]

    # Weather result
    weather_data: Optional[Dict[str, Any]]
    weather_time_label: Optional[str]
    weather_fetch_error: Optional[str]

    # SOP matching
    matched_sops: List[Dict[str, Any]]
    selected_sop: Optional[Dict[str, Any]]

    # Final outputs
    outcome: str  # "advice" | "no_matching_sop" | "location_failure" | "weather_failure"
    final_response: str
    sop_citation: Optional[Dict[str, Any]]
    relevant_weather: Optional[Dict[str, Any]]
