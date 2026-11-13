# SOP policy matching engine for weather advisory

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import yaml


SEVERITY_WEIGHTS = {
    "Critical": 4,
    "High": 3,
    "Moderate": 2,
    "Normal": 1,
}

_SOPS_CACHE: Optional[List[Dict[str, Any]]] = None


def load_sops(sops_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    global _SOPS_CACHE
    if sops_path is None:
        # Default relative to this file
        sops_path = Path(__file__).resolve().parent.parent / "policies" / "sops.yaml"

    with open(sops_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
        sops = data.get("sops", [])
        _SOPS_CACHE = sops
        return sops


def get_cached_sops() -> List[Dict[str, Any]]:
    global _SOPS_CACHE
    if _SOPS_CACHE is None:
        return load_sops()
    return _SOPS_CACHE


def _matches_activity(user_activity: Optional[str], applicable_activities: List[str]) -> Tuple[bool, bool]:
    if "*" in applicable_activities:
        return True, False

    if not user_activity:
        return False, False

    user_act = user_activity.lower().strip()
    for act in applicable_activities:
        act_lower = act.lower().strip()
        if user_act == act_lower or act_lower in user_act or user_act in act_lower:
            return True, True

    return False, False


def _evaluate_numeric_conditions(conditions: Dict[str, Any], weather_data: Dict[str, Any]) -> bool:
    # 1. Temperature checks
    if "temperature_min" in conditions:
        temp = weather_data.get("temperature_2m")
        if temp is None or temp < conditions["temperature_min"]:
            return False

    if "temperature_max" in conditions:
        temp = weather_data.get("temperature_2m")
        if temp is None or temp > conditions["temperature_max"]:
            return False

    # 2. Apparent Temperature checks
    if "apparent_temperature_min" in conditions:
        app_temp = weather_data.get("apparent_temperature")
        if app_temp is None or app_temp < conditions["apparent_temperature_min"]:
            return False

    # 3. Wind speed checks
    if "wind_speed_min" in conditions:
        wind = weather_data.get("wind_speed_10m")
        if wind is None or wind < conditions["wind_speed_min"]:
            return False

    if "wind_speed_max" in conditions:
        wind = weather_data.get("wind_speed_10m")
        if wind is None or wind > conditions["wind_speed_max"]:
            return False

    # 4. Wind gusts checks
    if "wind_gusts_min" in conditions:
        gusts = weather_data.get("wind_gusts_10m")
        if gusts is None or gusts < conditions["wind_gusts_min"]:
            return False

    # 5. Precipitation checks
    if "precipitation_min" in conditions:
        precip = weather_data.get("precipitation")
        if precip is None or precip < conditions["precipitation_min"]:
            return False

    if "precipitation_max" in conditions:
        precip = weather_data.get("precipitation")
        if precip is None or precip > conditions["precipitation_max"]:
            return False

    # 6. Visibility checks
    if "visibility_min" in conditions:
        vis = weather_data.get("visibility")
        if vis is None or vis < conditions["visibility_min"]:
            return False

    if "visibility_max" in conditions:
        vis = weather_data.get("visibility")
        if vis is None or vis > conditions["visibility_max"]:
            return False

    # 7. UV Index checks
    if "uv_index_min" in conditions:
        uv = weather_data.get("uv_index")
        if uv is None or uv < conditions["uv_index_min"]:
            return False

    if "uv_index_max" in conditions:
        uv = weather_data.get("uv_index")
        if uv is None or uv > conditions["uv_index_max"]:
            return False

    # 8. Weather code inclusion / exclusion
    w_code = weather_data.get("weather_code")
    if "weather_code_in" in conditions:
        if w_code is None or w_code not in conditions["weather_code_in"]:
            return False

    if "weather_code_not_in" in conditions:
        if w_code is not None and w_code in conditions["weather_code_not_in"]:
            return False

    return True


def match_sops(
    user_activity: Optional[str],
    weather_data: Optional[Dict[str, Any]],
    sops: Optional[List[Dict[str, Any]]] = None,) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:

    if not weather_data:
        return None, []

    if sops is None:
        sops = get_cached_sops()

    matched: List[Dict[str, Any]] = []

    for sop in sops:
        applicable = sop.get("applicable_activities", [])
        is_act_match, is_exact = _matches_activity(user_activity, applicable)
        if not is_act_match:
            continue

        conditions = sop.get("conditions", {})
        if _evaluate_numeric_conditions(conditions, weather_data):
            # Annotate with specificity for tie-breaking
            sop_entry = dict(sop)
            sop_entry["_is_exact_activity"] = is_exact
            sop_entry["_severity_weight"] = SEVERITY_WEIGHTS.get(sop.get("severity", "Normal"), 1)
            sop_entry["_priority"] = sop.get("priority", 0)
            matched.append(sop_entry)

    if not matched:
        return None, []
    # Sort matching SOPs:
    # 1st key: Severity weight descending (Critical > High > Moderate > Normal)
    # 2nd key: Activity specificity descending (exact activity over wildcard)
    # 3rd key: Explicit priority descending
    matched.sort(
        key=lambda s: (s["_severity_weight"], 1 if s["_is_exact_activity"] else 0, s["_priority"]),
        reverse=True,
    )

    selected = matched[0]
    return selected, matched
