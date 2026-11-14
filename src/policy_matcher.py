# SOP policy matching engine for weather advisory

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import yaml
from threading import RLock

SEVERITY_WEIGHTS = {
    "Critical": 4,
    "High": 3,
    "Moderate": 2,
    "Normal": 1,
}

_SOPS_CACHE: Optional[List[Dict[str, Any]]] = None
_DEFAULT_SOPS_PATH = Path(__file__).resolve().parent.parent / "policies" / "sops.yaml"
_SOPS_MTIME: Optional[int] = None
_SOPS_PATH: Optional[Path] = None
_CACHE_LOCK = RLock()


def load_sops(sops_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    global _SOPS_CACHE, _SOPS_MTIME, _SOPS_PATH
    path = Path(sops_path or _DEFAULT_SOPS_PATH).resolve()
    with _CACHE_LOCK:
        with path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        sops = payload.get("sops")
        if not isinstance(sops, list):
            raise ValueError(f"Invalid SOP configuration: 'sops' must be a list in {path}")
        ids = [item.get("id") for item in sops if isinstance(item, dict)]
        if len(ids) != len(sops) or any(not value for value in ids) or len(set(ids)) != len(ids):
            raise ValueError(f"Invalid SOP configuration: every SOP must have a unique non-empty id in {path}")
        _SOPS_CACHE = sops
        _SOPS_PATH = path
        _SOPS_MTIME = path.stat().st_mtime_ns
        return list(_SOPS_CACHE)


def get_cached_sops(sops_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    global _SOPS_CACHE, _SOPS_MTIME, _SOPS_PATH
    path = Path(sops_path or _DEFAULT_SOPS_PATH).resolve()
    with _CACHE_LOCK:
        try:
            current_mtime = path.stat().st_mtime_ns
        except OSError as exc:
            raise RuntimeError(f"SOP policy file is unavailable: {path}") from exc
        if _SOPS_CACHE is None or _SOPS_PATH != path or _SOPS_MTIME != current_mtime:
            return load_sops(path)
        return list(_SOPS_CACHE)


def _matches_activity(user_activity: Optional[str], applicable_activities: List[str]) -> Tuple[bool, bool]:
    if not user_activity:
        return False, False

    if "*" in applicable_activities:
        return True, False

    user_act = user_activity.lower().strip()
    for act in applicable_activities:
        act_lower = act.lower().strip()
        if user_act == act_lower or act_lower in user_act or user_act in act_lower:
            return True, True

    return False, False

def _required_fields_present(sop: Dict[str, Any], weather: Dict[str, Any]) -> bool:
    for field in sop.get("required_fields", []):
        if field not in weather or weather[field] is None:
            return False
    return True

def _evaluate_qualitative_conditions(conditions: Dict[str, Any], weather: Dict[str, Any]) -> bool:
    """Evaluate named qualitative profiles using explicit measurable thresholds."""
    profiles = conditions.get("qualitative_conditions", [])
    if isinstance(profiles, dict):
        profiles = [name for name, enabled in profiles.items() if enabled]
    if not profiles:
        return True

    for profile in profiles:
        if profile == "outdoor_comfortable":
            required = (
                "temperature_2m", "precipitation_probability",
                "wind_speed_10m", "uv_index", "weather_code",
            )
            if any(weather.get(field) is None for field in required):
                return False
            if not (18 <= weather["temperature_2m"] <= 29):
                return False
            if weather["precipitation_probability"] > 30:
                return False
            if weather["wind_speed_10m"] > 20:
                return False
            if weather["uv_index"] > 6:
                return False
            if weather["weather_code"] not in [0, 1, 2, 3]:
                return False
        else:
            # Unknown qualitative rule names fail closed.
            return False
    return True

def _evaluate_numeric_conditions(conditions: Dict[str, Any], weather_data: Dict[str, Any]) -> bool:
    numeric_fields = {
        "temperature_min": ("temperature_2m", "min"),
        "temperature_max": ("temperature_2m", "max"),
        "apparent_temperature_min": ("apparent_temperature", "min"),
        "wind_speed_min": ("wind_speed_10m", "min"),
        "wind_speed_max": ("wind_speed_10m", "max"),
        "wind_gusts_min": ("wind_gusts_10m", "min"),
        "precipitation_min": ("precipitation", "min"),
        "precipitation_max": ("precipitation", "max"),
        "precipitation_probability_min": ("precipitation_probability", "min"),
        "precipitation_probability_max": ("precipitation_probability", "max"),
        "visibility_min": ("visibility", "min"),
        "visibility_max": ("visibility", "max"),
        "uv_index_min": ("uv_index", "min"),
        "uv_index_max": ("uv_index", "max"),
    }
    for condition, (field, comparator) in numeric_fields.items():
        if condition not in conditions:
            continue
        actual = weather_data.get(field)
        if actual is None:
            return False
        threshold = conditions[condition]
        if comparator == "min" and actual < threshold:
            return False
        if comparator == "max" and actual > threshold:
            return False

    code = weather_data.get("weather_code")
    if "weather_code_in" in conditions:
        if code is None or code not in conditions["weather_code_in"]:
            return False
    if "weather_code_not_in" in conditions:
        if code is None or code in conditions["weather_code_not_in"]:
            return False

    return _evaluate_qualitative_conditions(conditions, weather_data)



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
        activities = sop.get("applicable_activities", [])
        is_activity_match, is_exact = _matches_activity(user_activity, activities)
        if not is_activity_match:
            continue
        if not _required_fields_present(sop, weather_data):
            continue
        if not _evaluate_numeric_conditions(sop.get("conditions", {}), weather_data):
            continue

        annotated = dict(sop)
        annotated["_is_exact_activity"] = is_exact
        annotated["_severity_weight"] = SEVERITY_WEIGHTS.get(sop.get("severity", "Normal"), 1)
        annotated["_priority"] = sop.get("priority", 0)
        matched.append(annotated)

    if not matched:
        return None, []
    matched.sort(
        key=lambda item: (
            item["_severity_weight"],
            1 if item["_is_exact_activity"] else 0,
            item["_priority"],
        ),
        reverse=True,
    )
    return matched[0], matched