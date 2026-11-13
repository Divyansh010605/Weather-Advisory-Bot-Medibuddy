from pathlib import Path
from unittest.mock import Mock

import yaml

from src.policy_matcher import get_cached_sops, load_sops, match_sops
from src.weather import fetch_weather


def sample_weather(**overrides):
    data = {
        "temperature_2m": 24.0,
        "apparent_temperature": 25.0,
        "relative_humidity_2m": 40,
        "precipitation": 0.0,
        "precipitation_probability": 10,
        "weather_code": 1,
        "wind_speed_10m": 10.0,
        "wind_gusts_10m": 15.0,
        "visibility": 10000.0,
        "uv_index": 3.0,
    }
    data.update(overrides)
    return data


def test_required_fields_are_fail_closed():
    sop = {
        "id": "SOP-TEST-1",
        "severity": "Normal",
        "applicable_activities": ["picnic"],
        "required_fields": ["temperature_2m", "precipitation_probability"],
        "conditions": {"temperature_min": 15},
    }
    missing_probability = sample_weather()
    missing_probability["precipitation_probability"] = None
    selected, matched = match_sops("picnic", missing_probability, [sop])
    assert selected is None
    assert matched == []


def test_qualitative_outdoor_comfortable_profile_passes():
    sop = {
        "id": "SOP-QUAL-1",
        "severity": "Normal",
        "applicable_activities": ["picnic"],
        "required_fields": ["temperature_2m", "precipitation_probability", "wind_speed_10m", "uv_index", "weather_code"],
        "conditions": {"qualitative_conditions": ["outdoor_comfortable"]},
    }
    selected, _ = match_sops("picnic", sample_weather(), [sop])
    assert selected and selected["id"] == "SOP-QUAL-1"


def test_qualitative_profile_rejects_uncomfortable_conditions():
    sop = {
        "id": "SOP-QUAL-1",
        "severity": "Normal",
        "applicable_activities": ["picnic"],
        "required_fields": ["temperature_2m", "precipitation_probability", "wind_speed_10m", "uv_index", "weather_code"],
        "conditions": {"qualitative_conditions": ["outdoor_comfortable"]},
    }
    selected, _ = match_sops("picnic", sample_weather(precipitation_probability=80), [sop])
    assert selected is None


def test_missing_activity_does_not_match_wildcard():
    sop = {
        "id": "SOP-WILDCARD",
        "severity": "Critical",
        "applicable_activities": ["*"],
        "required_fields": ["weather_code"],
        "conditions": {"weather_code_in": [95]},
    }
    selected, _ = match_sops(None, sample_weather(weather_code=95), [sop])
    assert selected is None


def test_sop_cache_hot_reloads_after_file_change(tmp_path):
    policy_file = tmp_path / "sops.yaml"
    first = {"sops": [{"id": "SOP-ONE", "title": "One"}]}
    policy_file.write_text(yaml.safe_dump(first), encoding="utf-8")
    assert get_cached_sops(policy_file)[0]["id"] == "SOP-ONE"

    second = {"sops": [{"id": "SOP-TWO", "title": "Two"}]}
    policy_file.write_text(yaml.safe_dump(second), encoding="utf-8")
    # Ensure mtime changes even on filesystems with coarse timestamp resolution.
    import os, time
    os.utime(policy_file, ns=(time.time_ns(), time.time_ns() + 2_000_000))
    assert get_cached_sops(policy_file)[0]["id"] == "SOP-TWO"


def test_weather_fetch_includes_precipitation_probability():
    payload = {
        "current": {
            "time": "2030-06-01T10:00",
            "temperature_2m": 24, "relative_humidity_2m": 40,
            "apparent_temperature": 25, "precipitation": 0,
            "weather_code": 1, "wind_speed_10m": 10,
            "wind_gusts_10m": 15, "visibility": 10000, "uv_index": 3,
        },
        "hourly": {
            "time": ["2030-06-01T10:00", "2030-06-01T11:00"],
            "temperature_2m": [24, 25], "relative_humidity_2m": [40, 38],
            "apparent_temperature": [25, 26], "precipitation": [0, 0],
            "weather_code": [1, 1], "wind_speed_10m": [10, 11],
            "wind_gusts_10m": [15, 16], "visibility": [10000, 10000],
            "uv_index": [3, 4], "precipitation_probability": [10, 15],
        },
    }
    response = Mock(status_code=200)
    response.json.return_value = payload
    client = Mock()
    client.get.return_value = response

    weather, error, _ = fetch_weather(23.2, 77.4, "today", client=client)
    assert error is None
    assert weather["precipitation_probability"] == 10
    requested_hourly = client.get.call_args.kwargs["params"]["hourly"]
    assert "precipitation_probability" in requested_hourly


def test_incomplete_weather_payload_fails_closed():
    response = Mock(status_code=200)
    response.json.return_value = {"current": {}, "hourly": {"time": []}}
    client = Mock()
    client.get.return_value = response
    weather, error, _ = fetch_weather(23.2, 77.4, "now", client=client)
    assert weather is None
    assert "incomplete" in error.lower()
