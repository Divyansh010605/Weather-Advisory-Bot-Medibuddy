from langgraph.graph import END, START, StateGraph

from src.nodes import (
    fetch_weather_node,
    generate_advice_node,
    location_failure_node,
    match_policy_node,
    no_matching_sop_node,
    parse_failure_node,
    parse_request_node,
    resolve_location_node,
    weather_failure_node,
)
from src.state import WeatherBotState


def route_after_parse(state: WeatherBotState) -> str:
    if state.get("parse_error"):
        return "parse_failure"

    if not state.get("extracted_location"):
        return "location_failure"

    return "resolve_location"


def route_after_location(state: WeatherBotState) -> str:
    if state.get("location_error") or not state.get("resolved_location"):
        return "location_failure"

    return "fetch_weather"


def route_after_weather(state: WeatherBotState) -> str:
    if state.get("weather_fetch_error") or not state.get("weather_data"):
        return "weather_failure"

    return "match_policy"


def route_after_policy(state: WeatherBotState) -> str:
    if state.get("selected_sop"):
        return "generate_advice"

    return "no_matching_sop"


def create_weather_bot_graph():
    workflow = StateGraph(WeatherBotState)

    workflow.add_node("parse_request", parse_request_node)
    workflow.add_node("parse_failure", parse_failure_node)
    workflow.add_node("resolve_location", resolve_location_node)
    workflow.add_node("fetch_weather", fetch_weather_node)
    workflow.add_node("match_policy", match_policy_node)
    workflow.add_node("generate_advice", generate_advice_node)
    workflow.add_node("no_matching_sop", no_matching_sop_node)
    workflow.add_node("location_failure", location_failure_node)
    workflow.add_node("weather_failure", weather_failure_node)

    workflow.add_edge(START, "parse_request")

    workflow.add_conditional_edges(
        "parse_request",
        route_after_parse,
        {
            "parse_failure": "parse_failure",
            "location_failure": "location_failure",
            "resolve_location": "resolve_location",
        },
    )

    workflow.add_conditional_edges(
        "resolve_location",
        route_after_location,
        {
            "fetch_weather": "fetch_weather",
            "location_failure": "location_failure",
        },
    )

    workflow.add_conditional_edges(
        "fetch_weather",
        route_after_weather,
        {
            "match_policy": "match_policy",
            "weather_failure": "weather_failure",
        },
    )

    workflow.add_conditional_edges(
        "match_policy",
        route_after_policy,
        {
            "generate_advice": "generate_advice",
            "no_matching_sop": "no_matching_sop",
        },
    )

    for node in (
        "parse_failure",
        "generate_advice",
        "no_matching_sop",
        "location_failure",
        "weather_failure",
    ):
        workflow.add_edge(node, END)

    return workflow.compile()


graph_app = create_weather_bot_graph()
