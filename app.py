# FastAPI application entrypoint for Weather Advisory Support Bot

import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import uuid

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
load_dotenv(BASE_DIR.parent / ".env")
load_dotenv()

from src.graph import graph_app
from src.policy_matcher import get_cached_sops
import logging

logger = logging.getLogger(__name__)

SESSION_MEMORY: Dict[str, List[Dict[str, str]]] = {}

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(
    title="Weather Advisory Support Bot",
    description="Live weather and SOP-grounded outdoor activity safety decision assistant",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files directory
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    outcome: Optional[str] = None
    sop_citation: Optional[Dict[str, Any]] = None
    relevant_weather: Optional[Dict[str, Any]] = None
    session_id: str


@app.get("/", response_class=FileResponse)
def serve_index():
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="index.html not found")
    return FileResponse(index_path)


@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "weather-advisory-bot"}


@app.get("/api/sops")
def list_sops():
    sops = get_cached_sops()
    return {"total": len(sops), "sops": sops}


@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    user_msg = (request.message or "").strip()
    if not user_msg:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    # Generate or reuse session ID
    session_id = request.session_id or f"sess-{uuid.uuid4()}"
    if session_id not in SESSION_MEMORY:
        SESSION_MEMORY[session_id] = []

    history = SESSION_MEMORY[session_id]

    # Execute LangGraph workflow
    initial_state = {
        "session_id": session_id,
        "user_message": user_msg,
        "history": list(history),
    }

    try:
        final_state = graph_app.invoke(initial_state)
    except Exception:
        logger.exception("Unexpected graph execution failure")
        raise HTTPException(
            status_code=500,
            detail="An internal server error occurred. Please try again."
        )
    bot_answer = final_state.get("final_response", "No response generated.")
    sop_citation = final_state.get("sop_citation")
    relevant_weather = final_state.get("relevant_weather")

    # Update session history (retains prior context for follow-up turns)
    history.append({"role": "user", "content": user_msg})
    history.append({"role": "assistant", "content": bot_answer})

    return ChatResponse(
    answer=bot_answer,
    outcome=final_state.get("outcome"),
    sop_citation=sop_citation,
    relevant_weather=relevant_weather,
    session_id=session_id,
)


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)
