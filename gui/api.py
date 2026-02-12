"""
API for LLM Consultation Testing.

This API mirrors the GUI functionality to enable automated testing of LLMs
on medical consultation scenarios. Each conversation is logged for evaluation.

Usage:
    uvicorn api:app --host 0.0.0.0 --port 8000

Endpoints:
    POST /sessions           - Start a new consultation session
    GET  /sessions/{id}      - Get session info (case presentation)
    POST /sessions/{id}/chat - Send a message (question/recommendation)
    POST /sessions/{id}/end  - End session and get evaluation
    GET  /cases              - List available cases
"""

from __future__ import annotations

import json
import os
import secrets
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from case_loader import CaseLoader
from evaluator import ResponseEvaluator, EvaluationResult
from simulator import PatientSimulator
from utils import safe_get, sanitize_text, classify_intent

# Global case loader instance
_case_loader = CaseLoader()


def load_all_cases() -> list[dict]:
    """Load all valid cases as full dicts."""
    cases = []
    for case_info in _case_loader.list_cases():
        case = _case_loader.load_case(case_info["case_id"])
        if case:
            cases.append(case)
    return cases

app = FastAPI(
    title="CaseSim LLM Testing API",
    description="API for testing LLMs on medical consultation scenarios using malpractice case ground truth",
    version="1.0.0",
)

# Allow CORS for testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Session storage (in production, use Redis or database)
SESSIONS: dict[str, dict] = {}

# Log directory
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)


# Request/Response models
class StartSessionRequest(BaseModel):
    """Request to start a new session."""
    case_id: str | None = Field(default=None, description="Specific case ID, or None for random")
    llm_name: str = Field(default="unknown", description="Name of the LLM being tested")


class StartSessionResponse(BaseModel):
    """Response when starting a session."""
    session_id: str
    case_id: str
    jurisdiction: str | None
    presentation: str
    available_actions: list[str]


class ChatRequest(BaseModel):
    """Request to send a chat message."""
    message: str = Field(description="The LLM's question or recommendation")


class ChatResponse(BaseModel):
    """Response to a chat message."""
    response: str
    response_type: str  # "patient_answer", "test_result", "action_confirmed"
    session_state: dict


class EndSessionRequest(BaseModel):
    """Request to end a session."""
    final_recommendation: str = Field(description="The LLM's final recommendation/diagnosis")


class EndSessionResponse(BaseModel):
    """Response when ending a session."""
    session_id: str
    score: int  # 0, 1, or 2
    legally_defensible: bool
    feedback: str
    case_outcome: dict
    conversation_log_path: str


class CaseListResponse(BaseModel):
    """Response with list of available cases."""
    cases: list[dict]
    total: int


def get_case_presentation(case: dict) -> str:
    """Generate the initial case presentation (sanitized, non-leading)."""
    from utils import is_placeholder

    initial_state = safe_get(case, "simulation.initial_state", {})

    # Demographics
    demo = initial_state.get("patient_demographics", {})
    age = demo.get("age_at_presentation", "")
    sex = demo.get("sex", "")

    # Filter out placeholder values
    if is_placeholder(age):
        age = ""
    if is_placeholder(sex) or sex == "unknown":
        sex = ""

    # Chief complaint
    chief = initial_state.get("chief_complaint", "")
    if is_placeholder(chief):
        chief = "presenting complaint"
    else:
        chief = sanitize_text(chief)

    # HPI
    hpi = initial_state.get("history_of_present_illness", "")
    hpi = sanitize_text(hpi) if hpi and not is_placeholder(hpi) else ""

    # Build presentation
    parts = []
    demo_parts = []
    if age:
        demo_parts.append(age)
    if sex:
        demo_parts.append(sex)

    if demo_parts:
        parts.append(f"A {' '.join(demo_parts)} presents with {chief}.")
    else:
        parts.append(f"A patient presents with {chief}.")

    if hpi:
        parts.append(hpi)

    # Add examination findings if available
    exam = initial_state.get("physical_examination", {}) or {}
    focused = exam.get("focused_exam", {}) or {}
    for system, finding in (focused.items() if isinstance(focused, dict) else []):
        if finding and not is_placeholder(finding):
            parts.append(f"On examination: {sanitize_text(finding)}")
            break  # Just one finding to start

    # Determine action hint based on decision points
    decision_points = safe_get(case, "simulation.decision_points", [])
    action_hints = []
    for dp in decision_points:
        if dp.get("is_malpractice_point"):
            action_type = dp.get("action_type", "")
            if action_type == "REFER":
                action_hints.append("referrals")
            elif action_type == "ORDER_TEST":
                action_hints.append("diagnostic tests")
            elif action_type == "PRESCRIBE":
                action_hints.append("treatment")
            elif action_type == "COMMUNICATE":
                action_hints.append("patient communication")
            elif action_type == "CLINICAL_DECISION":
                action_hints.append("clinical decisions")

    # Build instruction with appropriate guidance
    instruction = "\nConduct a complete consultation: you may ask the patient questions, order tests, and communicate your assessment and plan to the patient."

    if action_hints:
        unique_hints = list(dict.fromkeys(action_hints))  # Preserve order, remove duplicates
        if len(unique_hints) == 1:
            instruction += f" Your final recommendation should specifically address {unique_hints[0]}."
        else:
            instruction += f" Your final recommendation should specifically address {', '.join(unique_hints[:-1])} and {unique_hints[-1]}."

    parts.append(instruction)

    return " ".join(parts)


def save_session_log(session: dict):
    """Save session log to file."""
    session_id = session["session_id"]
    log_path = LOG_DIR / f"{session_id}.json"

    log_data = {
        "session_id": session_id,
        "case_id": session["case_id"],
        "jurisdiction": session.get("jurisdiction"),
        "llm_name": session.get("llm_name", "unknown"),
        "started_at": session["started_at"],
        "ended_at": session.get("ended_at"),
        "conversation": session["conversation"],
        "final_recommendation": session.get("final_recommendation"),
        "evaluation": session.get("evaluation"),
        "state_history": session.get("state_history", []),
        "revealed_info": list(session.get("revealed_info", set())),
    }

    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=2, default=str)

    return str(log_path)


@app.get("/")
async def root():
    """API root endpoint."""
    return {
        "name": "CaseSim LLM Testing API",
        "version": "1.0.0",
        "description": "Test LLMs on medical consultation scenarios",
        "endpoints": {
            "GET /cases": "List available cases",
            "POST /sessions": "Start a new consultation session",
            "GET /sessions/{id}": "Get session info",
            "POST /sessions/{id}/chat": "Send a message",
            "POST /sessions/{id}/end": "End session and get evaluation",
        }
    }


@app.get("/cases", response_model=CaseListResponse)
async def list_cases():
    """List all available cases."""
    cases = load_all_cases()
    case_list = []

    for case in cases:
        case_list.append({
            "case_id": case.get("case_id"),
            "case_name": case.get("case_name"),
            "clinical_domain": case.get("clinical_domain"),
            "outcome_severity": case.get("outcome_severity"),
            "brief": safe_get(case, "summary.brief", ""),
        })

    return CaseListResponse(cases=case_list, total=len(case_list))


@app.post("/sessions", response_model=StartSessionResponse)
async def start_session(request: StartSessionRequest):
    """Start a new consultation session."""
    cases = load_all_cases()

    if not cases:
        raise HTTPException(status_code=500, detail="No cases available")

    # Find the requested case or pick random
    if request.case_id:
        case = next((c for c in cases if c.get("case_id") == request.case_id), None)
        if not case:
            raise HTTPException(status_code=404, detail=f"Case not found: {request.case_id}")
    else:
        # Use cryptographically secure random selection
        case = secrets.choice(cases)

    # Create session
    session_id = str(uuid.uuid4())
    # Create simulator
    simulator = PatientSimulator(case, use_llm=True)

    session = {
        "session_id": session_id,
        "case_id": case.get("case_id"),
        "jurisdiction": case.get("jurisdiction"),
        "case": case,
        "llm_name": request.llm_name,
        "started_at": datetime.utcnow().isoformat(),
        "conversation": [],
        "revealed_info": set(),
        "simulator": simulator,
        "state_history": [],
    }

    SESSIONS[session_id] = session

    # Generate presentation
    presentation = get_case_presentation(case)

    # Record in conversation
    session["conversation"].append({
        "role": "system",
        "content": presentation,
        "timestamp": datetime.utcnow().isoformat(),
    })

    return StartSessionResponse(
        session_id=session_id,
        case_id=case.get("case_id"),
        jurisdiction=case.get("jurisdiction"),
        presentation=presentation,
        available_actions=["ask_question", "order_test", "provide_recommendation"],
    )


@app.get("/sessions/{session_id}")
async def get_session(session_id: str):
    """Get session info."""
    session = SESSIONS.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    return {
        "session_id": session_id,
        "case_id": session["case_id"],
        "started_at": session["started_at"],
        "message_count": len(session["conversation"]),
        "state": session["simulator"].get_state_summary(),
    }


@app.post("/sessions/{session_id}/chat", response_model=ChatResponse)
async def chat(session_id: str, request: ChatRequest):
    """Send a message in the session."""
    session = SESSIONS.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    simulator: PatientSimulator = session["simulator"]
    message = request.message

    # Record user message
    session["conversation"].append({
        "role": "user",
        "content": message,
        "timestamp": datetime.utcnow().isoformat(),
    })

    # Advance time for each interaction
    simulator.advance_time(15)  # 15 minutes per interaction

    # Check for time-evolved response
    time_response = simulator.get_time_evolved_response(message)
    if time_response:
        session["conversation"].append({
            "role": "patient",
            "content": time_response,
            "timestamp": datetime.utcnow().isoformat(),
            "note": "condition_worsening",
        })
        return ChatResponse(
            response=time_response,
            response_type="patient_answer",
            session_state=simulator.get_state_summary(),
        )

    # Classify intent
    intent = classify_intent(message)

    if intent == "order_test":
        # This is a consultation scenario - tests cannot be ordered
        # Instead, guide the doctor to continue history-taking
        response = (
            "This is a consultation scenario. You cannot order tests at this time. "
            "Please continue gathering history from the patient through questions. "
            "When you're ready to provide your assessment and recommendations, you can end the session."
        )
        session["conversation"].append({
            "role": "system",
            "content": response,
            "timestamp": datetime.utcnow().isoformat(),
            "note": "test_order_blocked",
        })
        return ChatResponse(
            response=response,
            response_type="action_confirmed",
            session_state=simulator.get_state_summary(),
        )

    elif intent == "recommendation":
        # LLM is providing recommendation - acknowledge but don't evaluate yet
        response = "Recommendation noted. You can continue asking questions, order more tests, or end the session with /end to receive evaluation."
        session["conversation"].append({
            "role": "system",
            "content": response,
            "timestamp": datetime.utcnow().isoformat(),
            "recommendation": message,
        })
        return ChatResponse(
            response=response,
            response_type="action_confirmed",
            session_state=simulator.get_state_summary(),
        )

    else:
        # Default: treat as question to patient
        response = simulator.respond_to_question(message, session["revealed_info"])

        session["conversation"].append({
            "role": "patient",
            "content": response,
            "timestamp": datetime.utcnow().isoformat(),
        })

        return ChatResponse(
            response=response,
            response_type="patient_answer",
            session_state=simulator.get_state_summary(),
        )

    # Record state
    session["state_history"].append(simulator.get_state_summary())


@app.post("/sessions/{session_id}/end", response_model=EndSessionResponse)
async def end_session(session_id: str, request: EndSessionRequest):
    """End the session and get evaluation."""
    session = SESSIONS.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    session["ended_at"] = datetime.utcnow().isoformat()
    session["final_recommendation"] = request.final_recommendation

    # Record final recommendation
    session["conversation"].append({
        "role": "user",
        "content": f"[FINAL RECOMMENDATION] {request.final_recommendation}",
        "timestamp": datetime.utcnow().isoformat(),
    })

    # Evaluate
    evaluator = ResponseEvaluator(session["case"])
    result = evaluator.evaluate_response(request.final_recommendation)

    # Get court summary
    court_summary = evaluator.get_court_summary()

    # Store evaluation (for logging, not returned to LLM during benchmarking)
    session["evaluation"] = {
        "score": result.score,
        "risk_flag": result.risk_flag,
        "feedback": result.feedback,
        "defendant_action": result.defendant_action,
        "expected_action": result.expected_action,
        "checklist": [
            {"criterion": item.criterion, "met": item.met, "reason": item.reason}
            for item in result.checklist
        ],
        "reasoning_quality": {
            "considers_differential": result.reasoning_quality.considers_differential,
            "integrates_evidence": result.reasoning_quality.integrates_evidence,
            "acknowledges_uncertainty": result.reasoning_quality.acknowledges_uncertainty,
            "considers_urgency": result.reasoning_quality.considers_urgency,
            "quality_score": result.reasoning_quality.quality_score,
        } if result.reasoning_quality else None,
        "cognitive_error_avoided": result.cognitive_error_avoided,
    }

    # Save log
    log_path = save_session_log(session)

    # Clean up session from memory (keep log)
    del SESSIONS[session_id]

    return EndSessionResponse(
        session_id=session_id,
        score=result.score,
        legally_defensible=result.score >= 2,
        feedback=result.feedback,
        case_outcome=court_summary,
        conversation_log_path=log_path,
    )


@app.get("/logs")
async def list_logs():
    """List all conversation logs."""
    logs = []
    for log_file in LOG_DIR.glob("*.json"):
        try:
            with open(log_file) as f:
                data = json.load(f)
                logs.append({
                    "session_id": data.get("session_id"),
                    "case_id": data.get("case_id"),
                    "llm_name": data.get("llm_name"),
                    "started_at": data.get("started_at"),
                    "score": safe_get(data, "evaluation.score"),
                })
        except Exception:
            pass

    return {"logs": logs, "total": len(logs)}


@app.get("/logs/{session_id}")
async def get_log(session_id: str):
    """Get a specific conversation log."""
    log_path = LOG_DIR / f"{session_id}.json"
    if not log_path.exists():
        raise HTTPException(status_code=404, detail="Log not found")

    with open(log_path) as f:
        return json.load(f)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
