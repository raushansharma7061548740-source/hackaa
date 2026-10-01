import io
import re
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from langgraph.types import Command
from pydantic import BaseModel
from pypdf import PdfReader

from accessibility_graph import MAX_CHARS, RULES, build_app

BASE = Path(__file__).parent
FRONTEND = BASE.parent / "frontend"

app = FastAPI(title="AI Accessibility Assistant")
graph = build_app()


class StartRequest(BaseModel):
    source: str
    profile: str = "dyslexia"


class ResumeRequest(BaseModel):
    thread_id: str
    answer: str


def parse_glossary(raw: str) -> list[dict]:
    items = []
    for line in (raw or "").splitlines():
        if "::" in line:
            term, meaning = line.split("::", 1)
            items.append({"term": term.strip(" -*•"), "meaning": meaning.strip()})
    return items


def to_response(thread_id: str, out: dict) -> dict:
    if "__interrupt__" in out:
        info = out["__interrupt__"][0].value
        return {
            "status": "needs_review",
            "thread_id": thread_id,
            "simplified": info["simplified"],
            "verdict": info["verdict"],
            "verifier_notes": info.get("verifier_notes", ""),
            "grade_before": out.get("grade_before"),
        }
    return {
        "status": "done",
        "thread_id": thread_id,
        "simplified": out["simplified"],
        "glossary": parse_glossary(out.get("glossary", "")),
        "grade_before": out.get("grade_before"),
        "grade_after": out.get("grade_after"),
    }


@app.get("/api/health")
def health():
    return {"ok": True, "profiles": list(RULES.keys())}


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    name = (file.filename or "").lower()
    data = await file.read()
    if name.endswith(".pdf"):
        try:
            reader = PdfReader(io.BytesIO(data))
            text = "\n".join(p.extract_text() or "" for p in reader.pages)
        except Exception as e:
            raise HTTPException(400, f"Could not read this PDF: {e}")
    elif name.endswith(".txt"):
        text = data.decode("utf-8", errors="ignore")
    else:
        raise HTTPException(400, "Only .pdf and .txt files are supported.")
    if not text.strip():
        raise HTTPException(400, "No text found. This may be a scanned PDF.")
    return {"filename": file.filename, "text": text[:MAX_CHARS], "truncated": len(text) > MAX_CHARS}


@app.post("/api/start")
def start(req: StartRequest):
    source = req.source.strip()
    if not source:
        raise HTTPException(400, "Please paste some text, a link, or upload a file.")
    if req.profile not in RULES:
        raise HTTPException(400, f"Unknown profile. Choose one of: {list(RULES)}")
    if re.fullmatch(r"\S+\.(pdf|txt)", source, re.IGNORECASE):
        raise HTTPException(400, "Use the file upload for PDF/TXT files.")

    thread_id = uuid.uuid4().hex
    config = {"configurable": {"thread_id": thread_id}}
    try:
        out = graph.invoke({"source": source, "profile": req.profile}, config)
    except Exception as e:
        raise HTTPException(500, f"Model error. Check your API key. Details: {e}")
    return to_response(thread_id, out)


@app.post("/api/resume")
def resume(req: ResumeRequest):
    config = {"configurable": {"thread_id": req.thread_id}}
    if not graph.get_state(config).next:
        raise HTTPException(404, "No paused session found. Please start again.")
    try:
        out = graph.invoke(Command(resume=req.answer.strip() or "approve"), config)
    except Exception as e:
        raise HTTPException(500, f"Model error: {e}")
    return to_response(req.thread_id, out)


@app.get("/")
def index():
    return FileResponse(FRONTEND / "index.html")