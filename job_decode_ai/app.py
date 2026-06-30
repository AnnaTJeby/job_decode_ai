from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rag_chain import build_chain, answer_question
from ingest import reset_index

app = FastAPI(
    title="JobDecode AI",
    version="1.0.0",
    description="RAG-powered backend for analyzing job descriptions and answering career questions"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:7861", "http://localhost:7861"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class LoadRequest(BaseModel):
    file_path: Optional[str] = None
    raw_text: Optional[str] = None
    temperature: float = 0.0


class QueryRequest(BaseModel):
    question: str


@app.get("/")
def root():
    return {"message": "JobDecode AI backend is running", "status": "ok"}


@app.get("/health")
def health():
    return {"status": "ok", "service": "jobdecode-ai"}


@app.post("/load")
def load_input(payload: LoadRequest):
    try:
        file_path = payload.file_path
        raw_text = payload.raw_text

        if raw_text is not None:
            raw_text = raw_text.replace("\r\n", "\n").replace("\r", "\n").strip()

        if not file_path and not raw_text:
            return {
                "status": "error",
                "message": "Provide either pasted text or an uploaded file."
            }

        return build_chain(
            file_path=file_path,
            raw_text=raw_text,
            temperature=payload.temperature
        )
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


@app.post("/ask")
def ask_question_api(payload: QueryRequest):
    try:
        return answer_question(payload.question)
    except Exception as exc:
        return {
            "answer": f"Error: {str(exc)}",
            "sources": []
        }


@app.post("/reset")
def reset_backend():
    try:
        return reset_index()
    except Exception as exc:
        return {"status": "error", "message": str(exc)}