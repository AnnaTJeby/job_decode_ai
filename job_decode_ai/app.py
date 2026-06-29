from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from rag_chain import answer_question, build_chain, reset_index

app = FastAPI(
    title="JobDecode AI",
    description="RAG-powered backend for analyzing job descriptions and answering career questions",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class LoadRequest(BaseModel):
    file_path: Optional[str] = Field(default=None, description="Path to a .pdf or .txt job description")
    raw_text: Optional[str] = Field(default=None, description="Raw job description text")
    temperature: float = Field(default=0.0, ge=0.0, le=1.0)


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Question about the uploaded job description")


@app.get("/")
def root() -> dict:
    return {"message": "JobDecode AI backend is running", "status": "ok"}


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "jobdecode-ai"}


@app.post("/load")
def load_input(request: LoadRequest) -> dict:
    if not request.file_path and not request.raw_text:
        raise HTTPException(status_code=400, detail="Provide either file_path or raw_text")

    result = build_chain(
        file_path=request.file_path,
        raw_text=request.raw_text,
        temperature=request.temperature,
    )
    return result


@app.post("/ask")
def ask_question_api(request: QueryRequest) -> dict:
    return answer_question(request.question)


@app.post("/reset")
def reset_backend() -> dict:
    return reset_index()