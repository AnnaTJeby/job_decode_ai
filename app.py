from typing import Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag_chain import build_chain, answer_question

app = FastAPI(title="JobDecode AI", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
    return {"message": "JobDecode AI backend is running"}

@app.post("/load")
def load_input(request: LoadRequest):
    status = build_chain(
        file_path=request.file_path,
        raw_text=request.raw_text,
        temperature=request.temperature
    )
    return {"status": status}

@app.post("/ask")
def ask_question_api(request: QueryRequest):
    return answer_question(request.question)