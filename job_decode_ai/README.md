# Job Decode AI Backend

This backend provides a FastAPI service for a RAG-powered career assistant.

## Features
- Accepts a job description as raw text or a PDF/TXT file
- Builds a FAISS vector index from the uploaded content
- Answers questions about the uploaded job description
- Exposes health and reset endpoints for deployment

## Local run
```bash
pip install -r requirements.txt
uvicorn app:app --reload
```

## Environment
Create a .env file with:
```env
GROQ_API_KEY=your_groq_api_key_here
```

## Cloud deployment
This app is ready to be deployed on platforms such as Render, Railway, or Azure App Service.
