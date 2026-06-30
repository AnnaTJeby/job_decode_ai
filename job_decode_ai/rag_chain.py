import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from ingest import ingest_data, get_retriever

try:
    from langchain_groq import ChatGroq
except ImportError:
    ChatGroq = None

load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")

rag_chain = None

PROMPT_TEMPLATE = """You are JobDecode AI, an assistant that analyzes uploaded job descriptions.

Answer questions strictly from the provided context.
If the answer is not present in the context, say that it is not available in the uploaded job description.

Context:
{context}

Question: {question}

Answer:"""

prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def build_chain(file_path=None, raw_text=None, temperature=0.0):
    global rag_chain

    result = ingest_data(file_path=file_path, raw_text=raw_text)

    if result.get("status") != "success":
        return result

    retriever = get_retriever()
    if retriever is None:
        return {"status": "error", "message": "Retriever could not be initialized."}

    groq_api_key = os.getenv("GROQ_API_KEY")

    if ChatGroq is not None and groq_api_key:
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=temperature,
            api_key=groq_api_key,
        )

        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
    else:
        rag_chain = None

    result["llm_enabled"] = bool(ChatGroq is not None and groq_api_key)
    return result


def answer_question(question: str):
    retriever = get_retriever()

    if retriever is None:
        return {
            "answer": "Please load and index a job description first.",
            "sources": [],
        }

    docs = retriever.invoke(question)

    sources = []
    for i, doc in enumerate(docs, 1):
        sources.append(
            {
                "chunk": i,
                "source": doc.metadata.get("source", "unknown"),
                "snippet": doc.page_content[:300].replace("\n", " ").strip(),
            }
        )

    if rag_chain is not None:
        answer = rag_chain.invoke(question)
    else:
        top_chunk = docs[0].page_content[:800] if docs else "No relevant content found."
        answer = (
            "LLM is not configured, so here is the closest retrieved content from the job description:\n\n"
            + top_chunk
        )

    return {"answer": answer, "sources": sources}