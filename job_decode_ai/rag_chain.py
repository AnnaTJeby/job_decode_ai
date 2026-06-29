import os
import shutil
from pathlib import Path
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings

try:
    from langchain_groq import ChatGroq
except ImportError:  # pragma: no cover
    ChatGroq = None

load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")

INDEX_PATH = Path("faiss_index")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

rag_chain = None
retriever_global = None

PROMPT_TEMPLATE = """You are JobDecode AI, an assistant that analyzes uploaded job descriptions.

Answer questions strictly from the provided context. If the answer is not present in the context, say that it is not available in the uploaded job description.

Context:
{context}

Question: {question}

Answer:"""

prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def load_documents_from_source(file_path=None, raw_text=None):
    if raw_text and raw_text.strip():
        return [
            Document(
                page_content=raw_text.strip(),
                metadata={"source": "typed_input"},
            )
        ]

    if file_path:
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        suffix = path.suffix.lower()

        if suffix == ".pdf":
            loader = PyPDFLoader(str(path))
        elif suffix in {".txt", ".md"}:
            loader = TextLoader(str(path), encoding="utf-8")
        else:
            raise ValueError("Unsupported file type. Use .pdf, .txt, or .md")

        return loader.load()

    raise ValueError("Provide either raw_text or file_path.")


def build_chain(file_path=None, raw_text=None, temperature=0.0):
    global rag_chain, retriever_global

    try:
        docs = load_documents_from_source(file_path=file_path, raw_text=raw_text)
    except Exception as exc:
        return {"status": "error", "message": f"Failed to load input: {exc}"}

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    if not chunks:
        return {"status": "error", "message": "No content could be extracted from the input."}

    store = FAISS.from_documents(chunks, embeddings)
    store.save_local(str(INDEX_PATH))

    retriever_global = store.as_retriever(search_kwargs={"k": 4})

    groq_api_key = os.getenv("GROQ_API_KEY")
    if ChatGroq and groq_api_key:
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=temperature,
            api_key=groq_api_key,
        )
        rag_chain = (
            {"context": retriever_global | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
    else:
        rag_chain = None

    return {
        "status": "success",
        "message": f"Indexed {len(chunks)} chunks successfully.",
        "chunks": len(chunks),
        "llm_enabled": bool(ChatGroq and groq_api_key),
    }


def answer_question(question: str):
    if retriever_global is None:
        return {
            "answer": "Please load and index a job description first.",
            "sources": [],
        }

    docs = retriever_global.invoke(question)
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
            "The LLM is not available right now, so here is the closest retrieved content from the job description:\n\n"
            + top_chunk
        )

    return {"answer": answer, "sources": sources}


def reset_index():
    global rag_chain, retriever_global
    rag_chain = None
    retriever_global = None
    if INDEX_PATH.exists():
        shutil.rmtree(INDEX_PATH, ignore_errors=True)
    return {"status": "success", "message": "Index cleared."}


if __name__ == "__main__":
    choice = input("Type 1 for pasted text, 2 for file input: ").strip()

    if choice == "1":
        print("Paste the job description:")
        raw_text = input(">> ").strip()
        status = build_chain(raw_text=raw_text, temperature=0.0)
    else:
        file_path = input("Enter file path (.pdf or .txt): ").strip()
        status = build_chain(file_path=file_path, temperature=0.0)

    print(status)

    if status.get("status") == "success":
        query = input("Ask a question: ").strip()
        result = answer_question(query)
        print("\nAnswer:\n", result["answer"])