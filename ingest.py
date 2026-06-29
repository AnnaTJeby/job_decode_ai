from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

def ingest_pdf(pdf_path="job_description.pdf"):
    pdf_file = Path(pdf_path)

    if not pdf_file.exists():
        print(f"PDF not found: {pdf_file}")
        print("Please add job_description.pdf to the project folder and try again.")
        return

    loader = PyPDFLoader(str(pdf_file))
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(docs)

    print(f"Total chunks: {len(chunks)}")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local("faiss_index")

    print("FAISS index saved.")

if __name__ == "__main__":
    ingest_pdf()