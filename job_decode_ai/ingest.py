from rag_chain import build_chain


def ingest_file(file_path: str = "job_description.pdf") -> dict:
    result = build_chain(file_path=file_path)
    print(result)
    return result


if __name__ == "__main__":
    ingest_file()