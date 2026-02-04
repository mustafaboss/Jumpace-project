"""
PDF ingestion and Pinecone indexing for construction rules RAG.
Run once after placing PDF(s) in data/ folder.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root (same directory as this script when run from project root)
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone as PineconeClient, ServerlessSpec


def get_env(name: str) -> str:
    """Require environment variable; raise if missing."""
    val = os.getenv(name)
    if not val:
        raise RuntimeError(f"Missing required env: {name}")
    return val.strip()


def main() -> None:
    api_key = get_env("OPENAI_API_KEY")
    embed_model = get_env("EMBED_MODEL")
    pinecone_key = get_env("PINECONE_API_KEY")
    index_name = get_env("PINECONE_INDEX")

    # Project root: directory containing this script
    project_root = Path(__file__).resolve().parent
    data_dir = project_root / "data"
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    pdf_paths = list(data_dir.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDF files found in {data_dir}")

    # Load all PDFs lazily (large PDFs can take time). Preserve page numbers in metadata.
    documents = []
    for path in pdf_paths:
        loader = PyPDFLoader(str(path))
        page_count = 0
        for d in loader.lazy_load():
            d.metadata["source_file"] = path.name
            documents.append(d)
            page_count += 1
            if page_count % 25 == 0:
                print(f"Loaded {page_count} pages from {path.name} ...")
        print(f"Loaded {page_count} pages from {path.name}.")

    # Chunk: 800–1200 tokens ≈ 3200–4800 chars; overlap 150–250 tokens ≈ 600–1000 chars
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=4000,
        chunk_overlap=800,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    # Embeddings (OpenAI)
    embeddings = OpenAIEmbeddings(
        model=embed_model,
        openai_api_key=api_key,
    )
    # Dimension for text-embedding-3-small
    embed_dim = 1536

    # Pinecone: ensure index exists then upsert
    pc = PineconeClient(api_key=pinecone_key)
    # Pinecone SDK returns different shapes across versions; handle both.
    listed = pc.list_indexes()
    try:
        existing_names = set(listed.names())
    except Exception:
        try:
            existing_names = set(i["name"] for i in listed)
        except Exception:
            existing_names = set()

    if index_name not in existing_names:
        pc.create_index(
            name=index_name,
            dimension=embed_dim,
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            metric="cosine",
        )

    # Store in Pinecone (new Pinecone SDK + langchain-pinecone)
    index = pc.Index(index_name)
    vectorstore = PineconeVectorStore(index=index, embedding=embeddings)
    vectorstore.add_documents(chunks)
    print(f"Ingested {len(chunks)} chunks from {len(pdf_paths)} PDF(s) into index '{index_name}'.")


if __name__ == "__main__":
    main()
