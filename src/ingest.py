"""Ingestão do PDF no pgVector.

Fluxo: PyPDFLoader -> RecursiveCharacterTextSplitter(1000/150) -> embeddings ->
PGVector, na collection do provider ativo (ver providers.get_collection_name()).
"""
import os

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_postgres import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter

from providers import PROVIDER, get_collection_name, get_embeddings

load_dotenv()

PDF_PATH = os.getenv("PDF_PATH", "document.pdf")
DATABASE_URL = os.getenv("DATABASE_URL")


def ingest_pdf():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL não configurada no .env")
    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(f"PDF não encontrado: {PDF_PATH}")

    collection = get_collection_name()
    print(f"[ingest] provider={PROVIDER} collection={collection} pdf={PDF_PATH}")

    # 1. Carrega o PDF (uma Document por página)
    docs = PyPDFLoader(PDF_PATH).load()
    print(f"[ingest] {len(docs)} páginas carregadas")

    # 2. Divide em chunks de 1000 chars com overlap de 150 (requisito do desafio)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(docs)
    print(f"[ingest] {len(chunks)} chunks gerados")

    # 3. Conecta ao pgVector. pre_delete_collection recria a collection do zero,
    #    tornando a ingestão idempotente para este dataset pequeno.
    store = PGVector(
        embeddings=get_embeddings(),
        collection_name=collection,
        connection=DATABASE_URL,
        use_jsonb=True,
        pre_delete_collection=True,
    )

    # 4. Gera embeddings e grava
    store.add_documents(chunks)
    print(f"[ingest] {len(chunks)} chunks indexados em '{collection}' ✓")


if __name__ == "__main__":
    ingest_pdf()
