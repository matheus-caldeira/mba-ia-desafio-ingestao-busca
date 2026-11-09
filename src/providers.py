"""Fábrica de providers de IA (embeddings + LLM) selecionada por env.

Único ponto do projeto que conhece Ollama vs Google. Os scripts de ingestão e
busca são agnósticos: chamam get_embeddings()/get_llm()/get_collection_name().

Seleção via env PROVIDER:
- "google" (padrão): Gemini, via GOOGLE_API_KEY — provider validado da entrega.
- "ollama": local, best-effort para dev (modelos 3B em CPU não extraem valor de
  tabela densa de forma confiável; ver README.md).
"""
import os

from dotenv import load_dotenv

load_dotenv()

PROVIDER = os.getenv("PROVIDER", "google").strip().lower()

_VALID = {"ollama", "google"}
if PROVIDER not in _VALID:
    raise ValueError(
        f"PROVIDER inválido: {PROVIDER!r}. Use um de: {', '.join(sorted(_VALID))}."
    )


def get_embeddings():
    """Retorna o modelo de embeddings do provider ativo."""
    if PROVIDER == "ollama":
        from langchain_ollama import OllamaEmbeddings

        return OllamaEmbeddings(
            model=os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        )

    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    return GoogleGenerativeAIEmbeddings(
        model=os.getenv("GOOGLE_EMBEDDING_MODEL", "gemini-embedding-001"),
    )


def get_llm():
    """Retorna o LLM de chat do provider ativo (temperatura 0 para respostas fiéis)."""
    if PROVIDER == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=os.getenv("OLLAMA_LLM_MODEL", "llama3.2"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            temperature=0,
        )

    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=os.getenv("GOOGLE_LLM_MODEL", "gemini-2.5-flash-lite"),
        temperature=0,
    )


def get_collection_name():
    """Nome da collection PGVector, sufixado pelo provider.

    Cada provider usa espaço de embedding próprio (modelos diferentes, e às vezes
    dimensões diferentes). Isolar por collection evita misturar vetores
    incompatíveis e o erro silencioso de dimensão ao trocar de provider.
    """
    base = os.getenv("PG_VECTOR_COLLECTION_NAME", "docs")
    return f"{base}_{PROVIDER}"
