"""Busca semântica + montagem da chain de resposta.

search_prompt() retorna uma chain LCEL: recupera os k=10 trechos mais relevantes
via similarity_search_with_score, concatena no PROMPT_TEMPLATE do enunciado
(responder só pelo contexto) e chama o LLM do provider ativo. chat.py consome via chain.invoke(pergunta).
"""
import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_postgres import PGVector

from providers import get_collection_name, get_embeddings, get_llm

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

# Prompt exigido pelo enunciado do desafio, usado na íntegra.
PROMPT_TEMPLATE = """
CONTEXTO:
{contexto}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{pergunta}

RESPONDA A "PERGUNTA DO USUÁRIO"
"""

def _get_store():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL não configurada no .env")
    return PGVector(
        embeddings=get_embeddings(),
        collection_name=get_collection_name(),
        connection=DATABASE_URL,
        use_jsonb=True,
    )


def search_prompt(question=None):
    """Monta e retorna a chain LCEL de resposta.

    A pergunta é usada tanto para recuperar o contexto quanto para preencher o
    prompt. Retorna a chain pronta para .invoke(pergunta).
    """
    # Instancia o store (e os embeddings) uma única vez, aqui, na thread com
    # event loop. Se isso fosse feito dentro do RunnableLambda, ele rodaria numa
    # thread do pool (pelo dict paralelo abaixo) sem event loop, e o cliente gRPC
    # async do Gemini quebraria com "no running event loop".
    store = _get_store()

    def retrieve_context(question: str) -> str:
        """Busca os 10 resultados mais relevantes (k=10) e concatena o conteúdo."""
        results = store.similarity_search_with_score(question, k=10)
        return "\n\n".join(doc.page_content for doc, _score in results)

    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    chain = (
        {
            "contexto": RunnableLambda(retrieve_context),
            "pergunta": RunnablePassthrough(),
        }
        | prompt
        | get_llm()
        | StrOutputParser()
    )
    return chain
