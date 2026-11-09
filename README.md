# Ingestão e Busca Semântica com LangChain e PostgreSQL

Entrega do desafio técnico **Ingestão e Busca Semântica** do MBA em Engenharia de
Software com IA (Full Cycle), feita a partir do fork de
[`devfullcycle/mba-ia-desafio-ingestao-busca`](https://github.com/devfullcycle/mba-ia-desafio-ingestao-busca).

Sistema de RAG (Retrieval-Augmented Generation) que **ingere um PDF** no PostgreSQL
com pgVector e responde perguntas via **CLI** usando **apenas** o conteúdo do
documento. Quando a resposta não está no PDF, o sistema recusa explicitamente.

```
PERGUNTA: Qual o faturamento da Empresa SuperTechIABrazil?
RESPOSTA: R$ 10.000.000,00

PERGUNTA: Quantos clientes temos em 2024?
RESPOSTA: Não tenho informações necessárias para responder sua pergunta.
```

## Arquitetura

```
document.pdf
   │  PyPDFLoader → RecursiveCharacterTextSplitter(1000 / overlap 150)
   ▼
src/ingest.py ── embeddings ──▶ PostgreSQL + pgVector (collection por provider)
                                        │
src/chat.py ──▶ src/search.py ──────────┘  similarity_search_with_score(k=10)
                     │  chain LCEL: contexto → prompt do enunciado → LLM
                     ▼
                  RESPOSTA
```

- **`src/providers.py`** — fábrica de IA. Único ponto que conhece Google vs Ollama.
  Expõe `get_embeddings()`, `get_llm()` e `get_collection_name()`. Os demais scripts
  são agnósticos ao provider.
- **`src/ingest.py`** — carrega o PDF, divide em chunks de **1000 chars / overlap 150**
  (requisito do desafio), gera embeddings e grava no pgVector. Recria a collection a
  cada execução (`pre_delete_collection=True`) — idempotente para este dataset.
- **`src/search.py`** — recupera os **k=10** trechos mais relevantes, monta o prompt e
  chama a LLM via uma chain LCEL.
- **`src/chat.py`** — CLI `PERGUNTA:` / `RESPOSTA:`.

## Providers de IA

O provider ativo é escolhido por `PROVIDER` no `.env`. **Cada provider tem sua própria
collection** (`docs_google` / `docs_ollama`) — espaços de embedding são incompatíveis
entre modelos, então trocar de provider exige **reingerir**.

| Provider | Embeddings | LLM | Status |
|---|---|---|---|
| **`google`** (padrão) | `gemini-embedding-001` | `gemini-2.5-flash-lite` | ✅ **validado ponta-a-ponta** |
| `ollama` | `nomic-embed-text` | `llama3.2` / `qwen2.5:3b` | ⚠️ best-effort (dev local) |

> **Nota sobre o Ollama.** O PDF é basicamente uma tabela densa de empresas fictícias, o
> que é difícil para o stack local em dois pontos:
>
> - **Recuperação:** com `nomic-embed-text`, o chunk da `SuperTechIABrazil` fica por volta
>   da 20ª posição para a pergunta do exemplo — fora do `k=10` exigido pelo desafio
>   (mesmo com os prefixos `search_query:`/`search_document:` do modelo, sobe só para ~13ª).
> - **Geração:** LLMs de **3B** em CPU não extraem de forma confiável um valor específico de
>   ~6000 chars de tabela concatenada (needle-in-haystack); o `qwen2.5:7b` tende a acertar,
>   mas leva > 3 min/resposta em CPU puro.
>
> Por isso o **Google é o provider oficial da entrega** e o Ollama fica como opção de dev
> sem custo de API.

## Pré-requisitos

- Python 3.12
- Docker + Docker Compose (para o PostgreSQL + pgVector)
- Uma **API Key da Google** (Gemini) — para o provider padrão.

## Configuração

```bash
# 1. venv LOCAL (dentro da pasta do projeto, nunca global)
python3 -m venv venv
./venv/bin/pip install -r requirements.txt

# 2. .env a partir do template
cp .env.example .env
#    edite .env e preencha GOOGLE_API_KEY
```

Variáveis principais (`.env.example`):

```env
PROVIDER=google                 # google (validado) | ollama (best-effort)
POSTGRES_PORT=2432              # porta do host do Postgres
DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:2432/rag
PG_VECTOR_COLLECTION_NAME=docs  # vira docs_google / docs_ollama
PDF_PATH=document.pdf
GOOGLE_API_KEY=                 # sua chave Gemini
```

> **Porta 2432.** Este projeto usa a porta **2432** para o Postgres (e não 5432) para não
> colidir com outros Postgres locais. O `docker-compose.yml` lê `POSTGRES_PORT` do `.env`
> (o Docker Compose carrega o `.env` da pasta automaticamente); sem `.env`, usa 5432 —
> nesse caso ajuste a porta em `DATABASE_URL`.

## Execução

```bash
# 1. Subir o PostgreSQL + pgVector (lê a porta do .env)
docker compose up -d

# 2. Ingerir o PDF (usa o PROVIDER do .env)
./venv/bin/python src/ingest.py

# 3. Rodar o chat interativo
./venv/bin/python src/chat.py
```

No chat, digite suas perguntas após `PERGUNTA:` e `sair` (ou `exit`/`quit`) para encerrar.

### Usando o Ollama (opcional, sem custo de API)

Requer um Ollama no ar (`OLLAMA_BASE_URL`, padrão `http://localhost:11434`) com os
modelos baixados (`nomic-embed-text` e um LLM):

```bash
ollama pull nomic-embed-text
ollama pull qwen2.5:3b
```

Então troque `PROVIDER=ollama` no `.env` (ou prefixe no comando) e **reingira** — a
collection é por provider:

```bash
PROVIDER=ollama ./venv/bin/python src/ingest.py
PROVIDER=ollama ./venv/bin/python src/chat.py
```

## Verificar o índice

```bash
docker compose exec -T postgres psql -U postgres -d rag -c \
  "SELECT c.name, COUNT(e.*) FROM langchain_pg_collection c \
   LEFT JOIN langchain_pg_embedding e ON e.collection_id = c.uuid GROUP BY c.name;"
```

## Sobre o prompt

O `search.py` usa **na íntegra** o prompt definido no enunciado (`PROMPT_TEMPLATE`): os
k=10 trechos recuperados são concatenados em `{contexto}` e a pergunta vai em
`{pergunta}`. As regras do prompt fazem o modelo recusar ("Não tenho informações
necessárias para responder sua pergunta.") tudo o que não estiver no PDF.

## Estrutura

```
├── docker-compose.yml     # pgvector/pgvector:pg17; porta host via ${POSTGRES_PORT:-5432}
├── requirements.txt       # LangChain + langchain-google-genai + langchain-ollama
├── .env.example           # template multi-provider
├── document.pdf           # PDF de exemplo (empresas fictícias)
├── venv/                  # virtualenv local (não versionado)
├── src/
│   ├── providers.py       # fábrica de embeddings/LLM/collection por provider
│   ├── ingest.py          # ingestão do PDF → pgVector
│   ├── search.py          # busca k=10 + chain LCEL de resposta
│   └── chat.py            # CLI PERGUNTA:/RESPOSTA:
└── README.md
```
