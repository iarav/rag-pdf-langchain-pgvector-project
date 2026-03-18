# PDF RAG with LangChain and PostgreSQL (pgVector)

## 1) Project Context

This project was developed for an MBA practical assignment focused on building a Retrieval-Augmented Generation (RAG) pipeline.

The goal is to ingest the contents of a PDF into a vector database and answer user questions in a command-line chat, using only the information found in that document.

The system combines:

- Semantic indexing (embeddings + vector storage)
- Similarity-based retrieval
- LLM response generation constrained by explicit prompt rules

## 2) Objective

The application must provide two core capabilities:

1. Ingestion
	 Read a PDF, split it into chunks, generate embeddings, and persist vectors in PostgreSQL with pgVector.

2. Semantic search + CLI chat
	 Receive a question in the terminal, retrieve the most relevant chunks from the vector store, build a context-aware prompt, and generate an answer based only on retrieved context.

If the requested information is not explicitly present in context, the answer must be:

Não tenho informações necessárias para responder sua pergunta.

## 3) Tech Stack

- Python
- LangChain ecosystem
- PostgreSQL + pgVector extension
- Docker and Docker Compose

Recommended models from the assignment:

- OpenAI embeddings: text-embedding-3-small
- OpenAI chat model: gpt-5-nano
- Gemini embeddings: models/embedding-001
- Gemini chat model: gemini-2.5-flash-lite

## 4) Repository Structure

Current workspace structure:

```text
.
├── docker-compose.yml
├── requirements.txt
├── .env
├── document.pdf
├── README.md
└── src/
		├── chat.py
		├── injest.py
		└── search.py
```

## 5) How the Solution Works

### 5.1 Ingestion flow

1. Load PDF pages with PyPDFLoader.
2. Split document into chunks using RecursiveCharacterTextSplitter.
	 - chunk_size = 1000
	 - chunk_overlap = 150
3. Generate embeddings for each chunk.
4. Store chunks + vectors in PostgreSQL (pgVector) using LangChain PGVector integration.

### 5.2 Search and answer flow

1. User asks a question in CLI.
2. The question is embedded.
3. The system retrieves the top 10 most similar chunks (k=10).
4. Retrieved chunks are concatenated into CONTEXTO.
5. The LLM is called with strict rules to avoid hallucination.
6. The answer is printed in CLI.

## 6) Prompt Policy (Grounded Answers Only)

The chat prompt enforces the following behavior:

- Answer only from CONTEXTO.
- If the answer is not explicit in CONTEXTO, return:
	Não tenho informações necessárias para responder sua pergunta.
- Do not invent facts.
- Do not use external knowledge.
- Do not provide unsupported opinions or interpretations.

This policy is essential to guarantee predictable and auditable answers in academic and business scenarios.

## 7) Prerequisites

Install the following tools:

- Python 3.10+
- Docker
- Docker Compose

## 8) Environment Configuration

Create and fill your environment variables in .env using .env.example

## 9) Installation

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv venv

# Linux/macOS
source venv/bin/activate

# Windows PowerShell
venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

## 10) Running the Project

### Step 1: Start PostgreSQL with pgVector

```bash
docker compose up -d
```

This project includes a bootstrap service that creates the vector extension automatically.

### Step 2: Ingest the PDF

```bash
python src/ingest.py
```

### Step 3: Start CLI chat

```bash
python src/chat.py
```

## 11) Example Interaction

```text
PERGUNTA: Qual o faturamento da Empresa SuperTechIABrazil?
RESPOSTA: O faturamento foi de 10 milhões de reais.
```

Out-of-context question:

```text
PERGUNTA: Quantos clientes temos em 2024?
RESPOSTA: Não tenho informações necessárias para responder sua pergunta.
```
