import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_postgres import PGVector

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
COLLECTION_NAME = os.getenv("PG_VECTOR_COLLECTION_NAME", "pdf_chunks")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small").strip("'\"")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-5-nano").strip("'\"")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "").strip()
GOOGLE_EMBEDDING_MODEL = os.getenv("GOOGLE_EMBEDDING_MODEL", "models/text-embedding-004").strip("'\"")
GOOGLE_CHAT_MODEL = os.getenv("GOOGLE_CHAT_MODEL", "gemini-2.5-flash-lite").strip("'\"")

FALLBACK_ANSWER = "Não tenho informações necessárias para responder sua pergunta."


PROMPT_TEMPLATE = """
CONTEXTO:
{contexto}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.
- Responda as perguntas de forma amigável e clara, mas sem adicionar informações não presentes no CONTEXTO.

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

def _build_embeddings():
  if OPENAI_API_KEY:
    return OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL, api_key=OPENAI_API_KEY)

  if GOOGLE_API_KEY:
    return GoogleGenerativeAIEmbeddings(model=GOOGLE_EMBEDDING_MODEL, google_api_key=GOOGLE_API_KEY)

  raise ValueError("Defina OPENAI_API_KEY ou GOOGLE_API_KEY no arquivo .env.")


def _build_llm():
  if OPENAI_API_KEY:
    return ChatOpenAI(model=OPENAI_CHAT_MODEL, api_key=OPENAI_API_KEY)

  if GOOGLE_API_KEY:
    return ChatGoogleGenerativeAI(model=GOOGLE_CHAT_MODEL, google_api_key=GOOGLE_API_KEY)

  raise ValueError("Defina OPENAI_API_KEY ou GOOGLE_API_KEY no arquivo .env.")


def search_prompt(question=None):
  if not question:
    return FALLBACK_ANSWER
  if not DATABASE_URL:
    raise ValueError("A variável DATABASE_URL não foi definida no .env.")

  embeddings = _build_embeddings()
  llm = _build_llm()

  vector_store = PGVector(
    embeddings=embeddings,
    collection_name=COLLECTION_NAME,
    connection=DATABASE_URL,
    use_jsonb=True,
  )

  results = vector_store.similarity_search_with_score(question, k=10)
  if not results:
    return FALLBACK_ANSWER

  context_chunks = []
  for doc, _score in results:
    if doc.page_content and doc.page_content.strip():
      context_chunks.append(doc.page_content.strip())

  if not context_chunks:
    return FALLBACK_ANSWER

  prompt = PROMPT_TEMPLATE.format(contexto="\n\n".join(context_chunks), pergunta=question)
  response = llm.invoke(prompt)

  content = getattr(response, "content", "")
  if isinstance(content, list):
    content = " ".join(str(item) for item in content)
  if not isinstance(content, str):
    content = str(content)

  content = content.strip()
  return content or FALLBACK_ANSWER