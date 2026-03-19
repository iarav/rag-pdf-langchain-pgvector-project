import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

PDF_PATH = os.getenv("PDF_PATH")
DATABASE_URL = os.getenv("DATABASE_URL")
COLLECTION_NAME = os.getenv("PG_VECTOR_COLLECTION_NAME", "pdf_chunks")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small").strip("'\"")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "").strip()
GOOGLE_EMBEDDING_MODEL = os.getenv("GOOGLE_EMBEDDING_MODEL", "models/text-embedding-004").strip("'\"")


def build_embeddings():
    if OPENAI_API_KEY:
        return OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL, api_key=OPENAI_API_KEY)

    if GOOGLE_API_KEY:
        return GoogleGenerativeAIEmbeddings(model=GOOGLE_EMBEDDING_MODEL, google_api_key=GOOGLE_API_KEY)

    raise ValueError("Defina OPENAI_API_KEY ou GOOGLE_API_KEY no arquivo .env.")

def ingest_pdf():
    if not PDF_PATH:
        raise ValueError("A variável PDF_PATH não foi definida no .env.")
    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(f"Arquivo PDF não encontrado em: {PDF_PATH}")
    if not DATABASE_URL:
        raise ValueError("A variável DATABASE_URL não foi definida no .env.")

    print("Carregando PDF...")
    pages = PyPDFLoader(PDF_PATH).load()
    if not pages:
        raise ValueError("Nenhuma página foi carregada do PDF.")

    print("Dividindo documento em chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(pages)
    if not chunks:
        raise ValueError("A divisão em chunks retornou vazio.")

    print("Gerando embeddings e salvando no pgVector...")
    embeddings = build_embeddings()
    vector_store = PGVector(
        embeddings=embeddings,
        collection_name=COLLECTION_NAME,
        connection=DATABASE_URL,
        use_jsonb=True,
    )

    vector_store.add_documents(chunks)
    print(f"Ingestão concluída com sucesso. Chunks salvos: {len(chunks)}")


if __name__ == "__main__":
    ingest_pdf()