# app/api.py  (or api.py if your module root is the same)
import os
import uuid
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from pydantic import BaseModel
from qdrant_client import QdrantClient

from .rag_graph import (
    upsert_pdf_for_document,
    build_generalized_graph,
    ask as run_graph,
    debug_collection_contents
)

# Config
UPLOAD_DIR = Path("./uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Qdrant location & collection must match rag_graph helpers
QDRANT_LOCATION = ":memory:"  # in-memory storage (data lost on restart)
COLLECTION = "use_case_data"
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4.1-nano"

# Global Qdrant client to ensure same instance across requests
qdrant_client = QdrantClient(QDRANT_LOCATION)

app = FastAPI(title="RAG Graph API (2-entry)")


class UploadResponse(BaseModel):
    document_id: str
    chunks: int
    status: str = "ready"

class AskResponse(BaseModel):
    answer: str

@app.on_event("startup")
def _startup():
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY must be set.")

@app.post("/documents", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported")

    # Save uploaded file
    document_id = str(uuid.uuid4())
    dest = UPLOAD_DIR / f"{document_id}.pdf"
    with dest.open("wb") as f:
        f.write(await file.read())

    # Upsert into vector store with metadata {document_id}
    try:
        chunks = upsert_pdf_for_document(
            document_id=document_id,
            pdf_path=str(dest),
            client=qdrant_client,
            qdrant_location=QDRANT_LOCATION,
            collection_name=COLLECTION,
            openai_embedding_model=EMBED_MODEL,
            chunk_size=800,
            chunk_overlap=120,
        )
    except Exception as e:
        raise HTTPException(500, f"Failed to index PDF: {e}")

    return UploadResponse(document_id=document_id, chunks=chunks)

@app.get("/debug/collection")
def debug_collection():
    """Debug endpoint to check collection contents"""
    return debug_collection_contents(qdrant_client, COLLECTION)


@app.post("/ask-simple", response_model=AskResponse)
def ask_simple(question: str = Form(...), k: int = Form(5)):
    """Simple endpoint to ask questions across all documents"""
    graph = build_generalized_graph(
        k=k,
        openai_embedding_model=EMBED_MODEL,
        openai_chat_model=CHAT_MODEL,
        client=qdrant_client,
        qdrant_location=QDRANT_LOCATION,
        collection_name=COLLECTION,
    )
    answer = run_graph(graph, question)
    return AskResponse(answer=answer)
