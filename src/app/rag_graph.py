# rag_graph.py
from __future__ import annotations

import os
from typing import List, TypedDict, Optional

from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance, VectorParams
)

from langchain_core.documents import Document
from langchain.prompts import ChatPromptTemplate
from langgraph.graph import START, StateGraph

# ---------- Public types ----------
class State(TypedDict):
    question: str
    context: List[Document]
    response: str

# ---------- Defaults & prompt ----------
RAG_PROMPT = """You are a helpful assistant who answers questions 
based on provided context only.
Do not use outside knowledge.

### Question
{question}

### Context
{context}
"""

def ask(graph: StateGraph, question: str) -> str:
    out = graph.invoke({"question": question})
    return out["response"]

def _ensure_collection(client: QdrantClient, collection_name: str):
    try:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        )
    except Exception:
        pass

def upsert_pdf_for_document(
    *,
    document_id: str,
    pdf_path: str,
    client: Optional[QdrantClient] = None,
    qdrant_location: str = ":memory:",
    collection_name: str = "use_case_data",
    openai_embedding_model: str = "text-embedding-3-small",
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> int:
    """
    Load a single PDF, split, embed, and upsert into Qdrant with 
    metadata {document_id}.
    Returns number of chunks added.
    """
    # 1) Load one PDF
    docs = PyMuPDFLoader(pdf_path).load()
    # If scanned PDF, page_content may be empty. Keep simple here as requested.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunks = [d for d in splitter.split_documents(docs) if d.page_content.strip()]

    # tag with document_id so we can filter later
    for i, d in enumerate(chunks):
        md = dict(d.metadata or {})
        md["document_id"] = document_id
        md.setdefault("chunk_id", i)
        d.metadata = md

    # 2) Embed + upsert
    if client is None:
        client = QdrantClient(qdrant_location)
    _ensure_collection(client, collection_name)
    embeddings = OpenAIEmbeddings(model=openai_embedding_model)
    store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
    )
    store.add_documents(chunks)
    return len(chunks)


def debug_collection_contents(
    client: QdrantClient,
    collection_name: str = "use_case_data",
    limit: int = 10
) -> dict:
    """Debug function to check what's actually in the collection"""
    try:
        # Get collection info
        collection_info = client.get_collection(collection_name)
        
        # Get some points to see the structure
        points = client.scroll(
            collection_name=collection_name,
            limit=limit,
            with_payload=True,
            with_vectors=False
        )[0]
        
        # Get unique document IDs
        document_ids = set()
        for point in points:
            if "document_id" in point.payload.get("metadata", {}):
                document_ids.add(point.payload["metadata"]["document_id"])
        
        return {
            "collection_exists": True,
            "points_count": collection_info.points_count,
            "unique_document_ids": list(document_ids),
            "sample_points": [
                {
                    "id": point.id,
                    "payload": point.payload
                } for point in points[:3]
            ]
        }
    except Exception as e:
        return {
            "collection_exists": False,
            "error": str(e)
        }

def build_generalized_graph(
    *,
    k: int = 5,
    openai_embedding_model: str = "text-embedding-3-small",
    openai_chat_model: str = "gpt-4.1-nano",
    client: Optional[QdrantClient] = None,
    qdrant_location: str = ":memory:",
    collection_name: str = "use_case_data",
) -> StateGraph:
    """
    Build a LangGraph that searches across ALL documents in 
    the collection (no document filtering).
    """
    # build a generalized retriever (no document filter)
    if client is None:
        client = QdrantClient(qdrant_location)
    _ensure_collection(client, collection_name)
    embeddings = OpenAIEmbeddings(model=openai_embedding_model)
    store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
    )
    retriever = store.as_retriever(search_kwargs={"k": k})

    rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT)
    llm = ChatOpenAI(model=openai_chat_model)

    def retrieve(state: State):
        context = retriever.invoke(state["question"])
        
        return {"context": context}

    def generate(state: State):
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        messages = rag_prompt.format_messages(
            question=state["question"],
            context=docs_content
        )
        response = llm.invoke(messages)
        return {"response": response.content}

    graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    return graph_builder.compile()



# ---------- CLI smoke test ----------
if __name__ == "__main__":
    assert os.getenv("OPENAI_API_KEY"), "Set OPENAI_API_KEY env var."
    # Example: upsert a single file and query it
    # upsert_pdf_for_document(document_id="doc1", pdf_path="./data/sample.pdf")
    # g = build_generalized_graph()
    # print(ask(g, "What is this PDF about?"))
