# rag_graph.py
from __future__ import annotations

import os
import time
from typing import List, TypedDict, Optional, Dict, Any
from enum import Enum

from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers.ensemble import EnsembleRetriever

from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance, VectorParams
)

from langchain_core.documents import Document
from langchain.prompts import ChatPromptTemplate
from langgraph.graph import START, StateGraph

# ---------- Public types ----------
class GraphType(Enum):
    SIMPLE = "simple"
    MULTI_QUERY = "multi_query"
    ENSEMBLE = "ensemble"
    HYBRID = "hybrid"

class State(TypedDict):
    question: str
    context: List[Document]
    response: str

class EnhancedState(TypedDict):
    question: str
    context: List[Document]
    response: str
    # Performance metrics
    start_time: float
    retrieve_time: Optional[float]
    generate_time: Optional[float]
    total_time: Optional[float]
    # Token tracking (for observability)
    input_tokens: int
    output_tokens: int
    retrieval_count: int
    # Graph metadata
    graph_type: str
    retrieved_documents: int

# ---------- Defaults & prompt ----------
RAG_PROMPT = """You are a helpful assistant who answers questions 
based on provided context only.
Do not use outside knowledge.

### Question
{question}

### Context
{context}
"""

# ---------- Performance tracking utilities ----------
def estimate_tokens(text: str) -> int:
    """Rough token estimation (1 token ≈ 4 characters for OpenAI models)"""
    return len(text) // 4

def ask(graph: StateGraph, question: str) -> str:
    out = graph.invoke({"question": question})
    return out["response"]

def ask_with_metrics(graph: StateGraph, question: str) -> Dict[str, Any]:
    """Enhanced ask function that returns both response and performance metrics"""
    start_time = time.time()
    out = graph.invoke({"question": question, "start_time": start_time})
    
    # Calculate total time if not already set
    if "total_time" not in out or out["total_time"] is None:
        out["total_time"] = time.time() - start_time
    
    return {
        "response": out["response"],
        "metrics": {
            "total_time": out.get("total_time", 0),
            "retrieve_time": out.get("retrieve_time", 0),
            "generate_time": out.get("generate_time", 0),
            "input_tokens": out.get("input_tokens", 0),
            "output_tokens": out.get("output_tokens", 0),
            "retrieved_documents": out.get("retrieved_documents", 0),
            "graph_type": out.get("graph_type", "unknown")
        }
    }

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
    Build a simple LangGraph that searches across ALL documents in 
    the collection (no document filtering) with performance tracking.
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

    def retrieve(state: EnhancedState):
        retrieve_start = time.time()
        context = retriever.invoke(state["question"])
        retrieve_time = time.time() - retrieve_start
        
        return {
            "context": context,
            "retrieve_time": retrieve_time,
            "retrieved_documents": len(context),
            "graph_type": "simple",
            "retrieval_count": 1
        }

    def generate(state: EnhancedState):
        generate_start = time.time()
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        
        # Estimate input tokens
        prompt_text = rag_prompt.format(question=state["question"], context=docs_content)
        input_tokens = estimate_tokens(prompt_text)
        
        messages = rag_prompt.format_messages(
            question=state["question"],
            context=docs_content
        )
        response = llm.invoke(messages)
        generate_time = time.time() - generate_start
        
        # Estimate output tokens
        output_tokens = estimate_tokens(response.content)
        
        # Calculate total time
        total_time = time.time() - state.get("start_time", generate_start)
        
        return {
            "response": response.content,
            "generate_time": generate_time,
            "total_time": total_time,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens
        }

    graph_builder = StateGraph(EnhancedState).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    return graph_builder.compile()


def build_multi_query_graph(
    *,
    k: int = 5,
    openai_embedding_model: str = "text-embedding-3-small",
    openai_chat_model: str = "gpt-4.1-nano",
    client: Optional[QdrantClient] = None,
    qdrant_location: str = ":memory:",
    collection_name: str = "use_case_data",
) -> StateGraph:
    """
    Build a Multi-Query RAG graph that generates multiple queries for better retrieval.
    """
    if client is None:
        client = QdrantClient(qdrant_location)
    _ensure_collection(client, collection_name)
    embeddings = OpenAIEmbeddings(model=openai_embedding_model)
    store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
    )
    base_retriever = store.as_retriever(search_kwargs={"k": k})
    llm = ChatOpenAI(model=openai_chat_model)
    
    # Create multi-query retriever
    multi_query_retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever, 
        llm=llm
    )

    rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT)

    def retrieve(state: EnhancedState):
        retrieve_start = time.time()
        context = multi_query_retriever.invoke(state["question"])
        retrieve_time = time.time() - retrieve_start
        
        return {
            "context": context,
            "retrieve_time": retrieve_time,
            "retrieved_documents": len(context),
            "graph_type": "multi_query",
            "retrieval_count": 3  # Multi-query typically generates 3 queries
        }

    def generate(state: EnhancedState):
        generate_start = time.time()
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        
        prompt_text = rag_prompt.format(question=state["question"], context=docs_content)
        input_tokens = estimate_tokens(prompt_text)
        
        messages = rag_prompt.format_messages(
            question=state["question"],
            context=docs_content
        )
        response = llm.invoke(messages)
        generate_time = time.time() - generate_start
        
        output_tokens = estimate_tokens(response.content)
        total_time = time.time() - state.get("start_time", generate_start)
        
        return {
            "response": response.content,
            "generate_time": generate_time,
            "total_time": total_time,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens
        }

    graph_builder = StateGraph(EnhancedState).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    return graph_builder.compile()


def build_ensemble_graph(
    *,
    k: int = 5,
    openai_embedding_model: str = "text-embedding-3-small",
    openai_chat_model: str = "gpt-4.1-nano",
    client: Optional[QdrantClient] = None,
    qdrant_location: str = ":memory:",
    collection_name: str = "use_case_data",
) -> StateGraph:
    """
    Build an Ensemble RAG graph that combines multiple retrieval strategies.
    """
    if client is None:
        client = QdrantClient(qdrant_location)
    _ensure_collection(client, collection_name)
    embeddings = OpenAIEmbeddings(model=openai_embedding_model)
    store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
    )
    
    # Get documents for BM25
    # Note: In production, you'd want to cache this
    all_docs = []
    try:
        # This is a simplified approach - in production you'd store documents separately
        search_results = store.similarity_search("", k=1000)  # Get many docs for BM25
        all_docs = search_results
    except:
        # Fallback if no documents exist yet
        pass
    
    semantic_retriever = store.as_retriever(search_kwargs={"k": k})
    
    if all_docs:
        bm25_retriever = BM25Retriever.from_documents(all_docs)
        bm25_retriever.k = k
        
        # Combine retrievers
        ensemble_retriever = EnsembleRetriever(
            retrievers=[semantic_retriever, bm25_retriever],
            weights=[0.6, 0.4]  # Favor semantic search slightly
        )
    else:
        ensemble_retriever = semantic_retriever

    llm = ChatOpenAI(model=openai_chat_model)
    rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT)

    def retrieve(state: EnhancedState):
        retrieve_start = time.time()
        context = ensemble_retriever.invoke(state["question"])
        retrieve_time = time.time() - retrieve_start
        
        return {
            "context": context,
            "retrieve_time": retrieve_time,
            "retrieved_documents": len(context),
            "graph_type": "ensemble",
            "retrieval_count": 2 if all_docs else 1  # Semantic + BM25 or just semantic
        }

    def generate(state: EnhancedState):
        generate_start = time.time()
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        
        prompt_text = rag_prompt.format(question=state["question"], context=docs_content)
        input_tokens = estimate_tokens(prompt_text)
        
        messages = rag_prompt.format_messages(
            question=state["question"],
            context=docs_content
        )
        response = llm.invoke(messages)
        generate_time = time.time() - generate_start
        
        output_tokens = estimate_tokens(response.content)
        total_time = time.time() - state.get("start_time", generate_start)
        
        return {
            "response": response.content,
            "generate_time": generate_time,
            "total_time": total_time,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens
        }

    graph_builder = StateGraph(EnhancedState).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    return graph_builder.compile()


def build_hybrid_graph(
    *,
    k: int = 5,
    openai_embedding_model: str = "text-embedding-3-small",
    openai_chat_model: str = "gpt-4.1-nano",
    client: Optional[QdrantClient] = None,
    qdrant_location: str = ":memory:",
    collection_name: str = "use_case_data",
) -> StateGraph:
    """
    Build a Hybrid RAG graph with conditional retrieval and reranking.
    """
    if client is None:
        client = QdrantClient(qdrant_location)
    _ensure_collection(client, collection_name)
    embeddings = OpenAIEmbeddings(model=openai_embedding_model)
    store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
    )
    retriever = store.as_retriever(search_kwargs={"k": k * 2})  # Get more docs for reranking
    llm = ChatOpenAI(model=openai_chat_model)
    rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT)

    def retrieve(state: EnhancedState):
        retrieve_start = time.time()
        context = retriever.invoke(state["question"])
        retrieve_time = time.time() - retrieve_start
        
        return {
            "context": context,
            "retrieve_time": retrieve_time,
            "retrieved_documents": len(context),
            "graph_type": "hybrid",
            "retrieval_count": 1
        }

    def rerank(state: EnhancedState):
        """Simple reranking based on keyword overlap (in production, use a reranking model)"""
        rerank_start = time.time()
        
        question_words = set(state["question"].lower().split())
        scored_docs = []
        
        for doc in state["context"]:
            doc_words = set(doc.page_content.lower().split())
            overlap_score = len(question_words.intersection(doc_words)) / max(len(question_words), 1)
            scored_docs.append((doc, overlap_score))
        
        # Sort by score and take top k
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        reranked_context = [doc for doc, score in scored_docs[:k]]
        
        rerank_time = time.time() - rerank_start
        
        return {
            "context": reranked_context,
            "retrieve_time": state["retrieve_time"] + rerank_time,
            "retrieved_documents": len(reranked_context)
        }

    def generate(state: EnhancedState):
        generate_start = time.time()
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        
        prompt_text = rag_prompt.format(question=state["question"], context=docs_content)
        input_tokens = estimate_tokens(prompt_text)
        
        messages = rag_prompt.format_messages(
            question=state["question"],
            context=docs_content
        )
        response = llm.invoke(messages)
        generate_time = time.time() - generate_start
        
        output_tokens = estimate_tokens(response.content)
        total_time = time.time() - state.get("start_time", generate_start)
        
        return {
            "response": response.content,
            "generate_time": generate_time,
            "total_time": total_time,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens
        }

    graph_builder = StateGraph(EnhancedState).add_sequence([retrieve, rerank, generate])
    graph_builder.add_edge(START, "retrieve")
    return graph_builder.compile()


def build_graph_by_type(
    graph_type: GraphType,
    **kwargs
) -> StateGraph:
    """
    Factory function to build different graph types.
    """
    if graph_type == GraphType.SIMPLE:
        return build_generalized_graph(**kwargs)
    elif graph_type == GraphType.MULTI_QUERY:
        return build_multi_query_graph(**kwargs)
    elif graph_type == GraphType.ENSEMBLE:
        return build_ensemble_graph(**kwargs)
    elif graph_type == GraphType.HYBRID:
        return build_hybrid_graph(**kwargs)
    else:
        raise ValueError(f"Unknown graph type: {graph_type}")


# ---------- CLI smoke test ----------
if __name__ == "__main__":
    assert os.getenv("OPENAI_API_KEY"), "Set OPENAI_API_KEY env var."
    # Example: upsert a single file and query it
    # upsert_pdf_for_document(document_id="doc1", pdf_path="./data/sample.pdf")
    # g = build_generalized_graph()
    # print(ask(g, "What is this PDF about?"))
