# app/api.py  (or api.py if your module root is the same)
import os
import uuid
from pathlib import Path
from typing import Dict, Any

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from pydantic import BaseModel
from qdrant_client import QdrantClient

from .rag_graph import (
    upsert_pdf_for_document,
    build_generalized_graph,
    build_graph_by_type,
    ask as run_graph,
    ask_with_metrics,
    debug_collection_contents,
    GraphType
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

app = FastAPI(title="RAG Graph API with Performance Metrics")


class UploadResponse(BaseModel):
    document_id: str
    chunks: int
    status: str = "ready"

class AskResponse(BaseModel):
    answer: str

class PerformanceMetrics(BaseModel):
    total_time: float
    retrieve_time: float
    generate_time: float
    input_tokens: int
    output_tokens: int
    retrieved_documents: int
    graph_type: str

class EnhancedAskResponse(BaseModel):
    answer: str
    metrics: PerformanceMetrics

class GraphComparisonResponse(BaseModel):
    question: str
    results: Dict[str, Dict[str, Any]]  # graph_type -> {answer, metrics}

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


@app.post("/ask-with-metrics", response_model=EnhancedAskResponse)
def ask_with_performance_metrics(
    question: str = Form(...), 
    graph_type: str = Form("simple"),
    k: int = Form(5)
):
    """Ask questions with detailed performance metrics"""
    try:
        # Convert string to GraphType enum
        graph_type_enum = GraphType(graph_type)
    except ValueError:
        raise HTTPException(400, f"Invalid graph type. Must be one of: {[t.value for t in GraphType]}")
    
    graph = build_graph_by_type(
        graph_type=graph_type_enum,
        k=k,
        openai_embedding_model=EMBED_MODEL,
        openai_chat_model=CHAT_MODEL,
        client=qdrant_client,
        qdrant_location=QDRANT_LOCATION,
        collection_name=COLLECTION,
    )
    
    result = ask_with_metrics(graph, question)
    
    return EnhancedAskResponse(
        answer=result["response"],
        metrics=PerformanceMetrics(**result["metrics"])
    )


@app.post("/compare-graphs", response_model=GraphComparisonResponse)
def compare_graph_types(
    question: str = Form(...),
    k: int = Form(5),
    graph_types: str = Form("simple,multi_query,ensemble,hybrid")  # Comma-separated list
):
    """Compare different graph types for the same question"""
    
    # Parse graph types
    requested_types = [t.strip() for t in graph_types.split(",")]
    results = {}
    
    for graph_type_str in requested_types:
        try:
            graph_type_enum = GraphType(graph_type_str)
            
            graph = build_graph_by_type(
                graph_type=graph_type_enum,
                k=k,
                openai_embedding_model=EMBED_MODEL,
                openai_chat_model=CHAT_MODEL,
                client=qdrant_client,
                qdrant_location=QDRANT_LOCATION,
                collection_name=COLLECTION,
            )
            
            result = ask_with_metrics(graph, question)
            results[graph_type_str] = {
                "answer": result["response"],
                "metrics": result["metrics"]
            }
            
        except ValueError:
            results[graph_type_str] = {
                "error": f"Invalid graph type: {graph_type_str}"
            }
        except Exception as e:
            results[graph_type_str] = {
                "error": f"Error processing {graph_type_str}: {str(e)}"
            }
    
    return GraphComparisonResponse(
        question=question,
        results=results
    )


@app.get("/graph-types")
def get_available_graph_types():
    """Get list of available graph types and their descriptions"""
    return {
        "available_types": [
            {
                "type": GraphType.SIMPLE.value,
                "description": "Basic semantic similarity retrieval with single query"
            },
            {
                "type": GraphType.MULTI_QUERY.value,
                "description": "Generates multiple reformulated queries for better retrieval"
            },
            {
                "type": GraphType.ENSEMBLE.value,
                "description": "Combines semantic and BM25 keyword-based retrieval"
            },
            {
                "type": GraphType.HYBRID.value,
                "description": "Semantic retrieval with keyword-based reranking"
            }
        ]
    }


@app.get("/performance-comparison")
def get_performance_comparison(
    question: str = "What are the main topics discussed?",
    k: int = 5
):
    """
    Get a performance comparison across all graph types for analysis.
    Useful for understanding trade-offs between different approaches.
    """
    results = {}
    
    for graph_type in GraphType:
        try:
            graph = build_graph_by_type(
                graph_type=graph_type,
                k=k,
                openai_embedding_model=EMBED_MODEL,
                openai_chat_model=CHAT_MODEL,
                client=qdrant_client,
                qdrant_location=QDRANT_LOCATION,
                collection_name=COLLECTION,
            )
            
            result = ask_with_metrics(graph, question)
            results[graph_type.value] = result["metrics"]
            
        except Exception as e:
            results[graph_type.value] = {"error": str(e)}
    
    # Add summary statistics
    successful_results = {k: v for k, v in results.items() if "error" not in v}
    if successful_results:
        summary = {
            "fastest_total_time": min(successful_results.values(), key=lambda x: x["total_time"]),
            "slowest_total_time": max(successful_results.values(), key=lambda x: x["total_time"]),
            "most_documents": max(successful_results.values(), key=lambda x: x["retrieved_documents"]),
            "lowest_tokens": min(successful_results.values(), key=lambda x: x["input_tokens"] + x["output_tokens"]),
            "highest_tokens": max(successful_results.values(), key=lambda x: x["input_tokens"] + x["output_tokens"]),
        }
        results["_summary"] = summary
    
    return {
        "question": question,
        "k": k,
        "results": results
    }
