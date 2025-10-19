# rag_graph.py
from __future__ import annotations

import os
import time
from typing import List, TypedDict, Optional, Dict, Any, Callable
from enum import Enum
from abc import ABC, abstractmethod

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

# ---------- Core RAG Components ----------
class RAGComponents:
    """Shared components for all RAG implementations"""
    
    def __init__(
        self,
        k: int = 5,
        openai_embedding_model: str = "text-embedding-3-small",
        openai_chat_model: str = "gpt-4.1-nano",
        client: Optional[QdrantClient] = None,
        qdrant_location: str = ":memory:",
        collection_name: str = "use_case_data",
    ):
        self.k = k
        self.collection_name = collection_name
        
        # Initialize client and collection
        if client is None:
            client = QdrantClient(qdrant_location)
        self.client = client
        _ensure_collection(client, collection_name)
        
        # Initialize embeddings and models
        self.embeddings = OpenAIEmbeddings(model=openai_embedding_model)
        self.llm = ChatOpenAI(model=openai_chat_model)
        self.rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT)
        
        # Initialize vector store
        self.store = QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=self.embeddings,
        )

    def generate_response(self, state: EnhancedState) -> Dict[str, Any]:
        """Shared generation logic with performance tracking"""
        generate_start = time.time()
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        
        # Estimate input tokens
        prompt_text = self.rag_prompt.format(question=state["question"], context=docs_content)
        input_tokens = estimate_tokens(prompt_text)
        
        messages = self.rag_prompt.format_messages(
            question=state["question"],
            context=docs_content
        )
        response = self.llm.invoke(messages)
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

# ---------- RAG Strategy Implementations ----------
class RAGStrategy(ABC):
    """Abstract base class for RAG strategies"""
    
    def __init__(self, components: RAGComponents):
        self.components = components
    
    @abstractmethod
    def build_retriever(self):
        """Build the specific retriever for this strategy"""
        pass
    
    @abstractmethod
    def get_graph_nodes(self) -> List[Callable]:
        """Return the list of nodes for the graph"""
        pass
    
    @property
    @abstractmethod
    def graph_type_name(self) -> str:
        """Return the name of this graph type"""
        pass
    
    @property
    @abstractmethod
    def retrieval_count(self) -> int:
        """Return the number of retrieval operations"""
        pass

class SimpleRAGStrategy(RAGStrategy):
    """Simple semantic similarity retrieval"""
    
    def build_retriever(self):
        return self.components.store.as_retriever(search_kwargs={"k": self.components.k})
    
    @property
    def graph_type_name(self) -> str:
        return "simple"
    
    @property
    def retrieval_count(self) -> int:
        return 1
    
    def get_graph_nodes(self) -> List[Callable]:
        retriever = self.build_retriever()
        
        def retrieve(state: EnhancedState):
            retrieve_start = time.time()
            context = retriever.invoke(state["question"])
            retrieve_time = time.time() - retrieve_start
            
            return {
                "context": context,
                "retrieve_time": retrieve_time,
                "retrieved_documents": len(context),
                "graph_type": self.graph_type_name,
                "retrieval_count": self.retrieval_count
            }
        
        return [retrieve, self.components.generate_response]

class MultiQueryRAGStrategy(RAGStrategy):
    """Multi-query generation for better retrieval"""
    
    def build_retriever(self):
        base_retriever = self.components.store.as_retriever(search_kwargs={"k": self.components.k})
        return MultiQueryRetriever.from_llm(retriever=base_retriever, llm=self.components.llm)
    
    @property
    def graph_type_name(self) -> str:
        return "multi_query"
    
    @property
    def retrieval_count(self) -> int:
        return 3  # Multi-query typically generates 3 queries
    
    def get_graph_nodes(self) -> List[Callable]:
        retriever = self.build_retriever()
        
        def retrieve(state: EnhancedState):
            retrieve_start = time.time()
            context = retriever.invoke(state["question"])
            retrieve_time = time.time() - retrieve_start
            
            return {
                "context": context,
                "retrieve_time": retrieve_time,
                "retrieved_documents": len(context),
                "graph_type": self.graph_type_name,
                "retrieval_count": self.retrieval_count
            }
        
        return [retrieve, self.components.generate_response]

class EnsembleRAGStrategy(RAGStrategy):
    """Ensemble of semantic and BM25 retrieval"""
    
    def build_retriever(self):
        semantic_retriever = self.components.store.as_retriever(search_kwargs={"k": self.components.k})
        
        # Get documents for BM25
        all_docs = []
        try:
            search_results = self.components.store.similarity_search("", k=1000)
            all_docs = search_results
        except:
            pass
        
        if all_docs:
            bm25_retriever = BM25Retriever.from_documents(all_docs)
            bm25_retriever.k = self.components.k
            
            return EnsembleRetriever(
                retrievers=[semantic_retriever, bm25_retriever],
                weights=[0.6, 0.4]  # Favor semantic search slightly
            )
        else:
            return semantic_retriever
    
    @property
    def graph_type_name(self) -> str:
        return "ensemble"
    
    @property
    def retrieval_count(self) -> int:
        return 2  # Semantic + BM25
    
    def get_graph_nodes(self) -> List[Callable]:
        retriever = self.build_retriever()
        
        def retrieve(state: EnhancedState):
            retrieve_start = time.time()
            context = retriever.invoke(state["question"])
            retrieve_time = time.time() - retrieve_start
            
            return {
                "context": context,
                "retrieve_time": retrieve_time,
                "retrieved_documents": len(context),
                "graph_type": self.graph_type_name,
                "retrieval_count": self.retrieval_count
            }
        
        return [retrieve, self.components.generate_response]

class HybridRAGStrategy(RAGStrategy):
    """Semantic retrieval with keyword-based reranking"""
    
    def build_retriever(self):
        return self.components.store.as_retriever(search_kwargs={"k": self.components.k * 2})
    
    @property
    def graph_type_name(self) -> str:
        return "hybrid"
    
    @property
    def retrieval_count(self) -> int:
        return 1
    
    def get_graph_nodes(self) -> List[Callable]:
        retriever = self.build_retriever()
        
        def retrieve(state: EnhancedState):
            retrieve_start = time.time()
            context = retriever.invoke(state["question"])
            retrieve_time = time.time() - retrieve_start
            
            return {
                "context": context,
                "retrieve_time": retrieve_time,
                "retrieved_documents": len(context),
                "graph_type": self.graph_type_name,
                "retrieval_count": self.retrieval_count
            }
        
        def rerank(state: EnhancedState):
            """Simple reranking based on keyword overlap"""
            rerank_start = time.time()
            
            question_words = set(state["question"].lower().split())
            scored_docs = []
            
            for doc in state["context"]:
                doc_words = set(doc.page_content.lower().split())
                overlap_score = len(question_words.intersection(doc_words)) / max(len(question_words), 1)
                scored_docs.append((doc, overlap_score))
            
            # Sort by score and take top k
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            reranked_context = [doc for doc, score in scored_docs[:self.components.k]]
            
            rerank_time = time.time() - rerank_start
            
            return {
                "context": reranked_context,
                "retrieve_time": state["retrieve_time"] + rerank_time,
                "retrieved_documents": len(reranked_context)
            }
        
        return [retrieve, rerank, self.components.generate_response]

# ---------- Strategy Factory ----------
def get_rag_strategy(graph_type: GraphType, components: RAGComponents) -> RAGStrategy:
    """Factory function to get the appropriate RAG strategy"""
    strategies = {
        GraphType.SIMPLE: SimpleRAGStrategy,
        GraphType.MULTI_QUERY: MultiQueryRAGStrategy,
        GraphType.ENSEMBLE: EnsembleRAGStrategy,
        GraphType.HYBRID: HybridRAGStrategy,
    }
    
    strategy_class = strategies.get(graph_type)
    if not strategy_class:
        raise ValueError(f"Unknown graph type: {graph_type}")
    
    return strategy_class(components)

# ---------- Unified Graph Builder ----------
def build_rag_graph(
    graph_type: GraphType,
    *,
    k: int = 5,
    openai_embedding_model: str = "text-embedding-3-small",
    openai_chat_model: str = "gpt-4.1-nano",
    client: Optional[QdrantClient] = None,
    qdrant_location: str = ":memory:",
    collection_name: str = "use_case_data",
) -> StateGraph:
    """
    Unified function to build any RAG graph type using the strategy pattern.
    """
    # Create shared components
    components = RAGComponents(
        k=k,
        openai_embedding_model=openai_embedding_model,
        openai_chat_model=openai_chat_model,
        client=client,
        qdrant_location=qdrant_location,
        collection_name=collection_name,
    )
    
    # Get the strategy and build the graph
    strategy = get_rag_strategy(graph_type, components)
    nodes = strategy.get_graph_nodes()
    
    # Build the graph
    graph_builder = StateGraph(EnhancedState).add_sequence(nodes)
    graph_builder.add_edge(START, nodes[0].__name__)
    
    return graph_builder.compile()

# ---------- Factory Function ----------
def build_graph_by_type(graph_type: GraphType, **kwargs) -> StateGraph:
    """Factory function to build different graph types using the unified builder"""
    return build_rag_graph(graph_type, **kwargs)

# ---------- Utility Functions ----------
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

# ---------- CLI smoke test ----------
if __name__ == "__main__":
    assert os.getenv("OPENAI_API_KEY"), "Set OPENAI_API_KEY env var."
    # Example: upsert a single file and query it
    # upsert_pdf_for_document(document_id="doc1", pdf_path="./data/sample.pdf")
    # g = build_rag_graph(GraphType.SIMPLE)
    # print(ask(g, "What is this PDF about?"))
