
"""
Shared test configuration and fixtures for the RAG system test suite.

This module provides:
1. Mock fixtures for external dependencies
2. Test data factories
3. Shared test utilities
"""

import os
import pytest
from unittest.mock import Mock, MagicMock, patch

# Set test environment variables before any imports
os.environ["OPENAI_API_KEY"] = "test-key-for-testing"


# Global mock fixtures that apply to all tests
@pytest.fixture(scope="session", autouse=True)
def mock_openai_globally():
    """Mock OpenAI components globally for all tests"""
    
    # Mock the OpenAI client creation at the module level
    with patch('openai.OpenAI') as mock_openai_client:
        # Create a mock client instance with proper response structure
        mock_client = Mock()
        
        # Mock embeddings response with proper structure
        mock_embedding_response = Mock()
        mock_embedding_response.data = [Mock(embedding=[0.1] * 1536)]
        mock_embedding_response.model_dump.return_value = {
            "data": [{"embedding": [0.1] * 1536}]
        }
        mock_client.embeddings.create.return_value = mock_embedding_response
        
        # Mock chat completion response with proper structure
        mock_chat_response = Mock()
        mock_chat_response.choices = [Mock(message=Mock(content="Test response"))]
        mock_chat_response.model_dump.return_value = {
            "choices": [{
                "message": {
                    "content": "Test response",
                    "role": "assistant"  # Required for ChatMessage validation
                }
            }],
            "error": None  # No error in the response
        }
        mock_client.chat.completions.create.return_value = mock_chat_response
        
        mock_openai_client.return_value = mock_client
        yield mock_client


@pytest.fixture(scope="session", autouse=True) 
def mock_embeddings_globally():
    """Mock OpenAI embeddings globally"""
    with patch('langchain_openai.OpenAIEmbeddings') as mock_embeddings_class:
        mock_embeddings = Mock()
        mock_embeddings.embed_documents.return_value = [[0.1] * 1536] * 5
        mock_embeddings.embed_query.return_value = [0.1] * 1536
        mock_embeddings_class.return_value = mock_embeddings
        yield mock_embeddings


@pytest.fixture(scope="session", autouse=True)
def mock_chat_globally():
    """Mock OpenAI Chat globally"""  
    with patch('langchain_openai.ChatOpenAI') as mock_chat_class:
        mock_chat = Mock()
        mock_chat.invoke.return_value = Mock(content="Test response")
        mock_chat_class.return_value = mock_chat
        yield mock_chat


@pytest.fixture(scope="session", autouse=True)
def mock_qdrant_globally():
    """Mock QdrantVectorStore globally"""
    with patch('langchain_qdrant.QdrantVectorStore') as mock_qdrant_class:
        mock_qdrant = Mock()
        mock_qdrant.similarity_search.return_value = [
            Mock(page_content="Test document", metadata={"source": "test.pdf"})
        ]
        # Mock the problematic validation method
        mock_qdrant._validate_collection_config = Mock()
        mock_qdrant_class.return_value = mock_qdrant
        yield mock_qdrant


# Individual mock fixtures for specific use cases
@pytest.fixture
def mock_llm():
    """Mock LLM for individual tests"""
    mock = Mock()
    mock.invoke.return_value = Mock(content="Test response")
    return mock


@pytest.fixture
def mock_embeddings():
    """Mock embeddings for individual tests"""
    mock = Mock()
    mock.embed_documents.return_value = [[0.1] * 1536] * 5
    mock.embed_query.return_value = [0.1] * 1536
    return mock


@pytest.fixture
def mock_vector_store():
    """Mock vector store for individual tests"""
    mock = Mock()
    mock.similarity_search.return_value = [
        Mock(page_content="Test document", metadata={"source": "test.pdf"})
    ]
    return mock


@pytest.fixture
def mock_qdrant_client():
    """Mock Qdrant client for individual tests"""
    mock = Mock()
    mock.get_collection.return_value = Mock(config=Mock())
    mock.scroll.return_value = ([], None)
    return mock


@pytest.fixture
def sample_pdf_content():
    """Sample PDF content for testing"""
    return b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n"


@pytest.fixture
def sample_documents():
    """Sample documents for testing"""
    from langchain.schema import Document
    return [
        Document(page_content="Sample document 1", metadata={"source": "doc1.pdf"}),
        Document(page_content="Sample document 2", metadata={"source": "doc2.pdf"}),
    ]


import os
import pytest
import tempfile
from unittest.mock import Mock, patch
from typing import Generator

# Set test environment variables
os.environ.setdefault("OPENAI_API_KEY", "test-key-for-testing")


@pytest.fixture(scope="session")
def test_data_dir() -> Generator[str, None, None]:
    """Create a temporary directory for test data"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def mock_openai_key():
    """Mock OpenAI API key for tests"""
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
        yield


@pytest.fixture
def mock_qdrant_client():
    """Create a mock Qdrant client"""
    client = Mock()
    client.create_collection = Mock()
    client.get_collection = Mock()
    client.scroll = Mock()
    return client


@pytest.fixture
def mock_embeddings():
    """Mock OpenAI embeddings"""
    with patch('app.rag_graph.OpenAIEmbeddings') as mock_emb:
        mock_instance = Mock()
        mock_emb.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_llm():
    """Mock OpenAI LLM"""
    with patch('app.rag_graph.ChatOpenAI') as mock_llm:
        mock_instance = Mock()
        mock_response = Mock()
        mock_response.content = "Mocked LLM response"
        mock_instance.invoke.return_value = mock_response
        mock_llm.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_vector_store():
    """Mock Qdrant vector store"""
    with patch('app.rag_graph.QdrantVectorStore') as mock_store:
        mock_instance = Mock()
        mock_retriever = Mock()
        mock_retriever.invoke.return_value = []
        mock_instance.as_retriever.return_value = mock_retriever
        mock_instance.similarity_search.return_value = []
        mock_instance.add_documents.return_value = None
        mock_store.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def sample_documents():
    """Create sample documents for testing"""
    from langchain_core.documents import Document
    
    return [
        Document(
            page_content="Artificial Intelligence (AI) is the simulation of human intelligence in machines.",
            metadata={"document_id": "doc1", "chunk_id": 0, "source": "test.pdf"}
        ),
        Document(
            page_content="Machine Learning is a subset of AI that enables computers to learn automatically.",
            metadata={"document_id": "doc1", "chunk_id": 1, "source": "test.pdf"}
        ),
        Document(
            page_content="Deep Learning uses neural networks with multiple layers to model complex patterns.",
            metadata={"document_id": "doc2", "chunk_id": 0, "source": "test2.pdf"}
        )
    ]


@pytest.fixture
def sample_state():
    """Create a sample enhanced state for testing"""
    import time
    from app.rag_graph import EnhancedState
    from langchain_core.documents import Document
    
    return {
        "question": "What is artificial intelligence?",
        "context": [
            Document(page_content="AI is artificial intelligence", metadata={})
        ],
        "response": "",
        "start_time": time.time(),
        "retrieve_time": None,
        "generate_time": None,
        "total_time": None,
        "input_tokens": 0,
        "output_tokens": 0,
        "retrieval_count": 0,
        "graph_type": "test",
        "retrieved_documents": 0
    }


# Test configuration
def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )


# Pytest collection hooks
def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically"""
    for item in items:
        # Mark slow tests
        if "performance" in item.name or "compare" in item.name:
            item.add_marker(pytest.mark.slow)


# Custom pytest command line options
def pytest_addoption(parser):
    """Add custom command line options"""
    parser.addoption(
        "--run-slow",
        action="store_true", 
        default=False,
        help="Run slow tests"
    )
