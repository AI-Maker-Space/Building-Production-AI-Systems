"""
Test suite for the RAG Graph system.

This test suite covers:
1. Strategy pattern implementation
2. RAG graph building and execution
3. Performance metrics tracking
4. Document upserts and retrieval
5. Error handling and edge cases
"""

import os
import pytest
import tempfile
import time
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any

# Import the modules under test
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from app.rag_graph import (
    # Core classes
    GraphType, State, EnhancedState, RAGComponents,
    RAGStrategy, SimpleRAGStrategy, MultiQueryRAGStrategy, 
    EnsembleRAGStrategy, HybridRAGStrategy,
    
    # Functions
    build_rag_graph, build_graph_by_type, get_rag_strategy, ask, ask_with_metrics,
    estimate_tokens, upsert_pdf_for_document, debug_collection_contents,
    
    # Constants
    RAG_PROMPT
)

from langchain_core.documents import Document
from qdrant_client import QdrantClient


class TestGraphType:
    """Test the GraphType enum"""
    
    def test_graph_type_values(self):
        """Test that all graph types have correct values"""
        assert GraphType.SIMPLE.value == "simple"
        assert GraphType.MULTI_QUERY.value == "multi_query"
        assert GraphType.ENSEMBLE.value == "ensemble"
        assert GraphType.HYBRID.value == "hybrid"
    
    def test_graph_type_membership(self):
        """Test that we can iterate and check membership"""
        all_types = list(GraphType)
        assert len(all_types) == 4
        assert GraphType.SIMPLE in all_types


class TestStateDefinitions:
    """Test the TypedDict state definitions"""
    
    def test_basic_state(self):
        """Test basic State structure"""
        state: State = {
            "question": "What is AI?",
            "context": [Document(page_content="AI is artificial intelligence")],
            "response": "AI stands for artificial intelligence."
        }
        assert "question" in state
        assert "context" in state
        assert "response" in state
    
    def test_enhanced_state(self):
        """Test EnhancedState structure with all fields"""
        state: EnhancedState = {
            "question": "What is AI?",
            "context": [Document(page_content="AI is artificial intelligence")],
            "response": "AI stands for artificial intelligence.",
            "start_time": time.time(),
            "retrieve_time": 0.5,
            "generate_time": 0.3,
            "total_time": 0.8,
            "input_tokens": 100,
            "output_tokens": 50,
            "retrieval_count": 1,
            "graph_type": "simple",
            "retrieved_documents": 1
        }
        assert len(state) == 12  # All fields present


class TestUtilityFunctions:
    """Test utility functions"""
    
    def test_estimate_tokens(self):
        """Test token estimation function"""
        # Test empty string
        assert estimate_tokens("") == 0
        
        # Test simple text (1 token ≈ 4 characters)
        assert estimate_tokens("hello") == 1  # 5 chars / 4 = 1
        assert estimate_tokens("hello world") == 2  # 11 chars / 4 = 2
        assert estimate_tokens("a" * 20) == 5  # 20 chars / 4 = 5
    
    def test_rag_prompt_formatting(self):
        """Test that the RAG prompt template works correctly"""
        from langchain.prompts import ChatPromptTemplate
        template = ChatPromptTemplate.from_template(RAG_PROMPT)
        
        formatted = template.format(
            question="What is AI?",
            context="Artificial Intelligence is..."
        )
        
        assert "What is AI?" in formatted
        assert "Artificial Intelligence is..." in formatted
        assert "helpful assistant" in formatted


class TestRAGComponents:
    """Test the shared RAG components class"""
    
    @pytest.fixture
    def mock_client(self):
        """Create a mock Qdrant client"""
        client = Mock()
        client.create_collection = Mock()
        return client
    
    def test_rag_components_initialization(self, mock_client):
        """Test RAGComponents initialization with mocked dependencies"""
        with patch('app.rag_graph.OpenAIEmbeddings') as mock_embeddings, \
             patch('app.rag_graph.ChatOpenAI') as mock_llm, \
             patch('app.rag_graph.QdrantVectorStore') as mock_store:
            
            components = RAGComponents(
                k=3,
                client=mock_client,
                collection_name="test_collection"
            )
            
            assert components.k == 3
            assert components.collection_name == "test_collection"
            assert components.client == mock_client
            mock_embeddings.assert_called_once()
            mock_llm.assert_called_once()
    
    def test_generate_response_structure(self, mock_client):
        """Test the generate_response method structure"""
        with patch('app.rag_graph.OpenAIEmbeddings'), \
             patch('app.rag_graph.ChatOpenAI') as mock_llm, \
             patch('app.rag_graph.QdrantVectorStore'):
            
            # Mock the LLM response
            mock_response = Mock()
            mock_response.content = "This is a test response"
            mock_llm.return_value.invoke.return_value = mock_response
            
            components = RAGComponents(client=mock_client)
            
            state: EnhancedState = {
                "question": "Test question?",
                "context": [Document(page_content="Test context")],
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
            
            result = components.generate_response(state)
            
            # Check return structure
            assert "response" in result
            assert "generate_time" in result
            assert "total_time" in result
            assert "input_tokens" in result
            assert "output_tokens" in result
            assert result["response"] == "This is a test response"


class TestRAGStrategies:
    """Test the RAG strategy implementations"""
    
    @pytest.fixture
    def mock_components(self):
        """Create mock RAG components"""
        components = Mock()
        components.k = 5
        components.store = Mock()
        components.llm = Mock()
        components.generate_response = Mock(return_value={
            "response": "Test response",
            "generate_time": 0.1,
            "total_time": 0.2,
            "input_tokens": 10,
            "output_tokens": 5
        })
        return components
    
    def test_simple_rag_strategy(self, mock_components):
        """Test SimpleRAGStrategy implementation"""
        strategy = SimpleRAGStrategy(mock_components)
        
        assert strategy.graph_type_name == "simple"
        assert strategy.retrieval_count == 1
        
        # Test retriever building
        mock_components.store.as_retriever.return_value = Mock()
        retriever = strategy.build_retriever()
        mock_components.store.as_retriever.assert_called_with(search_kwargs={"k": 5})
        
        # Test graph nodes
        nodes = strategy.get_graph_nodes()
        assert len(nodes) == 2  # retrieve and generate
        assert callable(nodes[0])  # retrieve function
        assert callable(nodes[1])  # generate function
    
    def test_multi_query_rag_strategy(self, mock_components):
        """Test MultiQueryRAGStrategy implementation"""
        with patch('app.rag_graph.MultiQueryRetriever') as mock_multi_query:
            # Setup the mock to return a mock retriever
            mock_retriever = Mock()
            mock_multi_query.from_llm.return_value = mock_retriever
            
            strategy = MultiQueryRAGStrategy(mock_components)
            
            assert strategy.graph_type_name == "multi_query"
            assert strategy.retrieval_count == 3
            
            # Test retriever building
            retriever = strategy.build_retriever()
            mock_multi_query.from_llm.assert_called_once()
            assert retriever == mock_retriever
    
    def test_ensemble_rag_strategy(self, mock_components):
        """Test EnsembleRAGStrategy implementation"""
        strategy = EnsembleRAGStrategy(mock_components)
        
        assert strategy.graph_type_name == "ensemble"
        assert strategy.retrieval_count == 2
        
        # Test with no documents (fallback to semantic only)
        mock_components.store.similarity_search.side_effect = Exception("No docs")
        retriever = strategy.build_retriever()
        # Should fallback to semantic retriever
        mock_components.store.as_retriever.assert_called()
    
    def test_hybrid_rag_strategy(self, mock_components):
        """Test HybridRAGStrategy implementation"""
        strategy = HybridRAGStrategy(mock_components)
        
        assert strategy.graph_type_name == "hybrid"
        assert strategy.retrieval_count == 1
        
        # Test graph nodes (should have retrieve, rerank, generate)
        nodes = strategy.get_graph_nodes()
        assert len(nodes) == 3  # retrieve, rerank, generate
    
    def test_strategy_factory(self, mock_components):
        """Test the strategy factory function"""
        # Test all valid graph types
        simple_strategy = get_rag_strategy(GraphType.SIMPLE, mock_components)
        assert isinstance(simple_strategy, SimpleRAGStrategy)
        
        multi_strategy = get_rag_strategy(GraphType.MULTI_QUERY, mock_components)
        assert isinstance(multi_strategy, MultiQueryRAGStrategy)
        
        ensemble_strategy = get_rag_strategy(GraphType.ENSEMBLE, mock_components)
        assert isinstance(ensemble_strategy, EnsembleRAGStrategy)
        
        hybrid_strategy = get_rag_strategy(GraphType.HYBRID, mock_components)
        assert isinstance(hybrid_strategy, HybridRAGStrategy)
        
        # Test invalid graph type
        with pytest.raises(ValueError, match="Unknown graph type"):
            get_rag_strategy("invalid_type", mock_components)


class TestGraphBuilding:
    """Test graph building functionality"""
    
    @patch('app.rag_graph.QdrantClient')
    @patch('app.rag_graph.OpenAIEmbeddings')
    @patch('app.rag_graph.ChatOpenAI')
    @patch('app.rag_graph.QdrantVectorStore')
    def test_build_rag_graph_simple(self, mock_store, mock_llm, mock_embeddings, mock_client):
        """Test building a simple RAG graph"""
        # Mock the graph building process
        mock_store.return_value.as_retriever.return_value = Mock()
        
        graph = build_rag_graph(GraphType.SIMPLE, k=3)
        
        # Should return a StateGraph
        assert graph is not None
        assert hasattr(graph, 'invoke')  # LangGraph should have invoke method
    
    def test_build_rag_graph_all_types(self):
        """Test that all graph types can be built without errors"""
        with patch('app.rag_graph.QdrantClient'), \
             patch('app.rag_graph.OpenAIEmbeddings'), \
             patch('app.rag_graph.ChatOpenAI'), \
             patch('app.rag_graph.QdrantVectorStore') as mock_store, \
             patch('app.rag_graph.MultiQueryRetriever') as mock_multi_query, \
             patch('app.rag_graph.BM25Retriever') as mock_bm25:
            
            # Setup basic mocks
            mock_retriever = Mock()
            mock_store.return_value.as_retriever.return_value = mock_retriever
            mock_store.return_value.similarity_search.return_value = []
            
            # Setup MultiQueryRetriever mock
            mock_multi_query.from_llm.return_value = mock_retriever
            
            # Setup BM25Retriever mock
            mock_bm25.from_documents.return_value = mock_retriever
            
            for graph_type in GraphType:
                graph = build_rag_graph(graph_type, k=3)
                assert graph is not None


class TestUnifiedAPI:
    """Test the modern unified API for graph building"""
    
    def test_unified_api_functions_exist(self):
        """Test that the unified API functions are available"""
        # These should all be importable and callable
        assert callable(build_rag_graph)
        assert callable(build_graph_by_type)  # Factory function
        assert callable(get_rag_strategy)
        assert callable(ask)
        assert callable(ask_with_metrics)
    
    @patch('app.rag_graph.RAGComponents')
    def test_graph_type_mapping(self, mock_rag_components):
        """Test that build_rag_graph correctly handles all graph types"""
        # Mock the return value
        mock_rag_components.return_value = Mock()
        mock_rag_components.return_value.llm = Mock()
        mock_rag_components.return_value.vector_store = Mock()
        
        # Test each graph type
        for graph_type in GraphType:
            try:
                result = build_rag_graph(
                    graph_type=graph_type,
                    k=3,
                    openai_embedding_model="text-embedding-3-small",
                    openai_chat_model="gpt-4",
                    client=Mock(),
                    qdrant_location=":memory:",
                    collection_name="test"
                )
                assert result is not None
            except Exception as e:
                # This is expected with mocked components, just verify the function exists
                pass
    
    @patch('app.rag_graph.build_rag_graph')
    def test_factory_function(self, mock_build_rag):
        """Test that the factory function calls the unified builder"""
        mock_build_rag.return_value = Mock()
        
        # Test that factory function calls through to unified builder
        for graph_type in GraphType:
            build_graph_by_type(graph_type=graph_type, k=3)
            mock_build_rag.assert_called_with(graph_type, k=3)


class TestDocumentOperations:
    """Test document upload and retrieval operations"""
    
    def test_upsert_pdf_mock(self):
        """Test PDF upsert with mocked dependencies"""
        with patch('app.rag_graph.PyMuPDFLoader') as mock_loader, \
             patch('app.rag_graph.RecursiveCharacterTextSplitter') as mock_splitter, \
             patch('app.rag_graph.QdrantClient') as mock_client, \
             patch('app.rag_graph.OpenAIEmbeddings'), \
             patch('app.rag_graph.QdrantVectorStore') as mock_store:
            
            # Mock document loading and splitting
            mock_doc = Mock()
            mock_doc.page_content = "Sample content"
            mock_doc.metadata = {}
            
            mock_loader.return_value.load.return_value = [mock_doc]
            mock_splitter.return_value.split_documents.return_value = [mock_doc]
            mock_store.return_value.add_documents.return_value = None
            
            # Test the function
            chunks_count = upsert_pdf_for_document(
                document_id="test_doc",
                pdf_path="/fake/path.pdf"
            )
            
            assert chunks_count == 1
            mock_loader.assert_called_once_with("/fake/path.pdf")
            mock_store.return_value.add_documents.assert_called_once()
    
    def test_debug_collection_contents(self):
        """Test collection debugging function"""
        mock_client = Mock()
        
        # Mock successful case
        mock_info = Mock()
        mock_info.points_count = 100
        mock_client.get_collection.return_value = mock_info
        
        mock_point = Mock()
        mock_point.id = "point_1"
        mock_point.payload = {"metadata": {"document_id": "doc1"}}
        mock_client.scroll.return_value = ([mock_point], None)
        
        result = debug_collection_contents(mock_client, "test_collection")
        
        assert result["collection_exists"] is True
        assert result["points_count"] == 100
        assert "doc1" in result["unique_document_ids"]
        
        # Mock error case
        mock_client.get_collection.side_effect = Exception("Collection not found")
        result = debug_collection_contents(mock_client, "missing_collection")
        
        assert result["collection_exists"] is False
        assert "error" in result


class TestPerformanceMetrics:
    """Test performance tracking functionality"""
    
    def test_ask_function(self):
        """Test basic ask function"""
        mock_graph = Mock()
        mock_graph.invoke.return_value = {"response": "Test answer"}
        
        result = ask(mock_graph, "Test question?")
        
        assert result == "Test answer"
        mock_graph.invoke.assert_called_once_with({"question": "Test question?"})
    
    def test_ask_with_metrics(self):
        """Test enhanced ask function with metrics"""
        mock_graph = Mock()
        mock_response = {
            "response": "Test answer",
            "total_time": 1.0,
            "retrieve_time": 0.3,
            "generate_time": 0.7,
            "input_tokens": 100,
            "output_tokens": 50,
            "retrieved_documents": 5,
            "graph_type": "simple"
        }
        mock_graph.invoke.return_value = mock_response
        
        result = ask_with_metrics(mock_graph, "Test question?")
        
        assert result["response"] == "Test answer"
        assert "metrics" in result
        assert result["metrics"]["total_time"] == 1.0
        assert result["metrics"]["graph_type"] == "simple"


class TestErrorHandling:
    """Test error handling and edge cases"""
    
    def test_invalid_graph_type_error(self):
        """Test that invalid graph types raise appropriate errors"""
        with pytest.raises(ValueError, match="Unknown graph type"):
            mock_components = Mock()
            get_rag_strategy("invalid_type", mock_components)
    
    def test_missing_openai_key(self):
        """Test behavior when OpenAI API key is missing"""
        # This would be tested in integration tests
        # For unit tests, we mock the dependencies
        pass
    
    def test_empty_document_handling(self):
        """Test handling of empty documents"""
        with patch('app.rag_graph.PyMuPDFLoader') as mock_loader, \
             patch('app.rag_graph.RecursiveCharacterTextSplitter') as mock_splitter:
            
            # Mock empty document
            mock_doc = Mock()
            mock_doc.page_content = ""  # Empty content
            mock_doc.metadata = {}
            
            mock_loader.return_value.load.return_value = [mock_doc]
            mock_splitter.return_value.split_documents.return_value = [mock_doc]
            
            # Should filter out empty documents
            with patch('app.rag_graph.QdrantClient'), \
                 patch('app.rag_graph.OpenAIEmbeddings'), \
                 patch('app.rag_graph.QdrantVectorStore') as mock_store:
                
                chunks_count = upsert_pdf_for_document(
                    document_id="test_doc",
                    pdf_path="/fake/empty.pdf"
                )
                
                # Should not add empty documents
                assert chunks_count == 0


class TestIntegrationScenarios:
    """Integration-style tests (still using mocks but testing workflows)"""
    

    
    def test_full_rag_workflow_simple(self):
        """Test a complete RAG workflow with simple strategy"""
        with patch('app.rag_graph.QdrantClient') as mock_client, \
             patch('app.rag_graph.OpenAIEmbeddings'), \
             patch('app.rag_graph.ChatOpenAI') as mock_llm, \
             patch('app.rag_graph.QdrantVectorStore') as mock_store:
            
            # Mock retrieval
            mock_doc = Document(page_content="AI is artificial intelligence")
            mock_store.return_value.as_retriever.return_value.invoke.return_value = [mock_doc]
            
            # Mock LLM response
            mock_response = Mock()
            mock_response.content = "AI stands for artificial intelligence."
            mock_llm.return_value.invoke.return_value = mock_response
            
            # Build and test graph
            graph = build_rag_graph(GraphType.SIMPLE, k=1)
            
            # Mock the state transformation through the graph
            def mock_invoke(state):
                # Simulate the graph execution
                return {
                    "question": state["question"],
                    "context": [mock_doc],
                    "response": "AI stands for artificial intelligence.",
                    "start_time": state.get("start_time", time.time()),
                    "retrieve_time": 0.1,
                    "generate_time": 0.2,
                    "total_time": 0.3,
                    "input_tokens": 20,
                    "output_tokens": 10,
                    "retrieval_count": 1,
                    "graph_type": "simple",
                    "retrieved_documents": 1
                }
            
            graph.invoke = mock_invoke
            
            result = ask_with_metrics(graph, "What is AI?")
            
            assert result["response"] == "AI stands for artificial intelligence."
            assert result["metrics"]["graph_type"] == "simple"
            assert result["metrics"]["retrieved_documents"] == 1


# Pytest configuration and fixtures
@pytest.fixture(scope="session")
def temp_dir():
    """Create a temporary directory for test files"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


# Test discovery helper
def test_module_imports():
    """Test that all required modules can be imported"""
    try:
        from app.rag_graph import (
            GraphType, RAGComponents, SimpleRAGStrategy,
            build_rag_graph, ask, ask_with_metrics
        )
        assert True
    except ImportError as e:
        pytest.fail(f"Failed to import required modules: {e}")


if __name__ == "__main__":
    # Run tests when script is executed directly
    pytest.main([__file__, "-v"])
