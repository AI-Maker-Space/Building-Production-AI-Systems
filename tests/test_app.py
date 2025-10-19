"""
Test suite for the FastAPI RAG application.

This test suite covers:
1. API endpoints functionality
2. File upload and processing
3. Query processing with different graph types
4. Performance metrics in API responses
5. Error handling in API layer
"""

import os
import pytest
import tempfile
import json
from unittest.mock import Mock, patch, MagicMock, PropertyMock
from fastapi.testclient import TestClient

# Import the app
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from app.app import app, qdrant_client
from app.rag_graph import GraphType


# Test client
client = TestClient(app)


class TestHealthCheck:
    """Test basic health and status endpoints"""
    
    def test_graph_types_endpoint(self):
        """Test the graph types listing endpoint"""
        response = client.get("/graph-types")
        assert response.status_code == 200
        data = response.json()
        assert "available_types" in data
        
        available_types = data["available_types"]
        assert len(available_types) == 4
        
        # Check that all expected graph types are present
        type_names = [item["type"] for item in available_types]
        expected_types = ["simple", "multi_query", "ensemble", "hybrid"]
        assert all(gt in type_names for gt in expected_types)
    



class TestFileUpload:
    """Test file upload functionality"""
    
    def test_upload_endpoint_exists(self):
        """Test that upload endpoint is available"""
        # Test with no file (should return error)
        response = client.post("/documents")
        assert response.status_code == 422  # Validation error for missing file
    
    @patch('app.app.upsert_pdf_for_document')
    def test_upload_pdf_success(self, mock_upsert):
        """Test successful PDF upload"""
        mock_upsert.return_value = 5  # Mock 5 chunks uploaded
        
        # Create a fake PDF file
        fake_pdf_content = b"fake pdf content"
        files = {"file": ("test.pdf", fake_pdf_content, "application/pdf")}
        
        response = client.post("/documents", files=files)
        assert response.status_code == 200
        
        data = response.json()
        assert "document_id" in data
        assert "chunks" in data
        assert "status" in data
        assert data["chunks"] == 5
        assert data["status"] == "ready"
        
        # Verify the upsert function was called
        mock_upsert.assert_called_once()
    
    def test_upload_non_pdf_file(self):
        """Test uploading non-PDF file"""
        fake_content = b"not a pdf"
        files = {"file": ("test.txt", fake_content, "text/plain")}
        
        response = client.post("/documents", files=files)
        assert response.status_code == 400
        data = response.json()
        assert "detail" in data
        assert "PDF" in data["detail"]
    
    @patch('app.app.upsert_pdf_for_document')
    def test_upload_processing_error(self, mock_upsert):
        """Test handling of upload processing errors"""
        mock_upsert.side_effect = Exception("Processing failed")
        
        fake_pdf_content = b"fake pdf content"
        files = {"file": ("test.pdf", fake_pdf_content, "application/pdf")}
        
        response = client.post("/documents", files=files)
        assert response.status_code == 500
        data = response.json()
        assert "detail" in data


class TestQueryEndpoints:
    """Test query processing endpoints"""
    
    @patch('app.app.build_generalized_graph')
    def test_simple_query_success(self, mock_build_graph):
        """Test successful simple query"""
        # Mock the graph and its response
        mock_graph = Mock()
        mock_graph.invoke.return_value = {
            "response": "Test response"
        }
        mock_build_graph.return_value = mock_graph
        
        response = client.post(
            "/ask-simple",
            data={"question": "What is AI?"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert data["answer"] == "Test response"
    
    def test_query_missing_question(self):
        """Test query without question"""
        response = client.post("/ask-simple", data={})
        assert response.status_code == 422  # Validation error
    
    def test_query_missing_question(self):
        """Test query without question"""
        response = client.post("/ask-simple", data={})
        assert response.status_code == 422  # Validation error
    """Test performance comparison endpoint"""


class TestDebugEndpoints:
    """Test debug and utility endpoints"""
    
    @patch('app.app.debug_collection_contents')
    def test_debug_collection_success(self, mock_debug):
        """Test successful collection debugging"""
        mock_debug.return_value = {
            "collection_exists": True,
            "points_count": 100,
            "unique_document_ids": ["doc1", "doc2"],
            "sample_points": []
        }
        
        response = client.get("/debug/collection")
        assert response.status_code == 200
        data = response.json()
        assert data["collection_exists"] is True
        assert data["points_count"] == 100
    
    @patch('app.app.debug_collection_contents')
    def test_debug_collection_not_found(self, mock_debug):
        """Test debugging non-existent collection"""
        mock_debug.return_value = {
            "collection_exists": False,
            "error": "Collection not found"
        }
        
        response = client.get("/debug/collection")
        assert response.status_code == 200
        data = response.json()
        assert data["collection_exists"] is False
        assert "error" in data


class TestErrorHandling:
    """Test global error handling"""
    
    def test_404_endpoint(self):
        """Test accessing non-existent endpoint"""
        response = client.get("/nonexistent")
        assert response.status_code == 404
    
    def test_method_not_allowed(self):
        """Test using wrong HTTP method"""
        response = client.get("/documents")  # Should be POST
        assert response.status_code == 405


class TestRequestValidation:
    """Test request validation and schemas"""
    
    def test_query_request_validation(self):
        """Test query request validation"""
        # Test with wrong field type (not applicable for form data)
        # Just test missing required field
        response = client.post("/ask-simple", data={})
        assert response.status_code == 422
    
    def test_compare_performance_validation(self):
        """Test performance comparison request validation"""
        # Test with missing question
        response = client.post("/compare-graphs", data={})
        assert response.status_code == 422


class TestResponseSchemas:
    """Test API response schemas"""
    
    def test_health_endpoint_schema(self):
        """Test that health endpoints return expected schema"""
        response = client.get("/graph-types")
        assert response.status_code == 200
        data = response.json()
        assert "available_types" in data
        assert isinstance(data["available_types"], list)


# Integration test fixtures
@pytest.fixture(scope="module")
def test_client():
    """Create a test client for the FastAPI app"""
    return TestClient(app)


@pytest.fixture
def sample_pdf_file():
    """Create a sample PDF-like file for testing"""
    return ("test.pdf", b"fake pdf content", "application/pdf")


# Test configuration
def test_app_startup():
    """Test that the app starts up correctly"""
    with TestClient(app) as test_client:
        response = test_client.get("/graph-types")
        assert response.status_code == 200


if __name__ == "__main__":
    # Run tests when script is executed directly
    pytest.main([__file__, "-v"])
