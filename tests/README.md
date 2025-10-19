# RAG System Test Suite

This directory contains comprehensive tests for the Building Production AI Systems RAG application.

## Test Structure

```
tests/
├── __init__.py              # Test package initialization
├── conftest.py              # Shared fixtures and configuration
├── test_rag_graph.py        # Core RAG functionality tests
└── test_app.py              # FastAPI application tests
```

## Test Categories

### 1. Unit Tests (`test_rag_graph.py`)

**Core Components:**
- `TestGraphType`: Enum validation and membership
- `TestStateDefinitions`: TypedDict structure validation
- `TestUtilityFunctions`: Token estimation, prompt formatting
- `TestRAGComponents`: Shared component initialization and methods
- `TestRAGStrategies`: Strategy pattern implementation
- `TestGraphBuilding`: Graph construction and compilation
- `TestLegacyAPI`: Backward compatibility verification
- `TestDocumentOperations`: PDF upload and processing
- `TestPerformanceMetrics`: Timing and token tracking
- `TestErrorHandling`: Exception handling and edge cases
- `TestIntegrationScenarios`: End-to-end workflow testing

### 2. API Tests (`test_app.py`)

**FastAPI Endpoints:**
- `TestHealthCheck`: Root and status endpoints
- `TestFileUpload`: PDF upload functionality
- `TestQueryEndpoints`: Query processing with different graph types
- `TestPerformanceComparison`: Multi-graph performance testing
- `TestDebugEndpoints`: Collection debugging utilities
- `TestErrorHandling`: HTTP error responses
- `TestRequestValidation`: Input validation and schemas
- `TestResponseSchemas`: Output format verification

## Running Tests

### Quick Start

```bash
# Install test dependencies
uv add pytest pytest-asyncio httpx

# Run fast tests (unit tests, no API calls)
python run_tests.py --type fast

# Run all tests with coverage
python run_tests.py --type all --coverage
```

### Test Commands

```bash
# Unit tests only (no API calls)
python run_tests.py --type unit

# Integration tests (requires OPENAI_API_KEY)
export OPENAI_API_KEY=your_key_here
python run_tests.py --type integration

# All tests
python run_tests.py --type all

# Slow tests only
python run_tests.py --type slow

# With coverage reporting
python run_tests.py --coverage
```

### Direct pytest Usage

```bash
# Run specific test file
pytest tests/test_rag_graph.py -v

# Run specific test class
pytest tests/test_rag_graph.py::TestRAGStrategies -v

# Run specific test method
pytest tests/test_rag_graph.py::TestRAGStrategies::test_simple_rag_strategy -v

# Run with markers
pytest -m "not slow" -v
pytest -m "unit" -v
pytest -m "integration" --run-integration -v
```

## Test Configuration

### Markers

Tests are organized with markers for selective execution:

- `@pytest.mark.unit`: Unit tests (fast, no external dependencies)
- `@pytest.mark.integration`: Integration tests (require API keys)
- `@pytest.mark.slow`: Slow-running tests
- `@pytest.mark.requires_api_key`: Tests that need real API keys

### Fixtures

Common fixtures are defined in `conftest.py`:

- `mock_qdrant_client`: Mock Qdrant database client
- `mock_embeddings`: Mock OpenAI embeddings
- `mock_llm`: Mock language model
- `mock_vector_store`: Mock vector store
- `sample_documents`: Test documents
- `sample_state`: Test state objects

## Test Coverage

The test suite covers:

- ✅ **Strategy Pattern**: All RAG strategies (Simple, Multi-Query, Ensemble, Hybrid)
- ✅ **Graph Building**: Unified graph builder with all types
- ✅ **Performance Metrics**: Timing, token counting, retrieval tracking
- ✅ **Error Handling**: Invalid inputs, API failures, edge cases
- ✅ **API Endpoints**: All FastAPI routes and HTTP methods
- ✅ **Backward Compatibility**: Legacy function interfaces
- ✅ **Document Processing**: PDF upload and chunking
- ✅ **Request/Response Validation**: Schema compliance

## Mocking Strategy

Tests use extensive mocking to:

1. **Avoid API Costs**: Mock OpenAI calls for unit tests
2. **Ensure Determinism**: Predictable test outcomes
3. **Test Error Conditions**: Simulate failure scenarios
4. **Speed**: Fast test execution without network calls
5. **Independence**: Tests don't require external services

## Integration Testing

Integration tests require:

```bash
export OPENAI_API_KEY=your_actual_api_key
```

These tests:
- Make real API calls to OpenAI
- Test actual document processing
- Verify end-to-end workflows
- Are slower and cost tokens

## Performance Testing

Performance tests measure:
- Graph execution times
- Token usage patterns
- Memory consumption
- Comparative performance across strategies

## Continuous Integration

For CI/CD pipelines:

```yaml
# Example GitHub Actions
- name: Run tests
  run: |
    python run_tests.py --type unit --coverage
  env:
    OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
```

## Test Development Guidelines

### Writing New Tests

1. **Use appropriate markers**: `@pytest.mark.unit` for fast tests
2. **Mock external dependencies**: API calls, file system operations
3. **Test both success and failure paths**: Happy path and edge cases
4. **Use descriptive test names**: `test_simple_rag_strategy_builds_correct_retriever`
5. **Follow AAA pattern**: Arrange, Act, Assert

### Example Test Structure

```python
class TestNewFeature:
    """Test description"""
    
    def test_feature_success_case(self, mock_dependency):
        """Test successful execution"""
        # Arrange
        setup_data = create_test_data()
        
        # Act
        result = feature_function(setup_data)
        
        # Assert
        assert result.status == "success"
        mock_dependency.assert_called_once()
    
    def test_feature_error_case(self, mock_dependency):
        """Test error handling"""
        # Arrange
        mock_dependency.side_effect = Exception("Test error")
        
        # Act & Assert
        with pytest.raises(Exception, match="Test error"):
            feature_function("invalid_input")
```

## Debugging Tests

```bash
# Run with detailed output
pytest -vvv -s tests/test_rag_graph.py::TestRAGStrategies::test_simple_rag_strategy

# Drop into debugger on failure
pytest --pdb tests/

# Show local variables in tracebacks
pytest --tb=long tests/

# Run only failed tests from last run
pytest --lf tests/
```

## Test Metrics

Track test quality with:
- **Coverage**: Aim for >90% code coverage
- **Speed**: Unit tests <1s each, integration tests <30s
- **Reliability**: Tests should pass consistently
- **Maintainability**: Clear, readable test code

## Contributing

When adding new features:

1. Write tests first (TDD approach)
2. Ensure tests cover success and failure cases
3. Update this README if adding new test categories
4. Run full test suite before submitting PR
5. Maintain or improve overall coverage percentage
