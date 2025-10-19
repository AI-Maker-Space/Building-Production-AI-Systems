# 🤖 End-to-End RAG: From Graphs to Production

This directory contains a **production-ready Retrieval-Augmented Generation (RAG)** system showcasing advanced RAG strategies with comprehensive testing. Built with FastAPI, LangChain, LangGraph, and Qdrant vector database, this system demonstrates professional AI development practices from prototyping to production deployment.

## ✨ Key Features

- **🧠 4 Advanced RAG Strategies**: Simple, Multi-Query, Ensemble, and Hybrid RAG implementations
- **⚡ Modern Unified API**: Clean, type-safe interface with GraphType enums
- **🧪 Comprehensive Test Suite**: 41 tests with 100% pass rate and professional mocking
- **📊 Performance Metrics**: Built-in timing, token usage, and retrieval tracking
- **🚀 Production-Ready**: FastAPI backend with proper error handling and validation

## 🏗️ Architecture Overview

This system demonstrates professional AI development with multiple components:

### 🚀 FastAPI Backend (`src/app/app.py`)
- **Modern API Design**: RESTful endpoints with proper validation and error handling
- **Document Upload**: Accepts PDF files and processes them into searchable chunks
- **Multi-Strategy RAG**: Support for 4 different RAG approaches with performance comparison
- **Vector Storage**: Uses Qdrant (in-memory) for efficient document retrieval
- **AI Integration**: Leverages OpenAI's GPT-4 and embedding models

### 🧠 RAG Graph System (`src/app/rag_graph.py`)
- **Strategy Pattern**: Clean implementation of multiple RAG strategies
- **Graph-based Workflows**: LangGraph orchestrates retrieval and generation
- **Performance Monitoring**: Built-in metrics for timing, tokens, and retrieval counts
- **Unified Builder**: Single `build_rag_graph()` function for all strategies
- **Type Safety**: GraphType enums and TypedDict state definitions

### 🧪 Professional Test Suite (`tests/`)
- **Comprehensive Coverage**: 41 tests covering all RAG strategies and API endpoints
- **Advanced Mocking**: Session-level fixtures for OpenAI, Qdrant, and LangChain
- **Fast Execution**: Complete test suite runs in <1 second
- **Multiple Test Types**: Fast, unit, slow, and file-specific testing options

## 📋 Prerequisites

Before initializing the system, ensure you have:

1. **Python 3.13** installed
2. **OpenAI API Key** (required for embeddings and chat completion)
3. **Dependencies** managed via `uv` or `pip`

## 🚀 Quick Start Guide

### 1. Environment Setup

Navigate to the end-to-end-rag directory:
```bash
cd Building-Production-AI-Systems/end-to-end-rag
```

Set up your OpenAI API key:
```bash
export OPENAI_API_KEY="your-openai-api-key-here"
```

Or create a `.env` file in the end-to-end-rag directory:
```bash
echo "OPENAI_API_KEY=your-openai-api-key-here" > .env
```

### 2. Install Dependencies

Using `uv` (recommended):
```bash
uv sync
```

Activate the virtual environment (from project root):
```bash
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```


### 3. Run Tests (Recommended First Step)

Verify everything works with the comprehensive test suite:
```bash
# From end-to-end-rag directory
python tests/run_tests.py --type fast

# Or from tests directory
cd tests && python run_tests.py --type fast
```

Expected output: `✅ 41 passed in ~0.7s`

### 4. Run the Application

Start the FastAPI server:
```bash
uvicorn src.app.app:app --reload --port 8000
```

The API will be available at `http://localhost:8000`

### 5. API Documentation

Once running, visit:
- **Interactive API Docs**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

### 6. Test Suite Options

Explore different testing capabilities:
```bash
# Run all fast tests (default)
python tests/run_tests.py --type fast

# Run specific test files
python tests/run_tests.py --file tests/test_rag_graph.py  # 27 RAG core tests
python tests/run_tests.py --file tests/test_app.py       # 14 API tests

# Run with coverage
python tests/run_tests.py --type fast --coverage

# Run all tests including slow performance comparisons
python tests/run_tests.py --type all

# Get help
python tests/run_tests.py --help
```

## 🔧 API Endpoints

### Core Endpoints

#### Document Upload
```http
POST /documents
Content-Type: multipart/form-data
```
Upload a PDF document to be processed and stored in the vector database.

#### Simple Questions
```http
POST /ask-simple
Content-Type: application/x-www-form-urlencoded
```
Ask questions using the default Simple RAG strategy.

#### Advanced Questions with Metrics
```http
POST /ask-with-metrics
Content-Type: application/x-www-form-urlencoded

question=What is AI?&graph_type=ensemble&k=5
```
Ask questions with specific RAG strategy and get detailed performance metrics.

**Supported RAG Strategies:**
- `simple`: Basic semantic similarity retrieval
- `multi_query`: Multiple reformulated queries for better retrieval
- `ensemble`: Combines semantic and BM25 keyword-based retrieval  
- `hybrid`: Semantic retrieval with keyword-based reranking

#### Performance Comparison
```http
GET /performance-comparison?question=What is AI?&k=5
```
Compare all RAG strategies for the same question with detailed metrics.

#### Available Graph Types
```http
GET /graph-types
```
Get list of available RAG strategies and their descriptions.

## 🧪 Testing the System

### 1. Upload a Document
```bash
curl -X POST "http://localhost:8000/documents" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@data/howpeopleuseai.pdf"
```

### 2. Ask a Simple Question
```bash
curl -X POST "http://localhost:8000/ask-simple" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "question=What are the key findings?&k=5"
```

### 3. Compare RAG Strategies
```bash
curl -X GET "http://localhost:8000/performance-comparison?question=What is AI?&k=3"
```

### 4. Advanced Query with Metrics
```bash
curl -X POST "http://localhost:8000/ask-with-metrics" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "question=Explain the main concepts&graph_type=hybrid&k=5"
```

## 🛠️ Configuration

### Environment Variables
- `OPENAI_API_KEY`: Required for OpenAI API access
- `QDRANT_LOCATION`: Vector database location (default: `:memory:`)
- `COLLECTION`: Collection name for documents (default: `use_case_data`)

### Model Configuration
The system uses these OpenAI models by default:
- **Embeddings**: `text-embedding-3-small`
- **Chat**: `gpt-4.1-nano`

You can modify these in `src/app/app.py` if needed.

### RAG Strategy Selection
Choose the best strategy for your use case:

| Strategy | Best For | Pros | Cons |
|----------|----------|------|------|
| **Simple** | Quick queries, clear documents | Fast, straightforward | May miss context |
| **Multi-Query** | Complex questions | Better recall | Slower, more tokens |
| **Ensemble** | Mixed content types | Combines semantic + keyword | More complex |
| **Hybrid** | Precise results | Best of both worlds | Highest complexity |

## 📁 Project Structure

```
end-to-end-rag/
├── README.md                    # This file
├── pyproject.toml              # Project dependencies and configuration
├── pytest.ini                 # Pytest configuration
├── .env                        # Environment variables (create this)
├── data/
│   └── howpeopleuseai.pdf     # Sample PDF for testing
├── src/
│   └── app/
│       ├── __init__.py        # Package initialization
│       ├── app.py            # FastAPI application & API endpoints
│       └── rag_graph.py      # RAG strategies & LangGraph workflows
├── tests/
│   ├── run_tests.py          # Custom test runner with multiple options
│   ├── conftest.py           # Pytest fixtures and configuration
│   ├── test_rag_graph.py     # RAG core functionality tests (27 tests)
│   ├── test_app.py           # FastAPI API endpoint tests (14 tests)
│   └── test-rag-api.md       # Test documentation and examples
└── uploads/                   # Directory for uploaded files (auto-created)
```

## 🔍 How It Works

### RAG Pipeline Overview
1. **Document Upload**: PDFs are processed using PyMuPDF, split into chunks, and converted to embeddings
2. **Vector Storage**: Embeddings are stored in Qdrant for efficient similarity search
3. **Strategy Selection**: Choose from 4 RAG strategies based on your needs
4. **Question Processing**: User questions are processed according to the selected strategy
5. **Context Retrieval**: Relevant document chunks are retrieved using various methods
6. **Response Generation**: Retrieved context is fed to GPT-4 to generate accurate answers
7. **Performance Tracking**: Metrics collected for timing, tokens, and retrieval counts

### Advanced RAG Strategies

#### 🎯 Simple RAG
Basic semantic similarity retrieval - fastest and most straightforward approach.

#### 🔍 Multi-Query RAG  
Generates multiple reformulated queries to improve retrieval recall and find diverse relevant content.

#### ⚖️ Ensemble RAG
Combines semantic similarity with BM25 keyword-based retrieval for comprehensive results.

#### 🎭 Hybrid RAG
Uses semantic retrieval followed by keyword-based reranking for optimal precision and recall.

### Testing & Quality Assurance
- **Professional Test Suite**: 41 comprehensive tests covering all components
- **Advanced Mocking**: Session-level fixtures for external API dependencies
- **Performance Validation**: All RAG strategies tested with metrics collection
- **API Testing**: Complete FastAPI endpoint validation with error handling

## 🚨 Important Notes

- **In-Memory Storage**: By default, Qdrant runs in-memory mode, so uploaded documents are lost when the server restarts
- **API Key Security**: Never commit your OpenAI API key to version control
- **File Uploads**: Only PDF files are currently supported
- **Python Version**: This project requires Python 3.13 exactly
- **Test Suite**: Always run tests first to verify your environment setup
- **Modern API**: Uses unified `build_rag_graph()` function - no legacy mode references
- **Workshop Ready**: Designed for educational and production demonstration purposes

## 🐛 Troubleshooting

### Common Issues

1. **Tests Failing**
   - Run `python tests/run_tests.py --type fast` to verify your setup
   - Expected: `41 passed in ~0.7s` with 100% success rate

2. **"OPENAI_API_KEY must be set" Error**
   - Ensure your OpenAI API key is properly set as an environment variable
   - Tests work without API key (use comprehensive mocking)

3. **Import Errors**
   - Make sure all dependencies are installed: `uv sync` or `pip install -e .`
   - Activate virtual environment: `source .venv/bin/activate`

4. **Port Already in Use**
   - Change the port: `uvicorn src.app.app:app --reload --port 8001`

5. **PDF Processing Fails**
   - Ensure your PDF is not password-protected or corrupted
   - Try with the included sample: `data/howpeopleuseai.pdf`

6. **Wrong Directory**
   - Make sure you're in the `end-to-end-rag/` directory
   - Virtual environment is in the parent directory: `../venv/`

## 🚀 Production Deployment

For production deployment, consider:
- Using a persistent Qdrant instance instead of in-memory storage
- Implementing proper authentication and rate limiting
- Adding monitoring and logging with structured metrics
- Using environment-specific configuration
- Setting up CI/CD with the test suite
- Implementing async processing for large documents

## 📊 Performance Metrics

The system tracks comprehensive metrics for each RAG strategy:
- **Timing**: Total time, retrieval time, generation time
- **Token Usage**: Input tokens, output tokens for cost tracking
- **Retrieval Quality**: Number of documents retrieved
- **Strategy Comparison**: Side-by-side performance analysis

## 🎓 Educational Value

This project demonstrates:
- **Modern RAG Patterns**: Multiple strategies with clear trade-offs
- **Professional Testing**: Comprehensive test suite with 100% reliability
- **Production Practices**: Error handling, validation, monitoring
- **API Design**: Clean, documented, type-safe endpoints
- **Performance Analysis**: Built-in metrics and comparison tools

## 🔗 Related Resources

- **LangChain Documentation**: https://python.langchain.com/
- **LangGraph Guide**: https://langchain-ai.github.io/langgraph/
- **FastAPI Documentation**: https://fastapi.tiangolo.com/
- **Qdrant Vector Database**: https://qdrant.tech/

---

**🚀 Ready for Production!** This RAG system combines educational clarity with production-grade reliability - perfect for workshops, demonstrations, and real-world deployments.
