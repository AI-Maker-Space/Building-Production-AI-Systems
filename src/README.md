# 🤖 RAG AI System - Source Code

This directory contains the source code for a **Retrieval-Augmented Generation (RAG)** AI system built with FastAPI, LangChain, and Qdrant vector database. The system allows users to upload PDF documents and ask questions about their content using OpenAI's GPT models.

## 🏗️ Architecture Overview

The application consists of two main components:

### 🚀 FastAPI Backend (`app/app.py`)
- **Document Upload**: Accepts PDF files and processes them into searchable chunks
- **Question Answering**: Provides a chat interface to query uploaded documents
- **Vector Storage**: Uses Qdrant (in-memory) for efficient document retrieval
- **AI Integration**: Leverages OpenAI's GPT-4 and embedding models

### 🧠 RAG Graph System (`app/rag_graph.py`)
- **Document Processing**: Chunks PDFs using LangChain's text splitters
- **Vector Embeddings**: Creates embeddings using OpenAI's `text-embedding-3-small`
- **Graph-based RAG**: Implements a LangGraph workflow for context retrieval and response generation
- **Prompt Engineering**: Uses structured prompts for accurate, context-based responses

## 📋 Prerequisites

Before initializing the system, ensure you have:

1. **Python 3.13** installed
2. **OpenAI API Key** (required for embeddings and chat completion)
3. **Dependencies** managed via `uv` or `pip`

## 🚀 Quick Start Guide

### 1. Environment Setup

First, navigate to the project root directory:
```bash
cd /path/to/Building-Production-AI-Systems
```

Set up your OpenAI API key:
```bash
export OPENAI_API_KEY="your-openai-api-key-here"
```

Or create a `.env` file in the project root:
```bash
echo "OPENAI_API_KEY=your-openai-api-key-here" > .env
```

### 2. Install Dependencies

Using `uv` (recommended):
```bash
uv sync
```

Or using pip:
```bash
pip install -e .
```

### 3. Run the Application

Start the FastAPI server:
```bash
uvicorn src.app.app:app --reload --port 8000
```

The API will be available at `http://localhost:8000`

### 4. API Documentation

Once running, visit:
- **Interactive API Docs**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## 🔧 API Endpoints

### Document Upload
```http
POST /documents
Content-Type: multipart/form-data
```
Upload a PDF document to be processed and stored in the vector database.

**Response:**
```json
{
  "document_id": "uuid-string",
  "chunks": 42,
  "status": "ready"
}
```

### Ask Questions
```http
POST /ask
Content-Type: application/json
```
Ask questions about the uploaded documents.

**Request Body:**
```json
{
  "question": "What is the main topic of the document?"
}
```

**Response:**
```json
{
  "answer": "Based on the uploaded document, the main topic is..."
}
```

## 🧪 Testing the System

### 1. Upload a Document
```bash
curl -X POST "http://localhost:8000/documents" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/your/document.pdf"
```

### 2. Ask a Question
```bash
curl -X POST "http://localhost:8000/ask" \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the key findings in this document?"}'
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

## 📁 File Structure

```
src/
├── README.md              # This file
└── app/
    ├── __init__.py        # Package initialization
    ├── app.py            # FastAPI application & API endpoints
    └── rag_graph.py      # RAG processing logic & LangGraph workflow
```

## 🔍 How It Works

1. **Document Upload**: PDFs are processed using PyMuPDF, split into chunks, and converted to embeddings
2. **Vector Storage**: Embeddings are stored in Qdrant for efficient similarity search
3. **Question Processing**: User questions are embedded and used to retrieve relevant document chunks
4. **Response Generation**: Retrieved context is fed to GPT-4 to generate accurate, contextual answers
5. **Graph Workflow**: LangGraph orchestrates the retrieval and generation process

## 🚨 Important Notes

- **In-Memory Storage**: By default, Qdrant runs in-memory mode, so uploaded documents are lost when the server restarts
- **API Key Security**: Never commit your OpenAI API key to version control
- **File Uploads**: Only PDF files are currently supported
- **Python Version**: This project requires Python 3.13 exactly

## 🐛 Troubleshooting

### Common Issues

1. **"OPENAI_API_KEY must be set" Error**
   - Ensure your OpenAI API key is properly set as an environment variable

2. **Import Errors**
   - Make sure all dependencies are installed: `uv sync` or `pip install -e .`

3. **Port Already in Use**
   - Change the port: `uvicorn src.app.app:app --reload --port 8001`

4. **PDF Processing Fails**
   - Ensure your PDF is not password-protected or corrupted

## 🚀 Deployment

For production deployment, consider:
- Using a persistent Qdrant instance instead of in-memory storage
- Implementing proper authentication and rate limiting
- Adding monitoring and logging
- Using environment-specific configuration

---

**Need help?** Check the main project README or consult the FastAPI and LangChain documentation.
