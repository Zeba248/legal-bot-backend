# 🏛️ Legal AI Chatbot - RAG-based Backend

A comprehensive Legal AI Chatbot backend designed for lawyers, built with RAG (Retrieval-Augmented Generation) architecture using Indian law documents (IPC and CrPC). The system provides accurate, context-based legal answers with proper source citations.

## 🎯 Features

### ✅ Core Functionality
- **RAG Architecture**: Complete implementation with retrieval and generation
- **PDF Processing**: Advanced OCR and text extraction for legal documents
- **Smart Chunking**: Efficient document chunking with legal context preservation
- **Vector Database**: ChromaDB for fast similarity search
- **Multiple Embeddings**: Support for OpenAI and HuggingFace embeddings
- **Custom Legal Prompt**: Specialized prompt template for Indian law
- **Source Citations**: Automatic section references (e.g., "As per IPC Section 420...")

### 🚀 API Endpoints
- `POST /ask` - Ask legal questions with RAG (requires API keys)
- `POST /search` - Search legal documents (works without API keys)
- `POST /upload-pdf` - Upload new legal documents
- `GET /get-chats` - Retrieve chat history
- `GET /system-info` - System status and configuration
- `GET /health` - Health check endpoint

### 🧠 Advanced Features
- **Confidence Scoring**: Assess answer reliability
- **Document Type Detection**: Automatic IPC/CrPC classification
- **Section Extraction**: Intelligent legal section identification
- **Modular Architecture**: Easily extendable components
- **Production Ready**: Comprehensive error handling and logging

## 📁 Project Structure

```
legal-bot-backend/
├── main.py                 # FastAPI application
├── pdf_processor.py        # PDF processing and chunking
├── vector_store.py         # ChromaDB vector database manager
├── rag_pipeline.py         # RAG pipeline implementation
├── initialize_data.py      # Data initialization script
├── utils.py               # Utility functions
├── requirements.txt       # Python dependencies
├── render.yaml           # Render deployment config
├── Data/                 # Legal PDF documents
│   ├── ipc.pdf
│   ├── crpc.pdf
│   └── crpc amendment.pdf
└── chroma_db/           # Vector database (auto-created)
```

## 🚀 Quick Start

### ⚡ One-Command Deployment
```bash
python3 deploy.py
```

This automated script will:
- ✅ Check Python version compatibility
- ✅ Set up virtual environment
- ✅ Install all dependencies
- ✅ Create environment configuration
- ✅ Process legal PDFs and create embeddings
- ✅ Initialize vector database
- ✅ Test system components

### 🔧 Manual Setup

#### 1. Environment Setup
```bash
# Clone the repository
git clone <repository-url>
cd legal-bot-backend

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

#### 2. Environment Variables (Optional)
```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your API keys (optional for basic functionality)
# GROQ_API_KEY=your_groq_api_key_here
# OPENAI_API_KEY=your_openai_api_key_here
```

#### 3. Initialize the System
```bash
# Process PDFs and create vector database
python3 initialize_data.py
```

#### 4. Start the Server
```bash
# Development server
python3 main.py

# Or with uvicorn
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### ⚡ Quick Test (No API Keys Required)
```bash
# Test document search (works without API keys)
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is theft according to IPC?"}'

# Check system status
curl "http://localhost:8000/system-info"
```

## 🔧 Configuration Options

### Embedding Models
- **HuggingFace** (default): `sentence-transformers/all-MiniLM-L6-v2`
- **OpenAI**: `text-embedding-3-small`

### LLM Providers
- **Groq** (default): `llama3-70b-8192`
- **OpenAI**: `gpt-3.5-turbo` or `gpt-4`

### Chunking Parameters
- **Chunk Size**: 1000 characters (configurable)
- **Chunk Overlap**: 200 characters (configurable)
- **Separators**: Optimized for legal documents

## 📚 API Documentation

### POST /search
Search legal documents without LLM generation (works without API keys).

**Request:**
```json
{
  "question": "What is theft according to IPC?",
  "max_docs": 5
}
```

**Response:**
```json
{
  "query": "What is theft according to IPC?",
  "num_results": 5,
  "results": [
    {
      "content": "Theft.—Whoever, intending to take dishonestly any movable property...",
      "metadata": {
        "source": "ipc.pdf",
        "document_type": "IPC",
        "chunk_id": 378,
        "total_chunks": 485
      },
      "content_preview": "Theft.—Whoever, intending to take dishonestly..."
    }
  ],
  "timestamp": "2024-01-15T10:30:00Z",
  "note": "This is document retrieval only. For full RAG answers, use /ask endpoint with API keys."
}
```

### POST /ask
Ask legal questions using RAG pipeline (requires API keys).

**Request:**
```json
{
  "question": "What is theft according to IPC?",
  "include_sources": true,
  "max_docs": 5
}
```

**Response:**
```json
{
  "answer": "According to the Indian Penal Code, theft is defined in Section 378...",
  "sources": [
    {
      "source": "ipc.pdf",
      "document_type": "IPC",
      "section": "378",
      "content_preview": "Theft.—Whoever, intending to take dishonestly..."
    }
  ],
  "confidence": "high",
  "num_sources": 3,
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### POST /upload-pdf
Upload new legal PDF documents.

**Request:** Multipart form with PDF file

**Response:**
```json
{
  "success": true,
  "message": "Successfully processed new_document.pdf",
  "filename": "new_document.pdf",
  "documents_processed": 45
}
```

### GET /get-chats
Retrieve chat history.

**Parameters:**
- `limit` (optional): Number of recent chats to return (default: 50)

**Response:**
```json
{
  "chats": [
    {
      "question": "What is Section 420 IPC?",
      "answer": "Section 420 of the Indian Penal Code deals with cheating...",
      "sources": [...],
      "confidence": "high",
      "timestamp": "2024-01-15T10:30:00Z"
    }
  ],
  "total_chats": 25,
  "returned_chats": 25
}
```

## 🧪 Testing the System

### Basic Health Check
```bash
curl http://localhost:8000/health
```

### Ask a Question
```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the punishment for murder under IPC?",
    "include_sources": true
  }'
```

### Upload a PDF
```bash
curl -X POST "http://localhost:8000/upload-pdf" \
  -F "file=@path/to/legal_document.pdf"
```

## 🔍 System Monitoring

### System Info Endpoint
```bash
curl http://localhost:8000/system-info
```

This returns:
- Vector database statistics
- RAG pipeline configuration
- Chat history count
- System initialization status

### Logs
The system provides comprehensive logging:
- PDF processing progress
- Embedding creation status
- Query processing details
- Error diagnostics

## 🚀 Deployment

### Render Deployment
The system is configured for Render deployment:

1. Connect your GitHub repository to Render
2. Set environment variables in Render dashboard
3. Deploy using the provided `render.yaml` configuration

### Environment Variables for Production
```bash
GROQ_API_KEY=your_production_groq_key
OPENAI_API_KEY=your_production_openai_key
```

## 🧩 Architecture Details

### RAG Pipeline Flow
1. **Question Input** → Preprocessing and validation
2. **Document Retrieval** → Similarity search in vector database
3. **Context Formation** → Format retrieved documents
4. **Answer Generation** → LLM generates response with legal prompt
5. **Source Citation** → Extract and format source references
6. **Response Formatting** → Structure final response with metadata

### Document Processing Pipeline
1. **PDF Extraction** → Multiple extraction methods (pdfplumber, PyMuPDF)
2. **Text Cleaning** → Legal-specific preprocessing
3. **Smart Chunking** → Context-aware document splitting
4. **Metadata Extraction** → Section and chapter identification
5. **Embedding Creation** → Vector representation generation
6. **Database Storage** → ChromaDB persistent storage

## 🔧 Customization

### Adding New Legal Documents
1. Place PDF files in the `Data/` directory
2. Restart the system or call `/initialize-system` endpoint
3. The system will automatically process and index new documents

### Modifying the Legal Prompt
Edit the `_create_legal_prompt_template()` method in `rag_pipeline.py`:

```python
template = """Your custom legal prompt template here...

Context: {context}
Question: {question}

Answer:"""
```

### Adjusting Chunking Strategy
Modify parameters in `pdf_processor.py`:

```python
self.text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,  # Increase for longer contexts
    chunk_overlap=300,  # Increase for better continuity
    separators=["\n\n", "\n", ". ", " ", ""]
)
```

## 🐛 Troubleshooting

### Common Issues

1. **System not initializing**
   - Check API keys are set correctly
   - Verify PDF files exist in Data/ directory
   - Check logs for specific error messages

2. **Poor answer quality**
   - Increase chunk_size for more context
   - Adjust temperature parameter
   - Verify document quality and OCR accuracy

3. **Slow responses**
   - Consider using OpenAI embeddings for faster retrieval
   - Reduce max_docs parameter
   - Check vector database size

### Debug Mode
Set logging level to DEBUG for detailed information:

```python
logging.basicConfig(level=logging.DEBUG)
```

## 📈 Performance Optimization

### For Large Document Collections
- Use batch processing for embeddings
- Implement document filtering by type
- Consider distributed vector storage

### For High Traffic
- Implement response caching
- Use connection pooling
- Add rate limiting

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Indian Legal System for providing comprehensive law documents
- LangChain for RAG framework
- ChromaDB for vector database
- Groq for fast LLM inference
- OpenAI for embeddings and language models

---

**Built with ❤️ for the Legal Community**

For support or questions, please open an issue or contact the development team.
