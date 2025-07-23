import os
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

# Import our custom modules
from pdf_processor import get_legal_pdf_processor
from vector_store import create_legal_vector_store
from rag_pipeline import create_legal_rag_pipeline
from initialize_data import LegalDataInitializer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Legal AI Chatbot API",
    description="RAG-based Legal AI Chatbot for Indian Law (IPC & CrPC)",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class QuestionRequest(BaseModel):
    question: str
    include_sources: bool = True
    max_docs: int = 5

class QuestionResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    confidence: str
    num_sources: int
    timestamp: str

class ChatMessage(BaseModel):
    question: str
    answer: str
    sources: List[Dict[str, Any]]
    confidence: str
    timestamp: str

class UploadResponse(BaseModel):
    success: bool
    message: str
    filename: Optional[str] = None
    documents_processed: Optional[int] = None

# Global variables for the RAG system
vector_store = None
rag_pipeline = None
chat_history: List[ChatMessage] = []
system_initialized = False

def initialize_system():
    """
    Initialize the RAG system on startup
    """
    global vector_store, rag_pipeline, system_initialized
    
    try:
        logger.info("🚀 Initializing Legal AI Chatbot system...")
        
        # Check if vector database already exists
        persist_directory = "./chroma_db"
        if os.path.exists(persist_directory) and os.listdir(persist_directory):
            logger.info("📚 Existing vector database found, loading...")
            
            # Initialize vector store
            vector_store = create_legal_vector_store(
                collection_name="legal_documents",
                persist_directory=persist_directory,
                embedding_model="huggingface"  # Use HuggingFace for reliability
            )
            
            # Check if collection has documents
            collection_info = vector_store.get_collection_info()
            if collection_info.get("document_count", 0) > 0:
                logger.info(f"✅ Loaded existing collection with {collection_info['document_count']} documents")
                
                # Initialize RAG pipeline with fallback
                if os.getenv("GROQ_API_KEY"):
                    rag_pipeline = create_legal_rag_pipeline(
                        vector_store=vector_store,
                        llm_provider="groq",
                        model_name="llama3-70b-8192",
                        temperature=0.1
                    )
                else:
                    logger.info("No Groq API key found, RAG pipeline will be initialized on first query")
                    rag_pipeline = None
                
                system_initialized = True
                logger.info("🎉 System initialized successfully with existing data!")
                return True
        
        # If no existing data, initialize from PDFs
        logger.info("📄 No existing data found, initializing from PDFs...")
        initializer = LegalDataInitializer(embedding_model="huggingface")
        
        result = initializer.run_full_initialization()
        
        if result["success"]:
            vector_store = initializer.vector_store
            
            # Initialize RAG pipeline with fallback
            if os.getenv("GROQ_API_KEY"):
                rag_pipeline = create_legal_rag_pipeline(
                    vector_store=vector_store,
                    llm_provider="groq",
                    model_name="llama3-70b-8192",
                    temperature=0.1
                )
            else:
                logger.info("No Groq API key found, RAG pipeline will be initialized on first query")
                rag_pipeline = None
            
            system_initialized = True
            logger.info("🎉 System initialized successfully from PDFs!")
            return True
        else:
            logger.error(f"❌ Failed to initialize system: {result['error']}")
            return False
            
    except Exception as e:
        logger.error(f"💥 System initialization failed: {str(e)}")
        return False

@app.on_event("startup")
async def startup_event():
    """
    Initialize the system when the API starts
    """
    initialize_system()

@app.get("/")
async def root():
    """
    Root endpoint with system status
    """
    global system_initialized, vector_store
    
    status = {
        "status": "Legal AI Chatbot API is running",
        "version": "1.0.0",
        "system_initialized": system_initialized,
        "timestamp": datetime.now().isoformat()
    }
    
    if system_initialized and vector_store:
        collection_info = vector_store.get_collection_info()
        status["collection_info"] = collection_info
    
    return status

@app.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """
    Main endpoint to ask legal questions using RAG
    """
    global rag_pipeline, system_initialized
    
    if not system_initialized:
        raise HTTPException(
            status_code=503,
            detail="System not initialized. Please wait for initialization to complete or check system logs."
        )
    
    # Initialize RAG pipeline on demand if not available
    if not rag_pipeline:
        logger.info("Initializing RAG pipeline on demand...")
        try:
            rag_pipeline = create_legal_rag_pipeline(
                vector_store=vector_store,
                llm_provider="groq" if os.getenv("GROQ_API_KEY") else "openai",
                model_name="llama3-70b-8192" if os.getenv("GROQ_API_KEY") else "gpt-3.5-turbo",
                temperature=0.1
            )
        except Exception as e:
            logger.error(f"Failed to initialize RAG pipeline: {str(e)}")
            raise HTTPException(
                status_code=503,
                detail="Failed to initialize language model. Please check API keys."
            )
    
    if not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty"
        )
    
    try:
        logger.info(f"📝 Processing question: {request.question[:100]}...")
        
        # Get answer from RAG pipeline
        result = rag_pipeline.answer_question(
            question=request.question,
            include_sources=request.include_sources,
            max_docs=request.max_docs
        )
        
        # Create response
        response = QuestionResponse(
            answer=result["answer"],
            sources=result.get("sources", []),
            confidence=result.get("confidence", "low"),
            num_sources=result.get("num_sources", 0),
            timestamp=datetime.now().isoformat()
        )
        
        # Add to chat history
        chat_message = ChatMessage(
            question=request.question,
            answer=result["answer"],
            sources=result.get("sources", []),
            confidence=result.get("confidence", "low"),
            timestamp=response.timestamp
        )
        chat_history.append(chat_message)
        
        # Keep only last 100 messages
        if len(chat_history) > 100:
            chat_history.pop(0)
        
        logger.info(f"✅ Question answered with {result.get('confidence', 'low')} confidence")
        
        return response
        
    except Exception as e:
        logger.error(f"❌ Error processing question: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing question: {str(e)}"
        )

@app.post("/upload-pdf", response_model=UploadResponse)
async def upload_pdf(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """
    Upload and process new legal PDF documents
    """
    global vector_store, system_initialized
    
    if not system_initialized or not vector_store:
        raise HTTPException(
            status_code=503,
            detail="System not initialized. Cannot process uploads."
        )
    
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported"
        )
    
    try:
        logger.info(f"📤 Processing uploaded PDF: {file.filename}")
        
        # Save uploaded file temporarily
        temp_file_path = f"./temp_{file.filename}"
        
        with open(temp_file_path, "wb") as temp_file:
            content = await file.read()
            temp_file.write(content)
        
        # Process the PDF
        pdf_processor = get_legal_pdf_processor()
        documents = pdf_processor.process_pdf_file(temp_file_path)
        
        if not documents:
            # Clean up temp file
            os.remove(temp_file_path)
            raise HTTPException(
                status_code=400,
                detail="Could not extract text from the PDF file"
            )
        
        # Add documents to vector store
        success = vector_store.add_documents(documents)
        
        # Clean up temp file
        os.remove(temp_file_path)
        
        if success:
            logger.info(f"✅ Successfully processed {len(documents)} documents from {file.filename}")
            
            return UploadResponse(
                success=True,
                message=f"Successfully processed {file.filename}",
                filename=file.filename,
                documents_processed=len(documents)
            )
        else:
            raise HTTPException(
                status_code=500,
                detail="Failed to add documents to vector store"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        # Clean up temp file if it exists
        temp_file_path = f"./temp_{file.filename}"
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        
        logger.error(f"❌ Error processing uploaded PDF: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing PDF: {str(e)}"
        )

@app.get("/get-chats")
async def get_chats(limit: int = 50):
    """
    Get chat history for memory/follow-up queries
    """
    global chat_history
    
    # Return the most recent chats (up to limit)
    recent_chats = chat_history[-limit:] if len(chat_history) > limit else chat_history
    
    return {
        "chats": recent_chats,
        "total_chats": len(chat_history),
        "returned_chats": len(recent_chats)
    }

@app.post("/search", response_model=dict)
async def search_documents(request: QuestionRequest):
    """
    Search legal documents without LLM generation (for testing retrieval)
    """
    global vector_store, system_initialized
    
    if not system_initialized or not vector_store:
        raise HTTPException(
            status_code=503,
            detail="System not initialized."
        )
    
    if not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty"
        )
    
    try:
        logger.info(f"🔍 Searching documents for: {request.question[:100]}...")
        
        # Search similar documents
        docs = vector_store.similarity_search(
            request.question, 
            k=request.max_docs or 5
        )
        
        results = []
        for doc in docs:
            results.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "content_preview": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
            })
        
        return {
            "query": request.question,
            "num_results": len(results),
            "results": results,
            "timestamp": datetime.now().isoformat(),
            "note": "This is document retrieval only. For full RAG answers, use /ask endpoint with API keys."
        }
        
    except Exception as e:
        logger.error(f"Error searching documents: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error searching documents: {str(e)}"
        )

@app.get("/system-info")
async def get_system_info():
    """
    Get detailed system information
    """
    global system_initialized, vector_store, rag_pipeline
    
    info = {
        "system_initialized": system_initialized,
        "timestamp": datetime.now().isoformat(),
        "chat_history_count": len(chat_history),
        "api_keys": {
            "groq": bool(os.getenv("GROQ_API_KEY")),
            "openai": bool(os.getenv("OPENAI_API_KEY"))
        }
    }
    
    if system_initialized:
        if vector_store:
            info["vector_store"] = vector_store.get_collection_info()
        
        if rag_pipeline:
            info["rag_pipeline"] = rag_pipeline.get_pipeline_info()
    
    return info

@app.post("/initialize-system")
async def reinitialize_system():
    """
    Manually reinitialize the system (useful for development/debugging)
    """
    try:
        logger.info("🔄 Manual system reinitialization requested...")
        success = initialize_system()
        
        if success:
            return {
                "success": True,
                "message": "System reinitialized successfully",
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(
                status_code=500,
                detail="System reinitialization failed"
            )
            
    except Exception as e:
        logger.error(f"❌ Manual reinitialization failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Reinitialization failed: {str(e)}"
        )

@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    return {
        "status": "healthy",
        "system_initialized": system_initialized,
        "timestamp": datetime.now().isoformat()
    }

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={
            "error": "Endpoint not found",
            "message": "The requested endpoint does not exist",
            "available_endpoints": [
                "/",
                "/ask",
                "/upload-pdf",
                "/get-chats",
                "/system-info",
                "/health"
            ]
        }
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred. Please check the server logs.",
            "timestamp": datetime.now().isoformat()
        }
    )

if __name__ == "__main__":
    # For local development
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
