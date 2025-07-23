import os
import fitz  # PyMuPDF
from groq import Groq
import time
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)

# Initialize Groq client
client = None
if os.getenv("GROQ_API_KEY"):
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def extract_pdf_text(contents: bytes) -> str:
    """
    Extract PDF Text - Legacy function for backward compatibility
    """
    with open("temp.pdf", "wb") as f:
        f.write(contents)

    text = ""
    try:
        with fitz.open("temp.pdf") as doc:
            for page in doc:
                text += page.get_text()
    except Exception as e:
        logger.error(f"Error extracting PDF text: {str(e)}")
    finally:
        if os.path.exists("temp.pdf"):
            os.remove("temp.pdf")

    return text.strip()

def get_groq_response(history: List[Dict[str, str]]) -> str:
    """
    Hinglish/English Aware Response Generator - Legacy function
    """
    if not client:
        return "⚠️ Groq API key not configured. Please set GROQ_API_KEY environment variable."
    
    for attempt in range(2):
        try:
            chat = client.chat.completions.create(
                model="llama3-70b-8192",
                messages=history,
                temperature=0.3,
            )
            return chat.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"[Groq Error] Attempt {attempt + 1} failed: {e}")
            time.sleep(1)

    return "⚠️ Backend is temporarily busy. Please try again shortly."

# New utility functions for RAG system

def format_legal_response(answer: str, sources: List[Dict[str, Any]], confidence: str) -> Dict[str, Any]:
    """
    Format a legal response with proper structure
    """
    response = {
        "answer": answer,
        "confidence": confidence,
        "timestamp": datetime.now().isoformat(),
        "sources_count": len(sources)
    }
    
    # Add source references if available
    if sources:
        response["sources"] = []
        for i, source in enumerate(sources, 1):
            source_info = {
                "reference_number": i,
                "document_type": source.get("document_type", "Unknown"),
                "section": source.get("section", "N/A"),
                "content_preview": source.get("content_preview", "")[:150] + "..."
            }
            response["sources"].append(source_info)
    
    return response

def validate_legal_question(question: str) -> Dict[str, Any]:
    """
    Validate and analyze a legal question
    """
    if not question or not question.strip():
        return {
            "valid": False,
            "error": "Question cannot be empty",
            "suggestions": ["Please provide a valid legal question"]
        }
    
    question = question.strip()
    
    # Check minimum length
    if len(question) < 10:
        return {
            "valid": False,
            "error": "Question too short",
            "suggestions": ["Please provide a more detailed question"]
        }
    
    # Check for legal context keywords
    legal_keywords = [
        "ipc", "crpc", "section", "law", "legal", "crime", "punishment", 
        "bail", "arrest", "court", "judge", "police", "theft", "murder",
        "assault", "fraud", "procedure", "criminal", "penal", "code"
    ]
    
    has_legal_context = any(keyword in question.lower() for keyword in legal_keywords)
    
    analysis = {
        "valid": True,
        "length": len(question),
        "has_legal_context": has_legal_context,
        "confidence_score": "high" if has_legal_context else "medium"
    }
    
    if not has_legal_context:
        analysis["suggestions"] = [
            "Consider mentioning specific legal terms like 'IPC', 'CrPC', 'Section', etc.",
            "Be more specific about the legal area you're asking about"
        ]
    
    return analysis

def extract_section_references(text: str) -> List[str]:
    """
    Extract section references from legal text
    """
    import re
    
    # Pattern to match section references
    patterns = [
        r'Section\s+(\d+)',
        r'Sec\.\s*(\d+)',
        r'S\.\s*(\d+)',
        r'§\s*(\d+)'
    ]
    
    sections = []
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        sections.extend(matches)
    
    # Remove duplicates and sort
    return sorted(list(set(sections)))

def create_legal_context_summary(documents: List[Dict[str, Any]]) -> str:
    """
    Create a summary of legal context from retrieved documents
    """
    if not documents:
        return "No relevant legal documents found."
    
    summary_parts = []
    
    # Group by document type
    doc_types = {}
    for doc in documents:
        doc_type = doc.get("document_type", "Unknown")
        if doc_type not in doc_types:
            doc_types[doc_type] = []
        doc_types[doc_type].append(doc)
    
    # Create summary for each document type
    for doc_type, docs in doc_types.items():
        sections = []
        for doc in docs:
            if "section" in doc:
                sections.append(doc["section"])
        
        if sections:
            unique_sections = sorted(list(set(sections)))
            summary_parts.append(f"{doc_type}: Sections {', '.join(unique_sections)}")
        else:
            summary_parts.append(f"{doc_type}: {len(docs)} relevant passages")
    
    return " | ".join(summary_parts)

def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename for safe storage
    """
    import re
    
    # Remove or replace unsafe characters
    sanitized = re.sub(r'[^\w\-_\.]', '_', filename)
    
    # Ensure it's not too long
    if len(sanitized) > 100:
        name, ext = os.path.splitext(sanitized)
        sanitized = name[:95] + ext
    
    return sanitized

def check_system_health() -> Dict[str, Any]:
    """
    Check the health of various system components
    """
    health = {
        "timestamp": datetime.now().isoformat(),
        "components": {}
    }
    
    # Check API keys
    health["components"]["groq_api"] = {
        "status": "available" if os.getenv("GROQ_API_KEY") else "missing",
        "message": "Groq API key configured" if os.getenv("GROQ_API_KEY") else "GROQ_API_KEY not set"
    }
    
    health["components"]["openai_api"] = {
        "status": "available" if os.getenv("OPENAI_API_KEY") else "missing",
        "message": "OpenAI API key configured" if os.getenv("OPENAI_API_KEY") else "OPENAI_API_KEY not set"
    }
    
    # Check data directory
    data_dir = "./Data"
    if os.path.exists(data_dir):
        pdf_files = [f for f in os.listdir(data_dir) if f.lower().endswith('.pdf')]
        health["components"]["data_directory"] = {
            "status": "available",
            "message": f"Found {len(pdf_files)} PDF files",
            "files": pdf_files
        }
    else:
        health["components"]["data_directory"] = {
            "status": "missing",
            "message": "Data directory not found"
        }
    
    # Check vector database
    chroma_dir = "./chroma_db"
    if os.path.exists(chroma_dir) and os.listdir(chroma_dir):
        health["components"]["vector_database"] = {
            "status": "available",
            "message": "Vector database directory exists"
        }
    else:
        health["components"]["vector_database"] = {
            "status": "missing",
            "message": "Vector database not initialized"
        }
    
    # Overall health
    all_critical = ["groq_api", "data_directory"]
    critical_ok = all(health["components"][comp]["status"] == "available" for comp in all_critical if comp in health["components"])
    
    health["overall_status"] = "healthy" if critical_ok else "degraded"
    
    return health

def log_user_interaction(question: str, answer: str, confidence: str, sources_count: int):
    """
    Log user interaction for analytics (in production, you might want to use a proper logging service)
    """
    interaction = {
        "timestamp": datetime.now().isoformat(),
        "question_length": len(question),
        "answer_length": len(answer),
        "confidence": confidence,
        "sources_count": sources_count,
        "has_section_ref": "section" in answer.lower()
    }
    
    logger.info(f"User interaction logged: {interaction}")
    return interaction
