import os
import re
import logging
from typing import List, Dict, Any
import fitz  # PyMuPDF
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LegalPDFProcessor:
    """
    Comprehensive PDF processor for legal documents with OCR and text cleaning capabilities
    """
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text from PDF using multiple methods for best results
        """
        text = ""
        
        try:
            # Method 1: Try pdfplumber first (better for structured PDFs)
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            
            # If pdfplumber didn't extract much text, try PyMuPDF
            if len(text.strip()) < 100:
                logger.info("Pdfplumber extracted minimal text, trying PyMuPDF...")
                with fitz.open(pdf_path) as doc:
                    for page in doc:
                        text += page.get_text() + "\n"
            
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {str(e)}")
            # Fallback to PyMuPDF
            try:
                with fitz.open(pdf_path) as doc:
                    for page in doc:
                        text += page.get_text() + "\n"
            except Exception as e2:
                logger.error(f"Fallback extraction also failed: {str(e2)}")
                return ""
        
        return text.strip()
    
    def clean_legal_text(self, text: str) -> str:
        """
        Clean and preprocess legal text for better chunking and retrieval
        """
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Fix common OCR issues
        text = re.sub(r'(\w)\s+(\w)', r'\1\2', text)  # Remove spaces within words
        text = re.sub(r'([a-z])\s+([A-Z])', r'\1 \2', text)  # Keep proper word boundaries
        
        # Normalize section references
        text = re.sub(r'Section\s+(\d+)', r'Section \1', text, flags=re.IGNORECASE)
        text = re.sub(r'Sec\.\s*(\d+)', r'Section \1', text, flags=re.IGNORECASE)
        
        # Normalize chapter references
        text = re.sub(r'Chapter\s+(\d+)', r'Chapter \1', text, flags=re.IGNORECASE)
        text = re.sub(r'Ch\.\s*(\d+)', r'Chapter \1', text, flags=re.IGNORECASE)
        
        # Fix common legal terms
        text = re.sub(r'\bIPC\b', 'Indian Penal Code', text)
        text = re.sub(r'\bCrPC\b', 'Code of Criminal Procedure', text)
        
        # Remove page numbers and headers/footers (common patterns)
        text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
        text = re.sub(r'\n\s*Page\s+\d+.*?\n', '\n', text, flags=re.IGNORECASE)
        
        return text.strip()
    
    def chunk_documents(self, text: str, source: str) -> List[Document]:
        """
        Split text into chunks and create Document objects with metadata
        """
        if not text:
            return []
        
        # Clean the text first
        cleaned_text = self.clean_legal_text(text)
        
        # Split into chunks
        chunks = self.text_splitter.split_text(cleaned_text)
        
        documents = []
        for i, chunk in enumerate(chunks):
            # Extract section information if available
            section_match = re.search(r'Section\s+(\d+)', chunk, re.IGNORECASE)
            chapter_match = re.search(r'Chapter\s+(\d+)', chunk, re.IGNORECASE)
            
            metadata = {
                "source": source,
                "chunk_id": i,
                "total_chunks": len(chunks),
                "chunk_size": len(chunk)
            }
            
            if section_match:
                metadata["section"] = section_match.group(1)
            if chapter_match:
                metadata["chapter"] = chapter_match.group(1)
            
            # Determine document type
            if "ipc" in source.lower() or "penal code" in chunk.lower():
                metadata["document_type"] = "IPC"
            elif "crpc" in source.lower() or "criminal procedure" in chunk.lower():
                metadata["document_type"] = "CrPC"
            else:
                metadata["document_type"] = "Legal Document"
            
            documents.append(Document(
                page_content=chunk,
                metadata=metadata
            ))
        
        return documents
    
    def process_pdf_file(self, pdf_path: str) -> List[Document]:
        """
        Complete pipeline to process a PDF file into Document chunks
        """
        logger.info(f"Processing PDF: {pdf_path}")
        
        if not os.path.exists(pdf_path):
            logger.error(f"PDF file not found: {pdf_path}")
            return []
        
        # Extract text
        text = self.extract_text_from_pdf(pdf_path)
        
        if not text:
            logger.error(f"No text extracted from {pdf_path}")
            return []
        
        logger.info(f"Extracted {len(text)} characters from {pdf_path}")
        
        # Create document chunks
        documents = self.chunk_documents(text, os.path.basename(pdf_path))
        
        logger.info(f"Created {len(documents)} document chunks from {pdf_path}")
        
        return documents
    
    def process_multiple_pdfs(self, pdf_directory: str) -> List[Document]:
        """
        Process multiple PDF files from a directory
        """
        all_documents = []
        
        if not os.path.exists(pdf_directory):
            logger.error(f"Directory not found: {pdf_directory}")
            return []
        
        pdf_files = [f for f in os.listdir(pdf_directory) if f.lower().endswith('.pdf')]
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {pdf_directory}")
            return []
        
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        for pdf_file in pdf_files:
            pdf_path = os.path.join(pdf_directory, pdf_file)
            documents = self.process_pdf_file(pdf_path)
            all_documents.extend(documents)
        
        logger.info(f"Total documents processed: {len(all_documents)}")
        return all_documents

def get_legal_pdf_processor(chunk_size: int = 1000, chunk_overlap: int = 200) -> LegalPDFProcessor:
    """
    Factory function to create a LegalPDFProcessor instance
    """
    return LegalPDFProcessor(chunk_size=chunk_size, chunk_overlap=chunk_overlap)