import os
import logging
from typing import List, Dict, Any
from pdf_processor import get_legal_pdf_processor
from vector_store import create_legal_vector_store
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LegalDataInitializer:
    """
    Initialize the legal document database by processing PDFs and creating embeddings
    """
    
    def __init__(
        self,
        data_directory: str = "./Data",
        persist_directory: str = "./chroma_db",
        embedding_model: str = "huggingface",  # Default to huggingface for reliability
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        self.data_directory = data_directory
        self.persist_directory = persist_directory
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize processors
        self.pdf_processor = get_legal_pdf_processor(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        self.vector_store = None
    
    def check_environment(self) -> Dict[str, Any]:
        """
        Check if all required files and environment variables are available
        """
        status = {
            "data_directory_exists": os.path.exists(self.data_directory),
            "pdf_files": [],
            "api_keys": {},
            "missing_requirements": []
        }
        
        # Check for PDF files
        if status["data_directory_exists"]:
            pdf_files = [f for f in os.listdir(self.data_directory) if f.lower().endswith('.pdf')]
            status["pdf_files"] = pdf_files
            
            # Look for specific legal documents
            ipc_files = [f for f in pdf_files if 'ipc' in f.lower()]
            crpc_files = [f for f in pdf_files if 'crpc' in f.lower()]
            
            status["ipc_files"] = ipc_files
            status["crpc_files"] = crpc_files
        
        # Check API keys
        status["api_keys"]["groq"] = bool(os.getenv("GROQ_API_KEY"))
        status["api_keys"]["openai"] = bool(os.getenv("OPENAI_API_KEY"))
        
        # Check requirements
        if not status["data_directory_exists"]:
            status["missing_requirements"].append("Data directory not found")
        
        if not status["pdf_files"]:
            status["missing_requirements"].append("No PDF files found in Data directory")
        
        # API keys are optional when using local embeddings
        if not status["api_keys"]["groq"] and not status["api_keys"]["openai"]:
            logger.info("No API keys found - will use local embeddings and models")
        
        return status
    
    def initialize_vector_store(self) -> bool:
        """
        Initialize the vector store
        """
        try:
            logger.info(f"Initializing vector store with {self.embedding_model} embeddings...")
            
            self.vector_store = create_legal_vector_store(
                collection_name="legal_documents",
                persist_directory=self.persist_directory,
                embedding_model=self.embedding_model,
                openai_api_key=os.getenv("OPENAI_API_KEY")
            )
            
            logger.info("Vector store initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {str(e)}")
            return False
    
    def process_legal_pdfs(self) -> List[Any]:
        """
        Process all legal PDF files and extract documents
        """
        logger.info("Starting PDF processing...")
        
        if not os.path.exists(self.data_directory):
            logger.error(f"Data directory not found: {self.data_directory}")
            return []
        
        # Process all PDFs in the directory
        all_documents = self.pdf_processor.process_multiple_pdfs(self.data_directory)
        
        if not all_documents:
            logger.warning("No documents were processed from PDFs")
            return []
        
        logger.info(f"Successfully processed {len(all_documents)} document chunks")
        
        # Log statistics by document type
        doc_stats = {}
        for doc in all_documents:
            doc_type = doc.metadata.get("document_type", "Unknown")
            doc_stats[doc_type] = doc_stats.get(doc_type, 0) + 1
        
        logger.info("Document statistics by type:")
        for doc_type, count in doc_stats.items():
            logger.info(f"  {doc_type}: {count} chunks")
        
        return all_documents
    
    def create_embeddings_and_store(self, documents: List[Any]) -> bool:
        """
        Create embeddings for documents and store in vector database
        """
        if not documents:
            logger.error("No documents provided for embedding creation")
            return False
        
        if not self.vector_store:
            logger.error("Vector store not initialized")
            return False
        
        try:
            logger.info(f"Creating embeddings for {len(documents)} documents...")
            start_time = time.time()
            
            # Reset collection to start fresh
            self.vector_store.reset_collection()
            
            # Add documents in batches to avoid memory issues
            batch_size = 50
            total_batches = (len(documents) + batch_size - 1) // batch_size
            
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                batch_num = (i // batch_size) + 1
                
                logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} documents)")
                
                success = self.vector_store.add_documents(batch)
                if not success:
                    logger.error(f"Failed to add batch {batch_num}")
                    return False
                
                # Small delay between batches
                time.sleep(0.5)
            
            end_time = time.time()
            logger.info(f"Successfully created and stored embeddings in {end_time - start_time:.2f} seconds")
            
            # Get final collection info
            collection_info = self.vector_store.get_collection_info()
            logger.info(f"Final collection info: {collection_info}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to create embeddings: {str(e)}")
            return False
    
    def test_retrieval(self, test_queries: List[str] = None) -> Dict[str, Any]:
        """
        Test the retrieval system with sample queries
        """
        if not test_queries:
            test_queries = [
                "What is theft according to IPC?",
                "What are the procedures for arrest in CrPC?",
                "What is Section 420 of IPC?",
                "What are the bail provisions?",
                "What is the punishment for murder?"
            ]
        
        if not self.vector_store:
            logger.error("Vector store not initialized for testing")
            return {"error": "Vector store not initialized"}
        
        results = {}
        
        for query in test_queries:
            logger.info(f"Testing query: {query}")
            
            try:
                # Test similarity search
                docs = self.vector_store.similarity_search(query, k=3)
                
                results[query] = {
                    "num_results": len(docs),
                    "results": []
                }
                
                for doc in docs:
                    results[query]["results"].append({
                        "content_preview": doc.page_content[:200] + "...",
                        "metadata": doc.metadata
                    })
                
            except Exception as e:
                logger.error(f"Error testing query '{query}': {str(e)}")
                results[query] = {"error": str(e)}
        
        return results
    
    def run_full_initialization(self) -> Dict[str, Any]:
        """
        Run the complete initialization process
        """
        logger.info("Starting full legal document initialization...")
        
        # Check environment
        env_status = self.check_environment()
        logger.info(f"Environment check: {env_status}")
        
        if env_status["missing_requirements"]:
            logger.error("Missing requirements:")
            for req in env_status["missing_requirements"]:
                logger.error(f"  - {req}")
            return {"success": False, "error": "Missing requirements", "details": env_status}
        
        # Initialize vector store
        if not self.initialize_vector_store():
            return {"success": False, "error": "Failed to initialize vector store"}
        
        # Process PDFs
        documents = self.process_legal_pdfs()
        if not documents:
            return {"success": False, "error": "No documents processed from PDFs"}
        
        # Create embeddings and store
        if not self.create_embeddings_and_store(documents):
            return {"success": False, "error": "Failed to create embeddings"}
        
        # Test retrieval
        test_results = self.test_retrieval()
        
        logger.info("Full initialization completed successfully!")
        
        return {
            "success": True,
            "documents_processed": len(documents),
            "collection_info": self.vector_store.get_collection_info(),
            "test_results": test_results
        }

def main():
    """
    Main function to run the initialization
    """
    print("🏛️ Legal AI Chatbot - Data Initialization")
    print("=" * 50)
    
    # Initialize with default settings
    initializer = LegalDataInitializer(
        embedding_model="huggingface"  # Use HuggingFace by default for reliability
    )
    
    # Run initialization
    result = initializer.run_full_initialization()
    
    if result["success"]:
        print("\n✅ Initialization completed successfully!")
        print(f"📄 Documents processed: {result['documents_processed']}")
        print(f"💾 Collection info: {result['collection_info']}")
        print("\n🔍 Test results:")
        for query, test_result in result.get("test_results", {}).items():
            if "error" not in test_result:
                print(f"  ✅ '{query}': {test_result['num_results']} results found")
            else:
                print(f"  ❌ '{query}': {test_result['error']}")
    else:
        print(f"\n❌ Initialization failed: {result['error']}")
        if "details" in result:
            print(f"Details: {result['details']}")

if __name__ == "__main__":
    main()