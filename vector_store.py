import os
import logging
from typing import List, Dict, Any, Optional, Tuple
import chromadb
from chromadb.config import Settings
from langchain_community.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LegalVectorStore:
    """
    Comprehensive vector store manager for legal documents using ChromaDB
    """
    
    def __init__(
        self,
        collection_name: str = "legal_documents",
        persist_directory: str = "./chroma_db",
        embedding_model: str = "openai",  # or "huggingface"
        openai_api_key: Optional[str] = None
    ):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embedding_model = embedding_model
        
        # Create persist directory if it doesn't exist
        os.makedirs(persist_directory, exist_ok=True)
        
        # Initialize embeddings
        self.embeddings = self._initialize_embeddings(openai_api_key)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Initialize vector store
        self.vector_store = None
        self._initialize_vector_store()
    
    def _initialize_embeddings(self, openai_api_key: Optional[str]):
        """
        Initialize embedding model based on configuration
        """
        if self.embedding_model == "openai":
            if not openai_api_key:
                openai_api_key = os.getenv("OPENAI_API_KEY")
            
            if not openai_api_key:
                logger.warning("OpenAI API key not found, falling back to HuggingFace embeddings")
                return HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                    model_kwargs={'device': 'cpu'}
                )
            
            return OpenAIEmbeddings(
                model="text-embedding-3-small",
                openai_api_key=openai_api_key
            )
        
        else:  # huggingface
            return HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}
            )
    
    def _initialize_vector_store(self):
        """
        Initialize the Chroma vector store
        """
        try:
            self.vector_store = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory,
                client=self.client
            )
            logger.info(f"Vector store initialized with collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {str(e)}")
            raise
    
    def add_documents(self, documents: List[Document]) -> bool:
        """
        Add documents to the vector store
        """
        if not documents:
            logger.warning("No documents provided to add")
            return False
        
        try:
            # Filter out empty documents
            valid_documents = [doc for doc in documents if doc.page_content.strip()]
            
            if not valid_documents:
                logger.warning("No valid documents to add after filtering")
                return False
            
            logger.info(f"Adding {len(valid_documents)} documents to vector store")
            
            # Add documents to vector store
            self.vector_store.add_documents(valid_documents)
            
            # Persist the changes
            self.vector_store.persist()
            
            logger.info(f"Successfully added {len(valid_documents)} documents")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add documents: {str(e)}")
            return False
    
    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Perform similarity search on the vector store
        """
        if not query.strip():
            logger.warning("Empty query provided")
            return []
        
        try:
            logger.info(f"Performing similarity search for: '{query[:50]}...' with k={k}")
            
            # Perform similarity search
            if filter_dict:
                results = self.vector_store.similarity_search(
                    query=query,
                    k=k,
                    filter=filter_dict
                )
            else:
                results = self.vector_store.similarity_search(
                    query=query,
                    k=k
                )
            
            logger.info(f"Found {len(results)} similar documents")
            return results
            
        except Exception as e:
            logger.error(f"Similarity search failed: {str(e)}")
            return []
    
    def similarity_search_with_scores(
        self,
        query: str,
        k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Document, float]]:
        """
        Perform similarity search with relevance scores
        """
        if not query.strip():
            logger.warning("Empty query provided")
            return []
        
        try:
            logger.info(f"Performing similarity search with scores for: '{query[:50]}...'")
            
            # Perform similarity search with scores
            if filter_dict:
                results = self.vector_store.similarity_search_with_score(
                    query=query,
                    k=k,
                    filter=filter_dict
                )
            else:
                results = self.vector_store.similarity_search_with_score(
                    query=query,
                    k=k
                )
            
            logger.info(f"Found {len(results)} similar documents with scores")
            return results
            
        except Exception as e:
            logger.error(f"Similarity search with scores failed: {str(e)}")
            return []
    
    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about the current collection
        """
        try:
            collection = self.client.get_collection(self.collection_name)
            count = collection.count()
            
            return {
                "collection_name": self.collection_name,
                "document_count": count,
                "embedding_model": self.embedding_model,
                "persist_directory": self.persist_directory
            }
        except Exception as e:
            logger.error(f"Failed to get collection info: {str(e)}")
            return {
                "collection_name": self.collection_name,
                "document_count": 0,
                "embedding_model": self.embedding_model,
                "persist_directory": self.persist_directory,
                "error": str(e)
            }
    
    def delete_collection(self) -> bool:
        """
        Delete the current collection
        """
        try:
            self.client.delete_collection(self.collection_name)
            logger.info(f"Deleted collection: {self.collection_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete collection: {str(e)}")
            return False
    
    def reset_collection(self) -> bool:
        """
        Reset the collection (delete and recreate)
        """
        try:
            # Delete existing collection
            try:
                self.client.delete_collection(self.collection_name)
                logger.info(f"Deleted existing collection: {self.collection_name}")
            except:
                pass  # Collection might not exist
            
            # Reinitialize vector store
            self._initialize_vector_store()
            logger.info(f"Reset collection: {self.collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to reset collection: {str(e)}")
            return False
    
    def search_by_document_type(
        self,
        query: str,
        document_type: str,
        k: int = 5
    ) -> List[Document]:
        """
        Search for documents of a specific type (IPC, CrPC, etc.)
        """
        filter_dict = {"document_type": document_type}
        return self.similarity_search(query, k, filter_dict)
    
    def search_by_section(
        self,
        query: str,
        section: str,
        k: int = 5
    ) -> List[Document]:
        """
        Search for documents from a specific section
        """
        filter_dict = {"section": section}
        return self.similarity_search(query, k, filter_dict)
    
    def get_retriever(self, search_kwargs: Optional[Dict[str, Any]] = None):
        """
        Get a LangChain retriever for the vector store
        """
        if not search_kwargs:
            search_kwargs = {"k": 5}
        
        return self.vector_store.as_retriever(search_kwargs=search_kwargs)

def create_legal_vector_store(
    collection_name: str = "legal_documents",
    persist_directory: str = "./chroma_db",
    embedding_model: str = "openai",
    openai_api_key: Optional[str] = None
) -> LegalVectorStore:
    """
    Factory function to create a LegalVectorStore instance
    """
    return LegalVectorStore(
        collection_name=collection_name,
        persist_directory=persist_directory,
        embedding_model=embedding_model,
        openai_api_key=openai_api_key
    )