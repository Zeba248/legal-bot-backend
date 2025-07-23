import os
import logging
from typing import List, Dict, Any, Optional, Tuple
from langchain_community.llms import OpenAI
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain.callbacks import get_openai_callback
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LegalRAGPipeline:
    """
    Complete RAG pipeline for legal document question answering
    """
    
    def __init__(
        self,
        vector_store,
        llm_provider: str = "groq",  # "groq" or "openai"
        model_name: str = None,
        temperature: float = 0.1,
        max_tokens: int = 1000,
        groq_api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None
    ):
        self.vector_store = vector_store
        self.llm_provider = llm_provider
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Initialize LLM
        self.llm = self._initialize_llm(
            model_name, groq_api_key, openai_api_key
        )
        
        # Initialize custom prompt template
        self.prompt_template = self._create_legal_prompt_template()
        
        # Initialize retrieval chain
        self.qa_chain = self._create_retrieval_chain()
    
    def _initialize_llm(
        self,
        model_name: Optional[str],
        groq_api_key: Optional[str],
        openai_api_key: Optional[str]
    ):
        """
        Initialize the language model based on provider
        """
        if self.llm_provider == "groq":
            if not groq_api_key:
                groq_api_key = os.getenv("GROQ_API_KEY")
            
            if not groq_api_key:
                raise ValueError("GROQ_API_KEY is required for Groq provider")
            
            model_name = model_name or "llama3-70b-8192"
            
            return ChatGroq(
                groq_api_key=groq_api_key,
                model_name=model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
        
        elif self.llm_provider == "openai":
            if not openai_api_key:
                openai_api_key = os.getenv("OPENAI_API_KEY")
            
            if not openai_api_key:
                raise ValueError("OPENAI_API_KEY is required for OpenAI provider")
            
            model_name = model_name or "gpt-3.5-turbo"
            
            return OpenAI(
                openai_api_key=openai_api_key,
                model_name=model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
        
        else:
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")
    
    def _create_legal_prompt_template(self) -> PromptTemplate:
        """
        Create the custom legal prompt template as specified
        """
        template = """You are a highly knowledgeable legal assistant trained on Indian law. Use only the provided legal context to answer. Do not guess or hallucinate and also uses gpt to answer smartly like a human being.

Context from Legal Documents:
{context}

Question: {question}

Instructions:
1. Answer based ONLY on the provided context from Indian legal documents (IPC, CrPC)
2. If the context contains relevant information, provide a comprehensive answer
3. Include specific section references when available (e.g., "As per IPC Section 420...")
4. If the context doesn't contain enough information to answer the question, clearly state that
5. Be precise, professional, and helpful like a knowledgeable legal expert
6. Respond in a natural, human-like manner while maintaining legal accuracy

Answer:"""

        return PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
    
    def _create_retrieval_chain(self):
        """
        Create the RetrievalQA chain
        """
        retriever = self.vector_store.get_retriever(
            search_kwargs={"k": 5}
        )
        
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": self.prompt_template},
            return_source_documents=True,
            verbose=True
        )
    
    def format_context(self, documents: List[Document]) -> str:
        """
        Format retrieved documents into context string
        """
        if not documents:
            return "No relevant legal documents found."
        
        context_parts = []
        
        for i, doc in enumerate(documents, 1):
            content = doc.page_content.strip()
            metadata = doc.metadata
            
            # Create a formatted context entry
            context_entry = f"Document {i}:\n"
            
            # Add source information
            if "source" in metadata:
                context_entry += f"Source: {metadata['source']}\n"
            
            if "document_type" in metadata:
                context_entry += f"Document Type: {metadata['document_type']}\n"
            
            if "section" in metadata:
                context_entry += f"Section: {metadata['section']}\n"
            
            if "chapter" in metadata:
                context_entry += f"Chapter: {metadata['chapter']}\n"
            
            context_entry += f"Content: {content}\n"
            context_entry += "-" * 50 + "\n"
            
            context_parts.append(context_entry)
        
        return "\n".join(context_parts)
    
    def extract_sources(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """
        Extract source information from documents
        """
        sources = []
        
        for doc in documents:
            metadata = doc.metadata
            source_info = {
                "content_preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
            }
            
            # Add available metadata
            for key in ["source", "document_type", "section", "chapter", "chunk_id"]:
                if key in metadata:
                    source_info[key] = metadata[key]
            
            sources.append(source_info)
        
        return sources
    
    def answer_question(
        self,
        question: str,
        include_sources: bool = True,
        max_docs: int = 5
    ) -> Dict[str, Any]:
        """
        Answer a legal question using the RAG pipeline
        """
        if not question.strip():
            return {
                "answer": "Please provide a valid question.",
                "sources": [],
                "confidence": "low",
                "error": "Empty question"
            }
        
        try:
            logger.info(f"Processing question: {question[:100]}...")
            
            # Get relevant documents
            relevant_docs = self.vector_store.similarity_search(
                query=question,
                k=max_docs
            )
            
            if not relevant_docs:
                return {
                    "answer": "I couldn't find relevant information in the legal documents to answer your question. Please try rephrasing your question or ask about topics covered in the Indian Penal Code (IPC) or Code of Criminal Procedure (CrPC).",
                    "sources": [],
                    "confidence": "low",
                    "error": "No relevant documents found"
                }
            
            # Format context
            context = self.format_context(relevant_docs)
            
            # Generate answer using the LLM
            response = self.qa_chain({
                "query": question,
                "context": context
            })
            
            answer = response.get("result", "").strip()
            source_documents = response.get("source_documents", [])
            
            # Extract sources if requested
            sources = []
            if include_sources and source_documents:
                sources = self.extract_sources(source_documents)
            
            # Determine confidence based on answer quality
            confidence = self._assess_confidence(answer, relevant_docs)
            
            return {
                "answer": answer,
                "sources": sources,
                "confidence": confidence,
                "num_sources": len(relevant_docs)
            }
        
        except Exception as e:
            logger.error(f"Error answering question: {str(e)}")
            return {
                "answer": "I apologize, but I encountered an error while processing your question. Please try again or contact support if the issue persists.",
                "sources": [],
                "confidence": "low",
                "error": str(e)
            }
    
    def _assess_confidence(self, answer: str, documents: List[Document]) -> str:
        """
        Assess confidence level of the answer
        """
        if not answer or len(answer.strip()) < 50:
            return "low"
        
        # Check for section references
        section_refs = re.findall(r'Section\s+\d+', answer, re.IGNORECASE)
        
        # Check for specific legal terms
        legal_terms = ["Indian Penal Code", "IPC", "CrPC", "Code of Criminal Procedure"]
        has_legal_terms = any(term.lower() in answer.lower() for term in legal_terms)
        
        # Check number of relevant documents
        num_docs = len(documents)
        
        if section_refs and has_legal_terms and num_docs >= 3:
            return "high"
        elif (section_refs or has_legal_terms) and num_docs >= 2:
            return "medium"
        else:
            return "low"
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """
        Get information about the RAG pipeline configuration
        """
        vector_info = self.vector_store.get_collection_info()
        
        return {
            "llm_provider": self.llm_provider,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "vector_store_info": vector_info,
            "prompt_template_type": "custom_legal"
        }

def create_legal_rag_pipeline(
    vector_store,
    llm_provider: str = "groq",
    model_name: str = None,
    temperature: float = 0.1,
    max_tokens: int = 1000,
    groq_api_key: Optional[str] = None,
    openai_api_key: Optional[str] = None
) -> LegalRAGPipeline:
    """
    Factory function to create a LegalRAGPipeline instance
    """
    return LegalRAGPipeline(
        vector_store=vector_store,
        llm_provider=llm_provider,
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        groq_api_key=groq_api_key,
        openai_api_key=openai_api_key
    )