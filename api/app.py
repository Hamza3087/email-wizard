"""
FastAPI application for the Email Wizard Assistant.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import logging

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import project modules
from src.model.embeddings import EmailEmbedder, ChromaDBStore
from src.model.retriever import ChromaDBRetriever
from src.model.generator import ResponseGenerator, RAGPipeline
from src.data.dataset import create_synthetic_dataset, load_dataset
from src.utils.helpers import setup_logging

# Set up logging
logger = setup_logging(log_file="logs/api.log")

# Create FastAPI app
app = FastAPI(
    title="Email Wizard Assistant API",
    description="API for the Email Wizard Assistant using RAG technology",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define request and response models
class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 3
    filter_criteria: Optional[Dict[str, Any]] = None


class EmailResponse(BaseModel):
    id: str
    content: str
    metadata: Dict[str, Any]
    similarity_score: Optional[float] = None


class QueryResponse(BaseModel):
    query: str
    response: str
    retrieved_emails: List[EmailResponse]


# Global variables to store models and data
embedder = None
chroma_store = None
retriever = None
generator = None
rag_pipeline = None


@app.on_event("startup")
async def startup_event():
    """Initialize models and data on startup."""
    global embedder, chroma_store, retriever, generator, rag_pipeline

    try:
        logger.info("Initializing Email Wizard Assistant...")

        # Create data directories if they don't exist
        os.makedirs("data/raw", exist_ok=True)
        os.makedirs("data/processed", exist_ok=True)
        os.makedirs("data/embeddings", exist_ok=True)
        os.makedirs("logs", exist_ok=True)

        # Check if we have a dataset, if not create one
        dataset_path = "data/processed/processed_emails.json"
        if not os.path.exists(dataset_path):
            logger.info("Creating synthetic dataset...")
            create_synthetic_dataset(
                num_emails=60,
                output_path="data/raw/synthetic_emails.json",
                preprocess=True,
                processed_path=dataset_path
            )

        # Load the dataset
        logger.info("Loading dataset...")
        emails = load_dataset(dataset_path, is_processed=True)
        logger.info(f"Loaded {len(emails)} emails")

        # Initialize embedder
        logger.info("Initializing embedder...")
        embedder = EmailEmbedder(model_name="all-MiniLM-L6-v2")

        # Initialize ChromaDB store
        logger.info("Initializing ChromaDB store...")
        chroma_store = ChromaDBStore(
            collection_name="email_embeddings",
            persist_directory="data/embeddings/chroma_db",
            embedding_function=embedder.embedding_function
        )

        # Add emails to ChromaDB if the collection is empty
        if chroma_store.get_collection_stats()["count"] == 0:
            logger.info("Adding emails to ChromaDB...")
            chroma_store.add_emails(emails)

        # Initialize retriever
        logger.info("Initializing retriever...")
        retriever = ChromaDBRetriever(chroma_store=chroma_store)

        # Initialize generator
        logger.info("Initializing generator...")
        generator = ResponseGenerator(model_name="google/flan-t5-base")

        # Initialize RAG pipeline
        logger.info("Initializing RAG pipeline...")
        rag_pipeline = RAGPipeline(
            retriever=retriever,
            generator=generator,
            top_k=3
        )

        logger.info("Email Wizard Assistant initialized successfully")

    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")
        raise


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Welcome to the Email Wizard Assistant API",
        "docs_url": "/docs"
    }


@app.post("/query_email", response_model=QueryResponse)
async def query_email(request: QueryRequest):
    """
    Query the Email Wizard Assistant.

    Args:
        request: Query request

    Returns:
        Query response
    """
    try:
        logger.info(f"Received query: {request.query}")

        # Process the query
        result = rag_pipeline.process_query(
            query=request.query,
            top_k=request.top_k,
            filter_criteria=request.filter_criteria
        )

        # Format the response
        response = QueryResponse(
            query=result["query"],
            response=result["response"],
            retrieved_emails=[
                EmailResponse(
                    id=email.get("id", ""),
                    content=email.get("content", ""),
                    metadata=email.get("metadata", {}),
                    similarity_score=email.get("similarity_score")
                )
                for email in result["retrieved_emails"]
            ]
        )

        logger.info(f"Generated response for query: {request.query}")

        return response

    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/stats")
async def get_stats():
    """Get statistics about the system."""
    try:
        stats = {
            "collection_stats": chroma_store.get_collection_stats() if chroma_store else None,
            "embedder_model": embedder.model_name if embedder else None,
            "generator_model": generator.model_name if generator else None
        }

        return stats

    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)