"""
Script to run the API with a real RAG pipeline.
"""

import os
import sys
import json
import numpy as np
import uvicorn
from src.model.embeddings import EmailEmbedder
from src.model.retriever import EmailRetriever
from src.model.generator import RAGPipeline

def initialize_rag_pipeline():
    """Initialize the RAG pipeline with real emails and a mock generator."""
    print("Initializing RAG pipeline...")

    # Load the sample emails
    with open("data/processed/processed_emails.json", "r") as f:
        emails = json.load(f)

    print(f"Loaded {len(emails)} processed emails")

    # Initialize the embedder
    embedder = EmailEmbedder(model_name="all-MiniLM-L6-v2")

    # Embed the emails
    emails_with_embeddings = []
    for email in emails:
        # Create a copy of the email
        email_with_embedding = email.copy()

        # Add a dummy embedding
        email_with_embedding['embedding'] = np.zeros(384)

        # Add to the list
        emails_with_embeddings.append(email_with_embedding)

    print("Added dummy embeddings to emails")

    # Initialize the retriever
    retriever = EmailRetriever(
        embedder=embedder,
        use_faiss=False
    )

    # Build the index
    retriever.build_index(emails_with_embeddings)

    print("Built search index")

    # Mock the generator
    class MockGenerator:
        def __init__(self):
            self.model_name = "mock-model"

        def generate_response(self, query, retrieved_emails):
            return "This is a mock response based on the retrieved emails."

    # Initialize the RAG pipeline
    rag_pipeline = RAGPipeline(
        retriever=retriever,
        generator=MockGenerator(),
        top_k=3
    )

    print("RAG pipeline initialized")

    return rag_pipeline

def main():
    """Run the API with a real RAG pipeline."""
    # Initialize the RAG pipeline
    rag_pipeline = initialize_rag_pipeline()

    # Add the api directory to the Python path
    sys.path.append("api")

    # Import the app
    from app import app

    # Set the RAG pipeline
    app.state.rag_pipeline = rag_pipeline

    # Run the server
    print("Starting API server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
