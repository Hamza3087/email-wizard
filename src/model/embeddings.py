"""
Embeddings module for the Email Wizard Assistant.
This module handles the embedding of emails using pre-trained models.
"""

import os
import json
import numpy as np
from typing import List, Dict, Union, Optional, Any
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions


class EmailEmbedder:
    """
    Class for embedding emails using pre-trained models.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize the EmailEmbedder.

        Args:
            model_name: Name of the pre-trained model to use
            device: Device to use for inference (cpu or cuda)
            cache_dir: Directory to cache the model
        """
        self.model_name = model_name
        self.device = device
        self.cache_dir = cache_dir

        # Load the model
        self.model = SentenceTransformer(model_name, device=device, cache_folder=cache_dir)

        # Set up embedding function for ChromaDB
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=model_name,
            device=device
        )

    def embed_text(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Embed text using the pre-trained model.

        Args:
            text: Text to embed (string or list of strings)

        Returns:
            Numpy array of embeddings
        """
        return self.model.encode(text, convert_to_numpy=True)

    def embed_email(self, email: Dict[str, Any]) -> Dict[str, Any]:
        """
        Embed a single email.

        Args:
            email: Dictionary containing email data

        Returns:
            Dictionary with email data and embeddings
        """
        # Get the cleaned text or chunks
        if 'chunks' in email and email['chunks']:
            # Embed each chunk separately
            chunks = email['chunks']
            chunk_embeddings = self.embed_text(chunks)

            # Add embeddings to the email dictionary
            email_with_embeddings = email.copy()
            email_with_embeddings['chunk_embeddings'] = chunk_embeddings.tolist()

            # Also compute a single embedding for the entire email
            if 'cleaned_text' in email:
                full_embedding = self.embed_text(email['cleaned_text'])
                email_with_embeddings['embedding'] = full_embedding.tolist()

            return email_with_embeddings

        elif 'cleaned_text' in email:
            # Embed the entire email
            embedding = self.embed_text(email['cleaned_text'])

            # Add embedding to the email dictionary
            email_with_embeddings = email.copy()
            email_with_embeddings['embedding'] = embedding.tolist()

            return email_with_embeddings

        else:
            # If no cleaned text or chunks, use the raw text
            raw_text = email.get('content', '')
            embedding = self.embed_text(raw_text)

            # Add embedding to the email dictionary
            email_with_embeddings = email.copy()
            email_with_embeddings['embedding'] = embedding.tolist()

            return email_with_embeddings

    def embed_emails(self, emails: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Embed a list of emails.

        Args:
            emails: List of email dictionaries

        Returns:
            List of email dictionaries with embeddings
        """
        return [self.embed_email(email) for email in emails]

    def save_embeddings(self, emails_with_embeddings: List[Dict[str, Any]], output_path: str) -> None:
        """
        Save embeddings to a file.

        Args:
            emails_with_embeddings: List of email dictionaries with embeddings
            output_path: Path to save the embeddings
        """
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save embeddings
        with open(output_path, 'w') as f:
            json.dump(emails_with_embeddings, f)

        print(f"Saved embeddings for {len(emails_with_embeddings)} emails to {output_path}")

    def load_embeddings(self, input_path: str) -> List[Dict[str, Any]]:
        """
        Load embeddings from a file.

        Args:
            input_path: Path to the embeddings file

        Returns:
            List of email dictionaries with embeddings
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Embeddings file not found: {input_path}")

        with open(input_path, 'r') as f:
            emails_with_embeddings = json.load(f)

        return emails_with_embeddings


class ChromaDBStore:
    """
    Class for storing and retrieving email embeddings using ChromaDB.
    """

    def __init__(
        self,
        collection_name: str = "email_embeddings",
        persist_directory: Optional[str] = "data/embeddings/chroma_db",
        embedding_function: Optional[Any] = None
    ):
        """
        Initialize the ChromaDBStore.

        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory to persist the ChromaDB
            embedding_function: Embedding function to use
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory

        # Create persist directory if it doesn't exist
        if persist_directory:
            os.makedirs(persist_directory, exist_ok=True)

        # Initialize ChromaDB client
        if persist_directory:
            self.client = chromadb.PersistentClient(path=persist_directory)
        else:
            self.client = chromadb.Client()

        # Set up embedding function
        self.embedding_function = embedding_function

        # Create or get collection
        try:
            self.collection = self.client.get_collection(
                name=collection_name,
                embedding_function=embedding_function
            )
            print(f"Using existing collection: {collection_name}")
        except:
            self.collection = self.client.create_collection(
                name=collection_name,
                embedding_function=embedding_function
            )
            print(f"Created new collection: {collection_name}")

    def add_emails(self, emails: List[Dict[str, Any]]) -> None:
        """
        Add emails to the ChromaDB collection.

        Args:
            emails: List of email dictionaries
        """
        # Prepare data for ChromaDB
        ids = []
        documents = []
        metadatas = []

        for email in emails:
            email_id = email.get('id', '')

            # Use cleaned text if available, otherwise use raw text
            if 'cleaned_text' in email:
                document = email['cleaned_text']
            else:
                document = email.get('content', '')

            # Extract metadata
            metadata = email.get('metadata', {})

            # Add to lists
            ids.append(email_id)
            documents.append(document)
            metadatas.append(metadata)

        # Add to collection
        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )

        print(f"Added {len(emails)} emails to ChromaDB collection: {self.collection_name}")

    def query(
        self,
        query_text: str,
        n_results: int = 3,
        where: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Query the ChromaDB collection.

        Args:
            query_text: Query text
            n_results: Number of results to return
            where: Filter to apply to the query

        Returns:
            Dictionary containing query results
        """
        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where=where
        )

        return results

    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the ChromaDB collection.

        Returns:
            Dictionary containing collection statistics
        """
        count = self.collection.count()

        return {
            "collection_name": self.collection_name,
            "count": count
        }