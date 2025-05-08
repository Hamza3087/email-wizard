"""
Retriever module for the Email Wizard Assistant.
This module handles the retrieval of relevant emails based on user queries.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import faiss
from sklearn.metrics.pairwise import cosine_similarity

from src.model.embeddings import EmailEmbedder, ChromaDBStore


class EmailRetriever:
    """
    Class for retrieving relevant emails based on user queries.
    """

    def __init__(
        self,
        embedder: EmailEmbedder,
        use_faiss: bool = True,
        index_path: Optional[str] = None
    ):
        """
        Initialize the EmailRetriever.

        Args:
            embedder: EmailEmbedder instance for embedding queries
            use_faiss: Whether to use FAISS for approximate nearest neighbor search
            index_path: Path to save/load the FAISS index
        """
        self.embedder = embedder
        self.use_faiss = use_faiss
        self.index_path = index_path
        self.index = None
        self.email_ids = []
        self.emails = []

    def build_index(self, emails_with_embeddings: List[Dict[str, Any]]) -> None:
        """
        Build a search index from emails with embeddings.

        Args:
            emails_with_embeddings: List of email dictionaries with embeddings
        """
        # Store emails for later retrieval
        self.emails = emails_with_embeddings

        # Extract email IDs and embeddings
        self.email_ids = []
        embeddings = []

        for email in emails_with_embeddings:
            if 'embedding' in email:
                self.email_ids.append(email['id'])
                embeddings.append(email['embedding'])

        # Convert to numpy array
        embeddings_array = np.array(embeddings, dtype=np.float32)

        if self.use_faiss:
            # Build FAISS index
            dimension = embeddings_array.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
            self.index.add(embeddings_array)

            # Save index if path is provided
            if self.index_path:
                faiss.write_index(self.index, self.index_path)
                print(f"Saved FAISS index to {self.index_path}")
        else:
            # Store embeddings for direct similarity computation
            self.embeddings = embeddings_array

        print(f"Built search index with {len(self.email_ids)} emails")

    def load_index(self, index_path: Optional[str] = None) -> None:
        """
        Load a FAISS index from disk.

        Args:
            index_path: Path to the FAISS index
        """
        if not self.use_faiss:
            raise ValueError("Cannot load index when use_faiss is False")

        path = index_path or self.index_path
        if not path:
            raise ValueError("No index path provided")

        self.index = faiss.read_index(path)
        print(f"Loaded FAISS index from {path}")

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve the most relevant emails for a query.

        Args:
            query: Query text
            top_k: Number of results to return
            threshold: Similarity threshold (0-1)

        Returns:
            List of relevant email dictionaries with similarity scores
        """
        # Embed the query
        query_embedding = self.embedder.embed_text(query)

        if self.use_faiss and self.index is not None:
            # Use FAISS for approximate nearest neighbor search
            query_embedding_array = np.array([query_embedding], dtype=np.float32)
            distances, indices = self.index.search(query_embedding_array, top_k)

            # Get the corresponding emails
            results = []
            for i, idx in enumerate(indices[0]):
                if idx < len(self.email_ids):
                    email_id = self.email_ids[idx]
                    distance = distances[0][i]

                    # Convert distance to similarity score (1 - normalized distance)
                    max_distance = 100.0  # Arbitrary normalization factor
                    similarity = 1.0 - min(distance / max_distance, 1.0)

                    # Apply threshold if provided
                    if threshold is not None and similarity < threshold:
                        continue

                    # Find the corresponding email
                    email = next((e for e in self.emails if e['id'] == email_id), None)
                    if email:
                        # Add similarity score
                        email_with_score = email.copy()
                        email_with_score['similarity_score'] = float(similarity)
                        results.append(email_with_score)

            return results

        else:
            # Use direct cosine similarity computation
            similarities = []

            for i, email in enumerate(self.emails):
                if 'embedding' in email:
                    email_embedding = np.array(email['embedding'])
                    similarity = cosine_similarity(
                        query_embedding.reshape(1, -1),
                        email_embedding.reshape(1, -1)
                    )[0][0]

                    # Apply threshold if provided
                    if threshold is None or similarity >= threshold:
                        similarities.append((i, similarity))

            # Sort by similarity (descending)
            similarities.sort(key=lambda x: x[1], reverse=True)

            # Get the top-k results
            results = []
            for i, similarity in similarities[:top_k]:
                email = self.emails[i].copy()
                email['similarity_score'] = float(similarity)
                results.append(email)

            return results


class ChromaDBRetriever:
    """
    Class for retrieving relevant emails using ChromaDB.
    """

    def __init__(
        self,
        chroma_store: ChromaDBStore
    ):
        """
        Initialize the ChromaDBRetriever.

        Args:
            chroma_store: ChromaDBStore instance
        """
        self.chroma_store = chroma_store

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        filter_criteria: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve the most relevant emails for a query using ChromaDB.

        Args:
            query: Query text
            top_k: Number of results to return
            filter_criteria: Filter to apply to the query

        Returns:
            List of relevant email dictionaries with similarity scores
        """
        # Query ChromaDB
        results = self.chroma_store.query(
            query_text=query,
            n_results=top_k,
            where=filter_criteria
        )

        # Format results
        formatted_results = []

        if results and 'ids' in results:
            for i in range(len(results['ids'][0])):
                email_id = results['ids'][0][i]
                document = results['documents'][0][i]
                metadata = results['metadatas'][0][i] if 'metadatas' in results else {}
                distance = results['distances'][0][i] if 'distances' in results else None

                # Convert distance to similarity score (1 - normalized distance)
                similarity = None
                if distance is not None:
                    max_distance = 2.0  # For cosine distance, max is 2
                    similarity = 1.0 - min(distance / max_distance, 1.0)

                formatted_result = {
                    'id': email_id,
                    'content': document,
                    'metadata': metadata
                }

                if similarity is not None:
                    formatted_result['similarity_score'] = similarity

                formatted_results.append(formatted_result)

        return formatted_results