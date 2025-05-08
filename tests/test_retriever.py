"""
Tests for the retriever module.
"""

import os
import sys
import unittest
from pathlib import Path
import numpy as np
from unittest.mock import MagicMock, patch

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model.retriever import EmailRetriever, ChromaDBRetriever
from src.model.embeddings import EmailEmbedder, ChromaDBStore


class TestEmailRetriever(unittest.TestCase):
    """Test cases for the EmailRetriever class."""

    def setUp(self):
        """Set up test fixtures."""
        # Mock the embedder
        self.embedder = MagicMock(spec=EmailEmbedder)
        self.embedder.embed_text.return_value = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

        # Create the retriever
        self.retriever = EmailRetriever(
            embedder=self.embedder,
            use_faiss=False
        )

        # Create test emails with embeddings
        self.test_emails = [
            {
                'id': 'email_1',
                'metadata': {'subject': 'Test Email 1'},
                'cleaned_text': 'This is test email 1',
                'embedding': [0.1, 0.2, 0.3, 0.4, 0.5]
            },
            {
                'id': 'email_2',
                'metadata': {'subject': 'Test Email 2'},
                'cleaned_text': 'This is test email 2',
                'embedding': [0.2, 0.3, 0.4, 0.5, 0.6]
            },
            {
                'id': 'email_3',
                'metadata': {'subject': 'Test Email 3'},
                'cleaned_text': 'This is test email 3',
                'embedding': [0.3, 0.4, 0.5, 0.6, 0.7]
            }
        ]

        # Build the index
        self.retriever.build_index(self.test_emails)

    def test_build_index(self):
        """Test building the search index."""
        # Check that the emails were stored
        self.assertEqual(len(self.retriever.emails), 3)
        self.assertEqual(len(self.retriever.email_ids), 3)

        # Check that the email IDs were stored correctly
        self.assertEqual(self.retriever.email_ids, ['email_1', 'email_2', 'email_3'])

    def test_retrieve(self):
        """Test retrieving similar emails."""
        # Test retrieving with top_k=2
        results = self.retriever.retrieve('test query', top_k=2)

        # Check that the correct number of results was returned
        self.assertEqual(len(results), 2)

        # Check that each result has a similarity score
        for result in results:
            self.assertIn('similarity_score', result)
            self.assertIsInstance(result['similarity_score'], float)
            self.assertTrue(0 <= result['similarity_score'] <= 1)

        # Test retrieving with threshold
        results = self.retriever.retrieve('test query', top_k=3, threshold=0.5)

        # Check that only results above the threshold were returned
        for result in results:
            self.assertGreaterEqual(result['similarity_score'], 0.5)


class TestChromaDBRetriever(unittest.TestCase):
    """Test cases for the ChromaDBRetriever class."""

    def setUp(self):
        """Set up test fixtures."""
        # Mock the ChromaDB store
        self.chroma_store = MagicMock(spec=ChromaDBStore)

        # Set up the mock query response
        self.chroma_store.query.return_value = {
            'ids': [['email_1', 'email_2']],
            'documents': [['This is test email 1', 'This is test email 2']],
            'metadatas': [[{'subject': 'Test Email 1'}, {'subject': 'Test Email 2'}]],
            'distances': [[0.1, 0.2]]
        }

        # Create the retriever
        self.retriever = ChromaDBRetriever(chroma_store=self.chroma_store)

    def test_retrieve(self):
        """Test retrieving similar emails."""
        # Test retrieving with top_k=2
        results = self.retriever.retrieve('test query', top_k=2)

        # Check that the correct number of results was returned
        self.assertEqual(len(results), 2)

        # Check that the results have the expected structure
        self.assertEqual(results[0]['id'], 'email_1')
        self.assertEqual(results[0]['content'], 'This is test email 1')
        self.assertEqual(results[0]['metadata'], {'subject': 'Test Email 1'})
        self.assertIn('similarity_score', results[0])

        # Test retrieving with filter criteria
        filter_criteria = {'topic': 'business'}
        self.retriever.retrieve('test query', top_k=2, filter_criteria=filter_criteria)

        # Check that the filter criteria was passed to the query
        self.chroma_store.query.assert_called_with(
            query_text='test query',
            n_results=2,
            where=filter_criteria
        )


if __name__ == "__main__":
    unittest.main()