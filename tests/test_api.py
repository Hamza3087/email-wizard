"""
Tests for the API module.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import the app
sys.path.append(str(Path(__file__).resolve().parent.parent / "api"))
from api.app import app, QueryRequest


class TestAPI(unittest.TestCase):
    """Test cases for the API."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a test client
        self.client = TestClient(app)

        # Mock the RAG pipeline
        self.mock_rag_pipeline = MagicMock()
        self.mock_rag_pipeline.process_query.return_value = {
            "query": "test query",
            "response": "This is a test response.",
            "retrieved_emails": [
                {
                    "id": "email_1",
                    "content": "This is test email 1",
                    "metadata": {"subject": "Test Email 1"},
                    "similarity_score": 0.9
                },
                {
                    "id": "email_2",
                    "content": "This is test email 2",
                    "metadata": {"subject": "Test Email 2"},
                    "similarity_score": 0.8
                }
            ]
        }

        # Patch the app's RAG pipeline
        self.rag_pipeline_patcher = patch('api.app.rag_pipeline', self.mock_rag_pipeline)
        self.rag_pipeline_patcher.start()

    def tearDown(self):
        """Tear down test fixtures."""
        self.rag_pipeline_patcher.stop()

    def test_root_endpoint(self):
        """Test the root endpoint."""
        response = self.client.get("/")

        # Check that the response is successful
        self.assertEqual(response.status_code, 200)

        # Check that the response contains the expected data
        data = response.json()
        self.assertIn("message", data)
        self.assertIn("docs_url", data)

    def test_query_email_endpoint(self):
        """Test the query_email endpoint."""
        # Create a test request
        request_data = {
            "query": "test query",
            "top_k": 3
        }

        # Send the request
        response = self.client.post("/query_email", json=request_data)

        # Check that the response is successful
        self.assertEqual(response.status_code, 200)

        # Check that the RAG pipeline was called with the expected arguments
        self.mock_rag_pipeline.process_query.assert_called_with(
            query="test query",
            top_k=3,
            filter_criteria=None
        )

        # Check that the response contains the expected data
        data = response.json()
        self.assertEqual(data["query"], "test query")
        self.assertEqual(data["response"], "This is a test response.")
        self.assertEqual(len(data["retrieved_emails"]), 2)
        self.assertEqual(data["retrieved_emails"][0]["id"], "email_1")
        self.assertEqual(data["retrieved_emails"][0]["content"], "This is test email 1")
        self.assertEqual(data["retrieved_emails"][0]["metadata"], {"subject": "Test Email 1"})
        self.assertEqual(data["retrieved_emails"][0]["similarity_score"], 0.9)

    def test_query_email_with_filter(self):
        """Test the query_email endpoint with filter criteria."""
        # Create a test request with filter criteria
        request_data = {
            "query": "test query",
            "top_k": 3,
            "filter_criteria": {"topic": "business"}
        }

        # Send the request
        response = self.client.post("/query_email", json=request_data)

        # Check that the response is successful
        self.assertEqual(response.status_code, 200)

        # Check that the RAG pipeline was called with the filter criteria
        self.mock_rag_pipeline.process_query.assert_called_with(
            query="test query",
            top_k=3,
            filter_criteria={"topic": "business"}
        )

    def test_health_check(self):
        """Test the health check endpoint."""
        response = self.client.get("/health")

        # Check that the response is successful
        self.assertEqual(response.status_code, 200)

        # Check that the response contains the expected data
        data = response.json()
        self.assertEqual(data["status"], "healthy")


if __name__ == "__main__":
    unittest.main()