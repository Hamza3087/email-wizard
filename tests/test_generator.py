"""
Tests for the generator module.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model.generator import ResponseGenerator, RAGPipeline


class TestResponseGenerator(unittest.TestCase):
    """Test cases for the ResponseGenerator class."""

    def setUp(self):
        """Set up test fixtures."""
        # Mock the generator
        self.generator_patcher = patch('src.model.generator.pipeline')
        self.mock_pipeline = self.generator_patcher.start()

        # Set up the mock generator
        self.mock_generator = MagicMock()
        self.mock_generator.return_value = [{'generated_text': 'Answer: This is a generated response.'}]
        self.mock_pipeline.return_value = self.mock_generator

        # Create the generator
        self.generator = ResponseGenerator(model_name="test-model")

        # Create test retrieved emails
        self.retrieved_emails = [
            {
                'id': 'email_1',
                'metadata': {
                    'subject': 'Test Email 1',
                    'sender': 'john.doe@example.com',
                    'date': 'Mon, 01 Jan 2023 12:00:00 +0000'
                },
                'cleaned_text': 'This is test email 1',
                'similarity_score': 0.9
            },
            {
                'id': 'email_2',
                'metadata': {
                    'subject': 'Test Email 2',
                    'sender': 'jane.smith@example.com',
                    'date': 'Tue, 02 Jan 2023 12:00:00 +0000'
                },
                'cleaned_text': 'This is test email 2',
                'similarity_score': 0.8
            }
        ]

    def tearDown(self):
        """Tear down test fixtures."""
        self.generator_patcher.stop()

    def test_format_context(self):
        """Test formatting retrieved emails as context."""
        context = self.generator.format_context(self.retrieved_emails)

        # Check that the context contains the email information
        self.assertIn('Email 1:', context)
        self.assertIn('Subject: Test Email 1', context)
        self.assertIn('From: john.doe@example.com', context)
        self.assertIn('Content: This is test email 1', context)

        self.assertIn('Email 2:', context)
        self.assertIn('Subject: Test Email 2', context)
        self.assertIn('From: jane.smith@example.com', context)
        self.assertIn('Content: This is test email 2', context)

    def test_generate_response(self):
        """Test generating a response."""
        response = self.generator.generate_response('test query', self.retrieved_emails)

        # Check that the generator was called with the expected prompt
        self.mock_generator.assert_called_once()
        prompt = self.mock_generator.call_args[0][0]
        self.assertIn('test query', prompt)

        # Check that the response was extracted correctly
        self.assertEqual(response, 'This is a generated response.')


class TestRAGPipeline(unittest.TestCase):
    """Test cases for the RAGPipeline class."""

    def setUp(self):
        """Set up test fixtures."""
        # Mock the retriever
        self.retriever = MagicMock()
        self.retriever.retrieve.return_value = [
            {
                'id': 'email_1',
                'metadata': {'subject': 'Test Email 1'},
                'cleaned_text': 'This is test email 1',
                'similarity_score': 0.9
            },
            {
                'id': 'email_2',
                'metadata': {'subject': 'Test Email 2'},
                'cleaned_text': 'This is test email 2',
                'similarity_score': 0.8
            }
        ]

        # Mock the generator
        self.generator = MagicMock(spec=ResponseGenerator)
        self.generator.generate_response.return_value = 'This is a generated response.'

        # Create the pipeline
        self.pipeline = RAGPipeline(
            retriever=self.retriever,
            generator=self.generator,
            top_k=3
        )

    def test_process_query(self):
        """Test processing a query through the pipeline."""
        result = self.pipeline.process_query('test query')

        # Check that the retriever was called with the expected arguments
        self.retriever.retrieve.assert_called_with(query='test query', top_k=3)

        # Check that the generator was called with the expected arguments
        self.generator.generate_response.assert_called_with(
            query='test query',
            retrieved_emails=self.retriever.retrieve.return_value
        )

        # Check that the result has the expected structure
        self.assertEqual(result['query'], 'test query')
        self.assertEqual(result['response'], 'This is a generated response.')
        self.assertEqual(result['retrieved_emails'], self.retriever.retrieve.return_value)

    def test_process_query_with_custom_top_k(self):
        """Test processing a query with a custom top_k value."""
        result = self.pipeline.process_query('test query', top_k=5)

        # Check that the retriever was called with the custom top_k value
        self.retriever.retrieve.assert_called_with(query='test query', top_k=5)

    def test_process_query_with_filter_criteria(self):
        """Test processing a query with filter criteria."""
        filter_criteria = {'topic': 'business'}

        # Update the retriever mock to handle filter_criteria
        self.retriever.retrieve.__code__ = MagicMock()
        self.retriever.retrieve.__code__.co_varnames = ('self', 'query', 'top_k', 'filter_criteria')

        result = self.pipeline.process_query('test query', filter_criteria=filter_criteria)

        # Check that the retriever was called with the filter criteria
        self.retriever.retrieve.assert_called_with(
            query='test query',
            top_k=3,
            filter_criteria=filter_criteria
        )


if __name__ == "__main__":
    unittest.main()