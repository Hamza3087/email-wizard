"""
Tests for the preprocessing module.
"""

import os
import sys
import unittest
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.preprocessing import (
    clean_email_text,
    extract_metadata,
    chunk_email,
    preprocess_email
)


class TestPreprocessing(unittest.TestCase):
    """Test cases for the preprocessing module."""

    def test_clean_email_text(self):
        """Test the clean_email_text function."""
        # Test input
        input_text = """From: john.doe@example.com
To: jane.smith@example.com
Subject: Test Email
Date: Mon, 01 Jan 2023 12:00:00 +0000

Hello Jane,

This is a test email.

Best regards,
John
--
John Doe
Software Engineer
Example Corp
"""

        # Expected output
        expected_output = """Hello Jane,

This is a test email.

Best regards,
John"""

        # Test the function
        result = clean_email_text(input_text)
        self.assertEqual(result, expected_output)

    def test_extract_metadata(self):
        """Test the extract_metadata function."""
        # Test input
        input_text = """From: john.doe@example.com
To: jane.smith@example.com
Subject: Test Email
Date: Mon, 01 Jan 2023 12:00:00 +0000

Hello Jane,

This is a test email.

Best regards,
John
"""

        # Expected output
        expected_output = {
            "subject": "Test Email",
            "sender": "john.doe@example.com",
            "date": "Mon, 01 Jan 2023 12:00:00 +0000"
        }

        # Test the function
        result = extract_metadata(input_text)
        self.assertEqual(result, expected_output)

    def test_chunk_email(self):
        """Test the chunk_email function."""
        # Test input
        input_text = "This is a short email that should not be chunked."

        # Test with short text (no chunking)
        result = chunk_email(input_text, chunk_size=100, overlap=20)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], input_text)

        # Test with long text (should be chunked)
        long_text = "This is a longer email. " * 20
        result = chunk_email(long_text, chunk_size=100, overlap=20)
        self.assertTrue(len(result) > 1)

    def test_preprocess_email(self):
        """Test the preprocess_email function."""
        # Test input
        email_data = {
            "id": "test_email_1",
            "content": """From: john.doe@example.com
To: jane.smith@example.com
Subject: Test Email
Date: Mon, 01 Jan 2023 12:00:00 +0000

Hello Jane,

This is a test email.

Best regards,
John
"""
        }

        # Test the function
        result = preprocess_email(email_data)

        # Check the result
        self.assertEqual(result["id"], "test_email_1")
        self.assertIn("metadata", result)
        self.assertIn("raw_text", result)
        self.assertIn("cleaned_text", result)
        self.assertIn("chunks", result)
        self.assertEqual(result["metadata"]["subject"], "Test Email")


if __name__ == "__main__":
    unittest.main()