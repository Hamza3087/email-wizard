"""
Email preprocessing module for the Email Wizard Assistant.
This module handles cleaning, normalization, and preparation of email data.
"""

import re
import pandas as pd
from typing import Dict, List, Union, Optional


def clean_email_text(text: str) -> str:
    """
    Clean and normalize email text.

    Args:
        text: Raw email text

    Returns:
        Cleaned and normalized text
    """
    # Remove email headers (common patterns)
    text = re.sub(r'From:.*?\n', '', text)
    text = re.sub(r'To:.*?\n', '', text)
    text = re.sub(r'Subject:.*?\n', '', text)
    text = re.sub(r'Date:.*?\n', '', text)
    text = re.sub(r'Cc:.*?\n', '', text)
    text = re.sub(r'Bcc:.*?\n', '', text)

    # Remove email signatures
    text = re.sub(r'--+.*?$', '', text, flags=re.DOTALL)

    # Remove URLs
    text = re.sub(r'https?://\S+', '[URL]', text)

    # Remove multiple newlines
    text = re.sub(r'\n\s*\n', '\n\n', text)

    # Remove leading/trailing whitespace
    text = text.strip()

    return text


def extract_metadata(email_text: str) -> Dict[str, str]:
    """
    Extract metadata from email text.

    Args:
        email_text: Raw email text

    Returns:
        Dictionary containing metadata (subject, sender, date)
    """
    metadata = {}

    # Extract subject
    subject_match = re.search(r'Subject: (.*?)\n', email_text)
    if subject_match:
        metadata['subject'] = subject_match.group(1).strip()
    else:
        metadata['subject'] = ""

    # Extract sender
    sender_match = re.search(r'From: (.*?)\n', email_text)
    if sender_match:
        metadata['sender'] = sender_match.group(1).strip()
    else:
        metadata['sender'] = ""

    # Extract date
    date_match = re.search(r'Date: (.*?)\n', email_text)
    if date_match:
        metadata['date'] = date_match.group(1).strip()
    else:
        metadata['date'] = ""

    return metadata


def chunk_email(email_text: str, chunk_size: int = 512, overlap: int = 128) -> List[str]:
    """
    Split long emails into overlapping chunks for better embedding.

    Args:
        email_text: Cleaned email text
        chunk_size: Maximum chunk size in characters
        overlap: Overlap between chunks in characters

    Returns:
        List of email chunks
    """
    if len(email_text) <= chunk_size:
        return [email_text]

    chunks = []
    start = 0

    while start < len(email_text):
        # Find a good breaking point (end of sentence or paragraph)
        end = min(start + chunk_size, len(email_text))

        if end < len(email_text):
            # Try to find sentence boundary
            sentence_end = email_text.rfind('.', start, end)
            paragraph_end = email_text.rfind('\n\n', start, end)

            if paragraph_end > start + chunk_size // 2:
                end = paragraph_end + 2
            elif sentence_end > start + chunk_size // 2:
                end = sentence_end + 1

        chunks.append(email_text[start:end])
        start = end - overlap

    return chunks


def preprocess_email(email_data: Dict[str, Union[str, Dict]]) -> Dict:
    """
    Preprocess a single email.

    Args:
        email_data: Dictionary containing email data

    Returns:
        Preprocessed email data
    """
    email_id = email_data.get('id', '')
    raw_text = email_data.get('content', '')

    # Extract metadata if not already present
    if 'metadata' not in email_data:
        metadata = extract_metadata(raw_text)
    else:
        metadata = email_data['metadata']

    # Clean the email text
    cleaned_text = clean_email_text(raw_text)

    # Chunk the email if it's too long
    chunks = chunk_email(cleaned_text)

    return {
        'id': email_id,
        'metadata': metadata,
        'raw_text': raw_text,
        'cleaned_text': cleaned_text,
        'chunks': chunks
    }


def preprocess_emails(emails: List[Dict]) -> List[Dict]:
    """
    Preprocess a list of emails.

    Args:
        emails: List of email dictionaries

    Returns:
        List of preprocessed email dictionaries
    """
    return [preprocess_email(email) for email in emails]


def save_preprocessed_emails(emails: List[Dict], output_path: str) -> None:
    """
    Save preprocessed emails to a file.

    Args:
        emails: List of preprocessed email dictionaries
        output_path: Path to save the preprocessed emails
    """
    df = pd.DataFrame(emails)
    df.to_json(output_path, orient='records', lines=True)
    print(f"Saved {len(emails)} preprocessed emails to {output_path}")


def load_preprocessed_emails(input_path: str) -> List[Dict]:
    """
    Load preprocessed emails from a file.

    Args:
        input_path: Path to the preprocessed emails file

    Returns:
        List of preprocessed email dictionaries
    """
    df = pd.read_json(input_path, orient='records', lines=True)
    return df.to_dict(orient='records')