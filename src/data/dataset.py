"""
Dataset module for the Email Wizard Assistant.
This module handles the creation and management of email datasets.
"""

import os
import json
import random
import pandas as pd
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta

# Import preprocessing functions
from src.data.preprocessing import preprocess_emails, save_preprocessed_emails


def generate_synthetic_email(
    email_id: str,
    topics: List[str],
    senders: List[str],
    recipients: List[str],
    date_range: Tuple[datetime, datetime]
) -> Dict:
    """
    Generate a synthetic email for testing purposes.

    Args:
        email_id: Unique identifier for the email
        topics: List of possible email topics
        senders: List of possible senders
        recipients: List of possible recipients
        date_range: Tuple of (start_date, end_date) for random date generation

    Returns:
        Dictionary containing synthetic email data
    """
    # Select random topic, sender, and recipient
    topic = random.choice(topics)
    sender = random.choice(senders)
    recipient = random.choice(recipients)

    # Generate random date within range
    time_between_dates = date_range[1] - date_range[0]
    days_between_dates = time_between_dates.days
    random_days = random.randrange(days_between_dates)
    random_date = date_range[0] + timedelta(days=random_days)
    date_str = random_date.strftime("%a, %d %b %Y %H:%M:%S %z")

    # Generate subject based on topic
    subjects = {
        "business": [
            "Quarterly Business Review",
            "Project Status Update",
            "Meeting Minutes: Strategic Planning",
            "Budget Approval Request",
            "New Client Opportunity"
        ],
        "technical": [
            "System Outage Notification",
            "Code Review Feedback",
            "Deployment Schedule Update",
            "Bug Fix Release Notes",
            "API Documentation Update"
        ],
        "personal": [
            "Vacation Plans",
            "Happy Birthday!",
            "Lunch Next Week?",
            "Family Reunion Planning",
            "Congratulations on Your Promotion"
        ]
    }

    # Generate email content based on topic and subject
    subject = random.choice(subjects.get(topic, subjects["business"]))

    # Templates for different email types
    templates = {
        "business": [
            "Dear {recipient},\n\nI wanted to provide an update on our current business activities. {subject} is progressing well, and we're on track to meet our quarterly targets. The team has been working diligently to ensure all deliverables are met on time.\n\nPlease review the attached documents and let me know if you have any questions or concerns.\n\nBest regards,\n{sender}",
            "Hi {recipient},\n\nRegarding the {subject}, I've compiled the latest figures and analysis. Our performance this quarter shows a 15% improvement over last quarter, which is encouraging.\n\nWe should schedule a meeting to discuss the implications and next steps. Are you available next week?\n\nThanks,\n{sender}",
            "Hello {recipient},\n\nI'm writing to follow up on the {subject}. The executive team has reviewed our proposal and provided some feedback that we should incorporate before the final submission.\n\nCould we connect tomorrow to go through these changes?\n\nRegards,\n{sender}"
        ],
        "technical": [
            "Hi {recipient},\n\nI'm reaching out regarding the {subject}. We've identified a critical issue in the production environment that needs immediate attention. The team is currently working on a fix and will deploy it as soon as it's ready.\n\nI'll keep you updated on the progress.\n\nBest,\n{sender}",
            "Hello {recipient},\n\nThe {subject} has been completed. All tests are passing, and the documentation has been updated accordingly. You can find the latest version in the repository.\n\nPlease review when you have a chance and let me know if any changes are needed.\n\nThanks,\n{sender}",
            "Dear {recipient},\n\nI've reviewed the code for the {subject} and have some suggestions for improvements. The current implementation works, but there are a few optimizations we could make to enhance performance.\n\nLet me know if you'd like to discuss these in more detail.\n\nRegards,\n{sender}"
        ],
        "personal": [
            "Hey {recipient},\n\nJust wanted to check in about {subject}. It's been a while since we caught up, and I'd love to hear how you're doing.\n\nWould you be free for coffee sometime next week?\n\nCheers,\n{sender}",
            "Hi {recipient},\n\nI hope this email finds you well! I'm excited about the upcoming {subject} and wanted to see if you're planning to attend.\n\nIt would be great to see you there. Let me know your thoughts.\n\nBest wishes,\n{sender}",
            "Dear {recipient},\n\nI wanted to reach out regarding the {subject}. I've been thinking about it a lot lately and would appreciate your input.\n\nCould we schedule a call to discuss it further?\n\nTake care,\n{sender}"
        ]
    }

    # Select a random template based on the topic
    template = random.choice(templates.get(topic, templates["business"]))

    # Fill in the template
    content = template.format(recipient=recipient, subject=subject.lower(), sender=sender)

    # Construct the full email with headers
    full_email = f"From: {sender}\nTo: {recipient}\nSubject: {subject}\nDate: {date_str}\n\n{content}"

    return {
        "id": email_id,
        "content": full_email,
        "metadata": {
            "subject": subject,
            "sender": sender,
            "recipient": recipient,
            "date": date_str,
            "topic": topic
        }
    }


def create_synthetic_dataset(
    num_emails: int = 60,
    output_path: str = "data/raw/synthetic_emails.json",
    preprocess: bool = True,
    processed_path: str = "data/processed/processed_emails.json"
) -> List[Dict]:
    """
    Create a synthetic email dataset for testing purposes.

    Args:
        num_emails: Number of synthetic emails to generate
        output_path: Path to save the raw synthetic emails
        preprocess: Whether to preprocess the emails
        processed_path: Path to save the preprocessed emails

    Returns:
        List of generated email dictionaries
    """
    # Define possible topics, senders, and recipients
    topics = ["business", "technical", "personal"]

    senders = [
        "john.doe@example.com",
        "jane.smith@example.com",
        "michael.johnson@example.com",
        "emily.williams@example.com",
        "robert.brown@example.com",
        "sarah.davis@example.com",
        "david.miller@example.com",
        "jennifer.wilson@example.com",
        "william.moore@example.com",
        "lisa.taylor@example.com"
    ]

    recipients = [
        "alex.anderson@example.com",
        "olivia.thomas@example.com",
        "james.jackson@example.com",
        "sophia.white@example.com",
        "benjamin.harris@example.com",
        "emma.martin@example.com",
        "daniel.thompson@example.com",
        "ava.garcia@example.com",
        "matthew.martinez@example.com",
        "charlotte.robinson@example.com"
    ]

    # Define date range (last 6 months)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)
    date_range = (start_date, end_date)

    # Generate synthetic emails
    emails = []
    for i in range(num_emails):
        email_id = f"email_{i+1}"
        email = generate_synthetic_email(email_id, topics, senders, recipients, date_range)
        emails.append(email)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save raw emails
    with open(output_path, 'w') as f:
        json.dump(emails, f, indent=2)

    print(f"Generated {num_emails} synthetic emails and saved to {output_path}")

    # Preprocess emails if requested
    if preprocess:
        processed_emails = preprocess_emails(emails)

        # Ensure processed directory exists
        os.makedirs(os.path.dirname(processed_path), exist_ok=True)

        # Save processed emails
        save_preprocessed_emails(processed_emails, processed_path)

    return emails


def load_dataset(
    path: str,
    is_processed: bool = True
) -> List[Dict]:
    """
    Load an email dataset from a file.

    Args:
        path: Path to the dataset file
        is_processed: Whether the dataset is already preprocessed

    Returns:
        List of email dictionaries
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file not found: {path}")

    if path.endswith('.json'):
        with open(path, 'r') as f:
            emails = json.load(f)
    elif path.endswith('.jsonl'):
        emails = []
        with open(path, 'r') as f:
            for line in f:
                emails.append(json.loads(line))
    else:
        raise ValueError(f"Unsupported file format: {path}")

    if not is_processed:
        emails = preprocess_emails(emails)

    return emails


def split_dataset(
    emails: List[Dict],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_seed: int = 42
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Split the dataset into training, validation, and test sets.

    Args:
        emails: List of email dictionaries
        train_ratio: Ratio of training data
        val_ratio: Ratio of validation data
        test_ratio: Ratio of test data
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (train_emails, val_emails, test_emails)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-10, "Ratios must sum to 1"

    # Shuffle emails
    random.seed(random_seed)
    shuffled_emails = emails.copy()
    random.shuffle(shuffled_emails)

    # Calculate split indices
    n = len(shuffled_emails)
    train_idx = int(n * train_ratio)
    val_idx = train_idx + int(n * val_ratio)

    # Split the dataset
    train_emails = shuffled_emails[:train_idx]
    val_emails = shuffled_emails[train_idx:val_idx]
    test_emails = shuffled_emails[val_idx:]

    return train_emails, val_emails, test_emails