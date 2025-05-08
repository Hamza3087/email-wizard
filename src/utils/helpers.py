"""
Helper utilities for the Email Wizard Assistant.
"""

import os
import time
import json
import logging
from typing import Dict, List, Any, Optional, Union, Callable
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime


def setup_logging(
    log_file: Optional[str] = None,
    level: int = logging.INFO
) -> logging.Logger:
    """
    Set up logging configuration.

    Args:
        log_file: Path to log file (if None, logs to console only)
        level: Logging level

    Returns:
        Configured logger
    """
    # Create logger
    logger = logging.getLogger("email_wizard")
    logger.setLevel(level)

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Create file handler if log_file is provided
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def time_function(func: Callable) -> Callable:
    """
    Decorator to measure function execution time.

    Args:
        func: Function to time

    Returns:
        Wrapped function
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function {func.__name__} took {end_time - start_time:.4f} seconds to execute")
        return result

    return wrapper


def save_json(data: Union[Dict, List], file_path: str) -> None:
    """
    Save data to a JSON file.

    Args:
        data: Data to save
        file_path: Path to save the data
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Saved data to {file_path}")


def load_json(file_path: str) -> Union[Dict, List]:
    """
    Load data from a JSON file.

    Args:
        file_path: Path to the JSON file

    Returns:
        Loaded data
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, 'r') as f:
        data = json.load(f)

    return data


def plot_evaluation_metrics(
    metrics: Dict[str, List[float]],
    title: str = "Evaluation Metrics",
    save_path: Optional[str] = None
) -> None:
    """
    Plot evaluation metrics.

    Args:
        metrics: Dictionary of metric names and values
        title: Plot title
        save_path: Path to save the plot (if None, displays the plot)
    """
    plt.figure(figsize=(10, 6))

    for metric_name, values in metrics.items():
        plt.plot(values, label=metric_name)

    plt.title(title)
    plt.xlabel("Samples")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Saved plot to {save_path}")
    else:
        plt.show()


def calculate_average_metrics(metrics: Dict[str, List[float]]) -> Dict[str, float]:
    """
    Calculate average values for each metric.

    Args:
        metrics: Dictionary of metric names and values

    Returns:
        Dictionary of metric names and average values
    """
    return {
        metric_name: np.mean(values) for metric_name, values in metrics.items()
    }


def get_timestamp() -> str:
    """
    Get current timestamp as a string.

    Returns:
        Timestamp string
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def format_results_for_display(
    query_results: Dict[str, Any],
    max_emails: int = 3,
    max_content_length: int = 200
) -> Dict[str, Any]:
    """
    Format query results for display.

    Args:
        query_results: Results from the RAG pipeline
        max_emails: Maximum number of emails to include
        max_content_length: Maximum length of email content

    Returns:
        Formatted results
    """
    formatted_results = {
        "query": query_results["query"],
        "response": query_results["response"],
        "retrieved_emails": []
    }

    # Format retrieved emails
    for i, email in enumerate(query_results["retrieved_emails"][:max_emails]):
        # Extract content
        if 'cleaned_text' in email:
            content = email['cleaned_text']
        else:
            content = email.get('content', '')

        # Truncate content if too long
        if len(content) > max_content_length:
            content = content[:max_content_length] + "..."

        # Extract metadata
        metadata = email.get('metadata', {})

        # Format email
        formatted_email = {
            "id": email.get('id', f"email_{i}"),
            "content": content,
            "metadata": metadata
        }

        # Add similarity score if available
        if 'similarity_score' in email:
            formatted_email['similarity_score'] = email['similarity_score']

        formatted_results["retrieved_emails"].append(formatted_email)

    return formatted_results