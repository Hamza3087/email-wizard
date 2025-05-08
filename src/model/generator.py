"""
Generator module for the Email Wizard Assistant.
This module handles the generation of responses based on retrieved emails.
"""

import os
from typing import List, Dict, Any, Optional
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline


class ResponseGenerator:
    """
    Class for generating responses based on retrieved emails.
    """

    def __init__(
        self,
        model_name: str = "google/flan-t5-base",
        device: Optional[str] = None,
        max_length: int = 512,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize the ResponseGenerator.

        Args:
            model_name: Name of the pre-trained model to use
            device: Device to use for inference (cpu or cuda)
            max_length: Maximum length of generated responses
            cache_dir: Directory to cache the model
        """
        self.model_name = model_name
        self.max_length = max_length

        # Determine device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=cache_dir)

        # Move model to device
        self.model.to(self.device)

        # Create generation pipeline
        self.generator = pipeline(
            "text2text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if self.device == "cuda" else -1
        )

        print(f"Loaded response generator model: {model_name} on {self.device}")

    def format_context(self, retrieved_emails: List[Dict[str, Any]]) -> str:
        """
        Format retrieved emails as context for the generator.

        Args:
            retrieved_emails: List of retrieved email dictionaries

        Returns:
            Formatted context string
        """
        context_parts = []

        for i, email in enumerate(retrieved_emails):
            # Extract email content
            if 'cleaned_text' in email:
                content = email['cleaned_text']
            else:
                content = email.get('content', '')

            # Extract metadata
            metadata = email.get('metadata', {})
            subject = metadata.get('subject', '')
            sender = metadata.get('sender', '')
            date = metadata.get('date', '')

            # Format email
            email_str = f"Email {i+1}:\n"
            if subject:
                email_str += f"Subject: {subject}\n"
            if sender:
                email_str += f"From: {sender}\n"
            if date:
                email_str += f"Date: {date}\n"

            email_str += f"Content: {content}\n\n"

            context_parts.append(email_str)

        return "".join(context_parts)

    def generate_response(
        self,
        query: str,
        retrieved_emails: List[Dict[str, Any]],
        temperature: float = 0.7,
        num_beams: int = 4
    ) -> str:
        """
        Generate a response based on the query and retrieved emails.

        Args:
            query: User query
            retrieved_emails: List of retrieved email dictionaries
            temperature: Sampling temperature (higher = more random)
            num_beams: Number of beams for beam search

        Returns:
            Generated response
        """
        # Format context from retrieved emails
        context = self.format_context(retrieved_emails)

        # Create prompt
        prompt = f"""
Based on the following emails, answer the query: "{query}"

{context}

Answer:
"""

        # Generate response
        response = self.generator(
            prompt,
            max_length=self.max_length,
            temperature=temperature,
            num_beams=num_beams,
            do_sample=True
        )

        # Extract generated text
        generated_text = response[0]['generated_text']

        # Clean up the response
        if "Answer:" in generated_text:
            generated_text = generated_text.split("Answer:")[1].strip()

        return generated_text


class RAGPipeline:
    """
    End-to-end RAG pipeline combining retrieval and generation.
    """

    def __init__(
        self,
        retriever: Any,
        generator: ResponseGenerator,
        top_k: int = 3
    ):
        """
        Initialize the RAGPipeline.

        Args:
            retriever: Retriever instance
            generator: ResponseGenerator instance
            top_k: Number of documents to retrieve
        """
        self.retriever = retriever
        self.generator = generator
        self.top_k = top_k

    def process_query(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_criteria: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process a query through the RAG pipeline.

        Args:
            query: User query
            top_k: Number of documents to retrieve (overrides default)
            filter_criteria: Filter to apply to the retrieval

        Returns:
            Dictionary containing the response and retrieved emails
        """
        # Retrieve relevant emails
        k = top_k or self.top_k

        if hasattr(self.retriever, 'retrieve'):
            if 'filter_criteria' in self.retriever.retrieve.__code__.co_varnames:
                retrieved_emails = self.retriever.retrieve(
                    query=query,
                    top_k=k,
                    filter_criteria=filter_criteria
                )
            else:
                retrieved_emails = self.retriever.retrieve(
                    query=query,
                    top_k=k
                )
        else:
            raise ValueError("Retriever does not have a retrieve method")

        # Generate response
        response = self.generator.generate_response(
            query=query,
            retrieved_emails=retrieved_emails
        )

        return {
            "query": query,
            "response": response,
            "retrieved_emails": retrieved_emails
        }