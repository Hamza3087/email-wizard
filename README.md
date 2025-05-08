# Email Wizard Assistant

An AI-powered Email Wizard Assistant using Retrieval-Augmented Generation (RAG) technology to help users quickly find answers to their email queries by retrieving relevant past emails and generating intelligent responses.

## Project Overview

This project implements a RAG-based system that:

1. Embeds emails using a pre-trained model
2. Stores embeddings for efficient retrieval
3. Retrieves relevant emails based on user queries
4. Generates coherent responses using a language model

## System Architecture

```
                  +-------------------+
                  |   User Query      |
                  +-------------------+
                           |
                           v
+-------------+    +-------------------+    +-------------+
| Email       |    | Embedding Model   |    | Vector      |
| Dataset     |--->| (Sentence-        |--->| Database    |
| (50-60      |    | Transformers)     |    | (ChromaDB)  |
| emails)     |    +-------------------+    +-------------+
+-------------+             |                      |
                           |                      |
                           v                      v
                  +-------------------+    +-------------+
                  | Similarity Search |<---| Retrieved   |
                  | (ANN)            |    | Emails      |
                  +-------------------+    +-------------+
                           |
                           v
                  +-------------------+
                  | Response          |
                  | Generation        |
                  | (Language Model)  |
                  +-------------------+
                           |
                           v
                  +-------------------+
                  | API Response      |
                  +-------------------+
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Setup

1. Clone the repository:

   ```
   git clone https://github.com/yourusername/email-wizard-assistant.git
   cd email-wizard-assistant
   ```

2. Install dependencies:

   ```
   pip install -r requirements.txt
   ```

3. Install the package in development mode:
   ```
   pip install -e .
   ```

## Usage

### Running the Jupyter Notebooks

1. Start Jupyter:

   ```
   jupyter notebook
   ```

2. Navigate to the `notebooks` directory and open the notebooks in order:
   - `01_data_preparation.ipynb`: Data exploration and preparation
   - `02_model_implementation.ipynb`: RAG implementation
   - `03_evaluation.ipynb`: System evaluation

### Starting the API Server

1. Run the FastAPI server:

   ```
   cd api
   uvicorn app:app --reload
   ```

2. Access the API documentation at `http://localhost:8000/docs`

### API Endpoints

- `POST /query_email`: Submit a query and get a generated response

Example request:

```json
{
  "query": "What's the status of my project?"
}
```

Example response:

```json
{
  "response": "Based on the latest emails, your project is currently in the testing phase. The development team completed the main features last week and is now addressing some minor bugs before the final release.",
  "retrieved_emails": [
    {
      "id": "email123",
      "subject": "Project Status Update",
      "content": "...",
      "similarity_score": 0.92
    }
  ]
}
```

## Evaluation

The system is evaluated based on:

1. Search Performance:

   - Speed: Time to retrieve similar emails
   - Accuracy: Relevance of retrieved emails to query

2. Response Quality:
   - Coherence: Is the response well-structured?
   - Relevance: Does it answer the query accurately?
   - Helpfulness: Is the information useful?

Detailed evaluation results can be found in the `notebooks/03_evaluation.ipynb` notebook.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Hugging Face for providing pre-trained models
- ChromaDB for vector storage capabilities
