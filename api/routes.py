"""
API routes for the Email Wizard Assistant.
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, List, Any, Optional
from pydantic import BaseModel

# Import models
from api.app import QueryRequest, QueryResponse, EmailResponse, rag_pipeline

# Create router
router = APIRouter(
    prefix="/api",
    tags=["email-wizard"]
)


@router.get("/")
async def api_root():
    """API root endpoint."""
    return {
        "message": "Email Wizard Assistant API",
        "version": "0.1.0",
        "endpoints": [
            "/api/query",
            "/api/health",
            "/api/stats"
        ]
    }


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Query the Email Wizard Assistant.

    Args:
        request: Query request

    Returns:
        Query response
    """
    try:
        # Process the query
        result = rag_pipeline.process_query(
            query=request.query,
            top_k=request.top_k,
            filter_criteria=request.filter_criteria
        )

        # Format the response
        response = QueryResponse(
            query=result["query"],
            response=result["response"],
            retrieved_emails=[
                EmailResponse(
                    id=email.get("id", ""),
                    content=email.get("content", ""),
                    metadata=email.get("metadata", {}),
                    similarity_score=email.get("similarity_score")
                )
                for email in result["retrieved_emails"]
            ]
        )

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))