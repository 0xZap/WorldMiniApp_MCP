#!/usr/bin/env python

import os
import asyncio
from typing import Dict, Any
from mcp.server.fastmcp import FastMCP

# For retrieving from the local vector store:
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import SKLearnVectorStore
from dotenv import load_dotenv

# For authentication and error handling
from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader
from starlette.status import HTTP_401_UNAUTHORIZED

BASE_PATH = os.path.dirname(__file__)

# Load environment variables from .env file
load_dotenv()

# Get API key from environment variable (or set it manually below)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Simple in-memory API key store - replace with your own keys
VALID_API_KEYS = {
    os.getenv("MCP_API_KEY", "test-key"): "test_user"  # Default test key if not set
}

# API Key security
api_key_header = APIKeyHeader(name="x-api-key", auto_error=False)

# Authentication middleware
async def verify_api_key(api_key: str = Security(api_key_header)) -> Dict[str, Any]:
    if not api_key:
        raise HTTPException(
            status_code=HTTP_401_UNAUTHORIZED,
            detail="No API key provided"
        )
    
    user_id = VALID_API_KEYS.get(api_key)
    if not user_id:
        raise HTTPException(
            status_code=HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    return {"api_key": api_key, "user_id": user_id}

# Initialize FastMCP with optimized settings
port = int(os.environ.get("PORT", "8080"))
        
mcp = FastMCP(
    "World-MCP-Server",
    host="0.0.0.0",
    port=port,
    debug=True,
    log_level="INFO",
    # Add FastAPI middleware for better error handling
    middleware=[verify_api_key],
    # Configure for high concurrency
    workers=4,  # Number of worker processes
    backlog=2048,  # Connection queue size
    limit_concurrency=1000,  # Max concurrent connections
    timeout_keep_alive=5,  # Keep-alive timeout
)

###############################################################################
# World UI Kit Query Tool
###############################################################################
@mcp.tool(
    name="world_uikit_query_tool",
    description="Searches World UI Kit documentation for component usage, styling guidelines, and best practices. Provides detailed code examples and explanations for implementing UI components in World mini applications."
)
async def world_uikit_query_tool(query: str) -> str:
    """
    Performs semantic search across the World UI Kit documentation.
    Returns detailed explanations with code examples from the most relevant document chunks.
    Use this tool when you need information about UI components, styling, layouts, or implementation best practices.
    
    Args:
        query: Specific question about UI Kit components, usage patterns, or styling options.
              For best results, be specific (e.g., "How to implement a BottomSheet component with snap points")
              
    Returns:
        Formatted text containing the 3 most relevant documentation sections with clear separation between documents.
    """
    store_path = os.path.join(BASE_PATH, "docs", "uikit", "sklearn_vectorstore_uikit.parquet")
    if not os.path.exists(store_path):
        return "UI Kit vector store file not found. Did you run uikit_build_docs.py?"
    
    try:
        retriever = SKLearnVectorStore(
            embedding=OpenAIEmbeddings(model="text-embedding-3-large"),
            persist_path=store_path,
            serializer="parquet"
        ).as_retriever(search_kwargs={"k": 3})

        relevant_docs = await retriever.ainvoke(query)
        if not relevant_docs:
            return "No relevant UI Kit documentation found for your query."
            
        formatted_context = ""
        for i, doc in enumerate(relevant_docs):
            formatted_context += f"==DOCUMENT {i+1}==\n{doc.page_content}\n\n"
        return formatted_context.strip()
    except Exception as e:
        # Log the error and return a user-friendly message
        print(f"Error in UI Kit query: {str(e)}")
        return "An error occurred while querying the UI Kit documentation."


###############################################################################
# World MiniKit Query Tool 
###############################################################################
@mcp.tool(
    name="world_minikit_query_tool",
    description="Retrieves information about World Mini Apps development, MiniKit SDK features, and integration patterns. Provides code samples and implementation guides for building World mini applications."
)
async def world_minikit_query_tool(query: str) -> str:
    """
    Performs semantic search across World Mini App/MiniKit documentation.
    Provides detailed implementation guidance, API explanations, and best practices for Mini App development.
    Use this tool for questions about application structure, API usage, or MiniKit integration.
    
    Args:
        query: Specific question about MiniKit functionality, implementation approaches, or APIs.
               Be specific with technical requirements (e.g., "How to implement World ID verification in a Mini App")
    
    Returns:
        Formatted text containing the 3 most relevant documentation sections with helpful context and examples.
    """
    store_path = os.path.join(BASE_PATH, "docs", "minikit", "sklearn_vectorstore_minikit.parquet")
    if not os.path.exists(store_path):
        return "MiniKit vector store file not found. Did you run minikit_build_docs.py?"
    
    try:
        retriever = SKLearnVectorStore(
            embedding=OpenAIEmbeddings(model="text-embedding-3-large"),
            persist_path=store_path,
            serializer="parquet"
        ).as_retriever(search_kwargs={"k": 3})

        relevant_docs = await retriever.ainvoke(query)
        if not relevant_docs:
            return "No relevant MiniKit documentation found for your query."
            
        formatted_context = ""
        for i, doc in enumerate(relevant_docs):
            formatted_context += f"==DOCUMENT {i+1}==\n{doc.page_content}\n\n"
        return formatted_context.strip()
    except Exception as e:
        # Log the error and return a user-friendly message
        print(f"Error in MiniKit query: {str(e)}")
        return "An error occurred while querying the MiniKit documentation."


###############################################################################
# UI Kit Resource
###############################################################################
@mcp.resource("docs://world-uikit/full")
async def get_all_world_uikit_docs() -> str:
    """
    Provides access to the complete World UI Kit documentation collection.
    This resource returns the entire UI Kit documentation as a single JSON object
    containing component definitions, usage examples, and implementation guidelines.
    
    Use this resource when you need comprehensive information about all available UI components
    rather than searching for specific topics.
    
    Returns:
        A comprehensive JSON document containing all UI Kit documentation.
    """
    doc_path = os.path.join(BASE_PATH, "docs", "uikit", "explanations_uikit.json")
    if not os.path.exists(doc_path):
        return "UI Kit explanations file not found. Did you run uikit_build_docs.py?"
    try:
        async with aiofiles.open(doc_path, "r") as file:
            return await file.read()
    except Exception as e:
        return f"Error reading UI Kit docs: {e!s}"


###############################################################################
# Mini App Resource
###############################################################################
@mcp.resource("docs://world-minikit/full")
async def get_all_world_minikit_docs() -> str:
    """
    Provides access to the complete World MiniKit documentation.
    This resource returns the entire Mini App documentation as a single text file
    containing framework concepts, API references, and implementation guidelines.
    
    Use this resource when you need comprehensive information about Mini App development
    or when exploring the entire documentation set rather than searching for specific topics.
    
    Returns:
        A comprehensive text document containing all MiniKit documentation.
    """
    doc_path = os.path.join(BASE_PATH, "docs", "minikit", "llms_full_minikit.txt")
    if not os.path.exists(doc_path):
        return "MiniKit docs not found. Did you run minikit_build_docs.py?"
    try:
        async with aiofiles.open(doc_path, "r") as file:
            return await file.read()
    except Exception as e:
        return f"Error reading MiniKit docs: {e!s}"


###############################################################################
# Main
###############################################################################
def main():
    """
    Main entry point for the MCP server.
    This function is called when the server is run as a CLI script.
    
    Transport mode can be controlled via MCP_TRANSPORT environment variable:
    - "stdio": For stdio transport (default for local development)
    - "sse": For SSE transport (used in production/server deployments)
    """
    # Get transport mode from environment variable, default to stdio for safety
    transport_mode = os.environ.get("MCP_TRANSPORT", "stdio")
    
    if transport_mode == "sse":
        # SSE mode needs host and port configuration
        mcp.run(
            transport="sse"
        )
    else:
        # stdio mode is simpler, good for local development
        mcp.run(transport="stdio")

if __name__ == "__main__":
    main()