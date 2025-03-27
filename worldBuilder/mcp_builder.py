#!/usr/bin/env python

import os
from mcp.server.fastmcp import FastMCP

# For retrieving from the local vector store:
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import SKLearnVectorStore
from dotenv import load_dotenv

BASE_PATH = os.path.dirname(__file__)

# Load environment variables from .env file
load_dotenv()

# Get API key from environment variable (or set it manually below)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


port = int(os.environ.get("PORT", "8080"))
        
# mcp = FastMCP(
#     "World-MCP-Server",
#     host="0.0.0.0",
#     port=port,
#     debug=True,
#     log_level="INFO",
# )

mcp = FastMCP(
    "World-MCP-Server",
    debug=True,
    log_level="INFO",
)

###############################################################################
# World UI Kit Query Tool
###############################################################################
@mcp.tool(
    name="world_uikit_query_tool",
    description="Query the World UI Kit documentation using a retriever. Returns the top few relevant doc chunks."
)
def world_uikit_query_tool(query: str) -> str:
    """
    Queries the World UI Kit documentation using a retriever.
    Returns the top few relevant doc chunks from 'docs/uikit/sklearn_vectorstore_uikit.parquet'.
    """
    store_path = os.path.join(BASE_PATH, "docs", "uikit", "sklearn_vectorstore_uikit.parquet")
    if not os.path.exists(store_path):
        return "UI Kit vector store file not found. Did you run uikit_build_docs.py?"
    
    retriever = SKLearnVectorStore(
        embedding=OpenAIEmbeddings(model="text-embedding-3-large"),
        persist_path=store_path,
        serializer="parquet"
    ).as_retriever(search_kwargs={"k": 3})

    relevant_docs = retriever.invoke(query)
    if not relevant_docs:
        return "No relevant UI Kit documentation found for your query."
        
    formatted_context = ""
    for i, doc in enumerate(relevant_docs):
        formatted_context += f"==DOCUMENT {i+1}==\n{doc.page_content}\n\n"
    return formatted_context.strip()


###############################################################################
# World MiniKit Query Tool 
###############################################################################
@mcp.tool(
    name="world_minikit_query_tool",
    description="Query the World Mini App/MiniKit documentation using a retriever. Returns the top few relevant doc chunks."
)
def world_minikit_query_tool(query: str) -> str:
    """
    Queries the World Mini App documentation using a retriever.
    Returns the top few relevant doc chunks from 'docs/minikit/sklearn_vectorstore_minikit.parquet'.
    """
    store_path = os.path.join(BASE_PATH, "docs", "minikit", "sklearn_vectorstore_minikit.parquet")
    if not os.path.exists(store_path):
        return "MiniKit vector store file not found. Did you run minikit_build_docs.py?"
    
    retriever = SKLearnVectorStore(
        embedding=OpenAIEmbeddings(model="text-embedding-3-large"),
        persist_path=store_path,
        serializer="parquet"
    ).as_retriever(search_kwargs={"k": 3})

    relevant_docs = retriever.invoke(query)
    if not relevant_docs:
        return "No relevant MiniKit documentation found for your query."
        
    formatted_context = ""
    for i, doc in enumerate(relevant_docs):
        formatted_context += f"==DOCUMENT {i+1}==\n{doc.page_content}\n\n"
    return formatted_context.strip()


###############################################################################
# Unified Query Tool (optional)
###############################################################################
@mcp.tool(
    name="world_docs_query_tool",
    description=(
        "A universal query tool that can search either 'uikit' or 'minikit' documentation "
        "based on the 'store_name' argument."
    )
)
def world_docs_query_tool(query: str, store_name: str) -> str:
    """
    A universal tool that picks which vector store to query, depending on 'store_name'.
    Possible values: 'uikit' or 'minikit'.
    """
    if store_name.lower() == "uikit":
        store_path = os.path.join(BASE_PATH, "docs", "uikit", "sklearn_vectorstore_uikit.parquet")
        store_name_display = "UI Kit"
    elif store_name.lower() == "minikit":
        store_path = os.path.join(BASE_PATH, "docs", "minikit", "sklearn_vectorstore_minikit.parquet")
        store_name_display = "MiniKit"
    else:
        return f"Unknown store_name '{store_name}'. Try 'uikit' or 'minikit'."

    if not os.path.exists(store_path):
        return f"Vector store file not found for {store_name_display}. Did you run the appropriate build_docs.py script?"

    retriever = SKLearnVectorStore(
        embedding=OpenAIEmbeddings(model="text-embedding-3-large"),
        persist_path=store_path,
        serializer="parquet"
    ).as_retriever(search_kwargs={"k": 3})

    relevant_docs = retriever.invoke(query)
    if not relevant_docs:
        return f"No relevant docs found in {store_name_display} documentation."

    formatted_context = ""
    for i, doc in enumerate(relevant_docs):
        formatted_context += f"==DOCUMENT {i+1} from '{store_name_display}'==\n{doc.page_content}\n\n"

    return formatted_context.strip()


###############################################################################
# UI Kit Resource
###############################################################################
@mcp.resource("docs://world-uikit/full")
def get_all_world_uikit_docs() -> str:
    """
    Returns the entire World UI Kit documentation.
    """
    doc_path = os.path.join(BASE_PATH, "docs", "uikit", "explanations_uikit.json")
    if not os.path.exists(doc_path):
        return "UI Kit explanations file not found. Did you run uikit_build_docs.py?"
    try:
        with open(doc_path, "r") as file:
            return file.read()
    except Exception as e:
        return f"Error reading UI Kit docs: {e!s}"


###############################################################################
# Mini App Resource
###############################################################################
@mcp.resource("docs://world-minikit/full")
def get_all_world_minikit_docs() -> str:
    """
    Returns the entire World MiniKit documentation.
    """
    doc_path = os.path.join(BASE_PATH, "docs", "minikit", "llms_full_minikit.txt")
    if not os.path.exists(doc_path):
        return "MiniKit docs not found. Did you run minikit_build_docs.py?"
    try:
        with open(doc_path, "r") as file:
            return file.read()
    except Exception as e:
        return f"Error reading MiniKit docs: {e!s}"


###############################################################################
# Main
###############################################################################
def main():
    """
    Main entry point for the MCP server.
    This function is called when the server is run as a CLI script.
    """
    # Check if running in GCP/Docker environment
    # mcp.run(transport="sse")
    
    mcp.run(transport="stdio")

if __name__ == "__main__":
    main()