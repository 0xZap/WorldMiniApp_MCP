#!/usr/bin/env python

import os
from mcp.server.fastmcp import FastMCP

# For retrieving from the local vector store:
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import SKLearnVectorStore

BASE_PATH = os.path.dirname(__file__)

mcp = FastMCP("WorldMiniKit-js-MCP-Server")

@mcp.tool()
def world_mini_kit_js_query_tool(query: str) -> str:
    """
    Query the World MiniKit-js documentation using a retriever.
    Returns the top few relevant doc chunks.
    """
    store_path = os.path.join(BASE_PATH, "sklearn_vectorstore.parquet")
    retriever = SKLearnVectorStore(
        embedding=OpenAIEmbeddings(model="text-embedding-3-large"),
        persist_path=store_path,
        serializer="parquet"
    ).as_retriever(search_kwargs={"k": 3})

    relevant_docs = retriever.invoke(query)
    formatted_context = ""
    for i, doc in enumerate(relevant_docs):
        formatted_context += f"==DOCUMENT {i+1}==\n{doc.page_content}\n\n"
    return formatted_context.strip()

@mcp.resource("docs://world-mini-kit-js/full")
def get_all_world_mini_kit_js_docs() -> str:
    """
    Returns the entire World MiniKit-js docs text from llms_full.txt
    (Essentially ~300k tokens of raw doc content).
    """
    doc_path = os.path.join(BASE_PATH, "llms_full.txt")
    if not os.path.exists(doc_path):
        return "llms_full.txt was not found. Did you run minikit_build_docs.py?"
    try:
        with open(doc_path, "r") as file:
            return file.read()
    except Exception as e:
        return f"Error reading docs: {e!s}"

if __name__ == "__main__":
    # Run an MCP server over stdio
    mcp.run(transport="stdio")