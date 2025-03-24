#!/usr/bin/env python

import os
from mcp.server.fastmcp import FastMCP, Context
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import SKLearnVectorStore

BASE_PATH = os.path.dirname(__file__)

def main():
    # Detect the port from environment (Cloud Run typically sets PORT=8080)
    port = int(os.environ.get("PORT", "8000"))
        
    mcp = FastMCP(
        "WorldUI-Docs-MCP-Server",
        host="0.0.0.0",
        port=port,
        debug=True,
        log_level="INFO",
    )

    @mcp.tool()
    def world_ui_query_tool(query: str, ctx: Context) -> str:
        """
        Query the World UI documentation using a retriever.
        Returns the top few relevant doc chunks.
        """
        ctx.info(f"Received query: {query}")

        store_path = os.path.join(BASE_PATH, "sklearn_vectorstore.parquet")
        retriever = SKLearnVectorStore(
            embedding=OpenAIEmbeddings(model="text-embedding-3-large"),
            persist_path=store_path,
            serializer="parquet"
        ).as_retriever(search_kwargs={"k": 3})

        relevant_docs = retriever.invoke(query)
        ctx.debug(f"Found {len(relevant_docs)} relevant documents.")

        formatted_context = ""
        for i, doc in enumerate(relevant_docs):
            formatted_context += f"==DOCUMENT {i+1}==\n{doc.page_content}\n\n"

        ctx.info("Returning final formatted context.")
        return formatted_context.strip()

    @mcp.resource("docs://world-ui/full")
    def get_all_world_ui_docs() -> str:
        """
        Returns the entire World UI docs text from world_ui_full.txt
        (Essentially ~300k tokens of raw doc content).
        """
        doc_path = os.path.join(BASE_PATH, "llms_full.txt")
        if not os.path.exists(doc_path):
            return "llms_full.txt was not found. Did you run worldUI_build_docs.py?"
        try:
            with open(doc_path, "r") as file:
                return file.read()
        except Exception as e:
            return f"Error reading docs: {e!s}"
        
    mcp.run(transport="sse")

if __name__ == "__main__":
    # Run an MCP server over SSE
    main()