#!/usr/bin/env python

import os
import re
import tiktoken
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# LangChain / Vector Store imports
from langchain_community.document_loaders import RecursiveUrlLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import SKLearnVectorStore

load_dotenv()

def bs4_extractor(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    # Attempt to target the main <article> content
    main_content = soup.find("article", class_="md-content__inner")
    content = main_content.get_text() if main_content else soup.get_text()
    return re.sub(r"\n\n+", "\n\n", content).strip()

def count_tokens(text, model="cl100k_base"):
    """Count tokens using tiktoken for chunking."""
    encoder = tiktoken.get_encoding(model)
    return len(encoder.encode(text))

def load_worldcoin_docs():
    urls = [
        "https://docs.world.org/world-chain",
        "https://docs.world.org/world-chain/quick-start/why",
        "https://docs.world.org/world-chain/quick-start/features",
        "https://docs.world.org/world-chain/quick-start/data",
        "https://docs.world.org/world-chain/quick-start",
        "https://docs.world.org/world-chain/quick-start/fund-wallet",
        "https://docs.world.org/world-chain/quick-start/info",
        "https://docs.world.org/world-chain/developers/deploy",
        "https://docs.world.org/world-chain/developers/template",
        "https://docs.world.org/world-chain/developers/world-chain-contracts",
        "https://docs.world.org/world-chain/developers/fees",
        "https://docs.world.org/world-chain/developers/evm-equivalence",
        "https://docs.world.org/world-chain/developers/grants",
        "https://docs.world.org/world-chain/providers/nodes",
        "https://docs.world.org/world-chain/providers/bridges",
        "https://docs.world.org/world-chain/providers/data",
        "https://docs.world.org/world-chain/providers/explorers",
        "https://docs.world.org/world-chain/providers/developer-tooling",
        "https://docs.world.org/world-chain/providers/paymasters",
        "https://docs.world.org/world-chain/providers/onramps",
        "https://docs.world.org/world-chain/reference/address-book",
        "https://docs.world.org/world-chain/reference/node-setup",
        "https://docs.world.org/world-chain/tokens/bridging",
        "https://docs.world.org/world-chain/tokens/superchain-token",
    ]
    docs = []
    for url in urls:
        loader = RecursiveUrlLoader(url, max_depth=3, extractor=bs4_extractor) # (test) max_depth changed to 3 for better results
        for d in loader.lazy_load():
            docs.append(d)
    print(f"Loaded {len(docs)} docs total.")
    return docs

def save_llms_full(docs):
    """Concatenate docs into a single text file called llms_full.txt."""
    out_path = "llms_full.txt"
    with open(out_path, "w") as f:
        for i, doc in enumerate(docs):
            src = doc.metadata.get("source", "unknown")
            f.write(f"DOCUMENT {i+1}\n")
            f.write(f"SOURCE: {src}\n")
            f.write("CONTENT:\n")
            f.write(doc.page_content)
            f.write("\n\n" + "="*80 + "\n\n")
    print(f"Wrote all docs to {out_path}")

def split_docs(docs):
    """Split the loaded docs into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=8000,
        chunk_overlap=500
    )
    splitted = text_splitter.split_documents(docs)
    print(f"Split into {len(splitted)} chunks.")
    return splitted

def create_vectorstore(split_docs):
    """Create a local SKLearn vector store from doc chunks."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    persist_path = os.path.join(os.getcwd(), "sklearn_vectorstore.parquet")
    vs = SKLearnVectorStore.from_documents(
        documents=split_docs,
        embedding=embeddings,
        persist_path=persist_path,
        serializer="parquet"
    )
    vs.persist()
    print(f"Vector store persisted at {persist_path}")
    return vs

if __name__ == "__main__":
    # 1) Load raw docs
    raw_docs = load_worldcoin_docs()
    # 2) Save them to a single text file
    save_llms_full(raw_docs)
    # 3) Split them into chunks
    splitted = split_docs(raw_docs)
    # 4) Build + persist the vector store
    create_vectorstore(splitted)