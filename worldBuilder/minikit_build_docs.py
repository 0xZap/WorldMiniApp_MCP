import os
import re
import tiktoken
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import sys
# LangChain / Vector Store imports
from langchain_community.document_loaders import RecursiveUrlLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import SKLearnVectorStore

load_dotenv()

if not os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") == "your_openai_api_key_here":
    print("Error: OPENAI_API_KEY is not set or is still the default value.")
    print("Please set your OpenAI API key in the .env file located in the worldBuilder directory.")
    print("Example: OPENAI_API_KEY=sk-yourapikey")
    sys.exit(1)

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
        "https://docs.world.org/mini-apps",
        "https://docs.world.org/mini-apps/quick-start/installing",
        "https://docs.world.org/mini-apps/quick-start/commands",
        "https://docs.world.org/mini-apps/quick-start/responses",
        "https://docs.world.org/mini-apps/quick-start/testing",
        "https://docs.world.org/mini-apps/quick-start/app-store",
        "https://docs.world.org/mini-apps/design/app-guidelines",
        "https://docs.world.org/mini-apps/design/ui-kit",
        "https://docs.world.org/mini-apps/commands/verify",
        "https://docs.world.org/mini-apps/commands/pay",
        "https://docs.world.org/mini-apps/commands/wallet-auth",
        "https://docs.world.org/mini-apps/commands/connect-wallet",
        "https://docs.world.org/mini-apps/commands/send-transaction",
        "https://docs.world.org/mini-apps/commands/sign-message",
        "https://docs.world.org/mini-apps/commands/sign-typed-data",
        "https://docs.world.org/mini-apps/commands/share-contacts",
        "https://docs.world.org/mini-apps/commands/send-notifications",
        "https://docs.world.org/mini-apps/commands/get-permissions",
        "https://docs.world.org/mini-apps/commands/send-haptic-feedback",
        "https://docs.world.org/mini-apps/reference/api",
        "https://docs.world.org/mini-apps/reference/errors",
        "https://docs.world.org/mini-apps/reference/address-book",
        "https://docs.world.org/mini-apps/reference/usernames",
        "https://docs.world.org/mini-apps/reference/status-page",
        "https://docs.world.org/mini-apps/reference/payment-methods",
        "https://docs.world.org/mini-apps/sharing/quick-actions",
        "https://docs.world.org/mini-apps/sharing/uno-qa",
        "https://docs.world.org/mini-apps/sharing/eggs-vault-qa",
        "https://docs.world.org/mini-apps/sharing/earn-wld-qa",
        "https://docs.world.org/mini-apps/sharing/dna-qa",
        "https://docs.world.org/mini-apps/sharing/sage-qa",
        "https://docs.world.org/mini-apps/sharing/world-chat-qa",
        "https://docs.world.org/mini-apps/more/grants"
    ]
    docs = []
    for url in urls:
        loader = RecursiveUrlLoader(url, max_depth=5, extractor=bs4_extractor)
        for d in loader.lazy_load():
            docs.append(d)
    print(f"Loaded {len(docs)} docs total.")
    return docs

def save_llms_full(docs):
    """Concatenate docs into a single text file."""
    out_dir = os.path.join("worldBuilder", "docs", "minikit")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "llms_full_minikit.txt")
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
    out_dir = os.path.join("worldBuilder", "docs", "minikit")
    os.makedirs(out_dir, exist_ok=True)
    persist_path = os.path.join(out_dir, "sklearn_vectorstore_minikit.parquet")
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