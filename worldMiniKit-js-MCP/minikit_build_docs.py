import os
import re
import tiktoken
from dotenv import load_dotenv

# LangChain / Vector Store imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import SKLearnVectorStore
from langchain.docstore.document import Document

load_dotenv()

def load_text_from_md(path: str) -> str:
        """Load text from markdown file
        
        Args:
        - path: path to the prompt

        Returns:
        - prompt: text read from md
        """
        current_dir = os.path.dirname(os.path.abspath(__file__))
        prompt_path = os.path.join(current_dir, path)
        
        with open(prompt_path, "r") as file:
            return file.read()

def load_worldkit_js_docs():
    docs = load_text_from_md("worldkit.md")
    return [Document(page_content=docs, metadata={"source": "worldkit.md"})]

def save_llms_full(docs):
    """Concatenate docs into a single text file called llms_full.txt."""
    out_path = "llms_full.txt"
    with open(out_path, "w") as f:
        f.write("DOCUMENT\n")
        f.write("SOURCE: unknown\n")
        f.write("CONTENT:\n")
        f.write(docs[0].page_content)
        f.write("\n\n" + "="*80 + "\n\n")
    print(f"Wrote all docs to {out_path}")

def split_docs(docs):
    """Split the loaded docs into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=8000,
        chunk_overlap=500,
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
    raw_docs = load_worldkit_js_docs()
    # 2) Save them to a single text file
    save_llms_full(raw_docs)
    # 3) Split them into chunks
    splitted = split_docs(raw_docs)
    # 4) Build + persist the vector store
    create_vectorstore(splitted)