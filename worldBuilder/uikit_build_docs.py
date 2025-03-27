import os
import glob
import re
import json
import hashlib
import argparse
from typing import List, Dict, Any

import openai
from dotenv import load_dotenv

from langchain.docstore.document import Document
# Updated import path to avoid deprecation warnings:
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import SKLearnVectorStore

##############################################################################
# 1) ENV / SETUP
##############################################################################
load_dotenv()  # Loads .env with OPENAI_API_KEY if present

# Get API key from environment variable (or set it manually below)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# LLM model for generating usage explanations
LLM_MODEL_NAME = "o3-mini-2025-01-31"  # Or any other specialized model you have access to

# Embedding model (you could also use "text-embedding-ada-002" or a code model)
EMBEDDING_MODEL_NAME = "text-embedding-3-large"

# Base directory to your UI component .ts/.tsx files
BASE_DIR = "/Users/bryanborck/Desktop/Zap/mini-apps-ui-kit/packages/mini-apps-ui-kit-react/src/components"

##############################################################################
# 2) UTILITY: GATHER FILES
##############################################################################
def get_ts_files(base_dir: str) -> List[str]:
    """
    Recursively find all .ts or .tsx files within `base_dir`,
    ignoring 'index.ts' or 'index.tsx'.
    """
    pattern = os.path.join(base_dir, "**/*.ts*")
    all_files = glob.glob(pattern, recursive=True)

    # Exclude index.ts or index.tsx
    ts_files = [
        f for f in all_files
        if not f.endswith("index.tsx") and not f.endswith("index.ts")
    ]
    return ts_files

##############################################################################
# 3) NAIVE COMPONENT SPLITTING
##############################################################################
def split_into_components(file_content: str) -> List[str]:
    """
    Naive approach: split the file content on 'export '.
    Each chunk presumably contains one exported component or function.
    """
    raw_sections = file_content.split("export ")
    chunks = []
    for i, section in enumerate(raw_sections):
        # Skip anything before the first "export"
        if i == 0:
            continue
        chunk = "export " + section.strip()
        # Minimal length threshold (avoid tiny lines)
        if len(chunk) > 30:
            chunks.append(chunk)
    return chunks

##############################################################################
# 4) CHECK IF CHUNK IS A RE-EXPORT (TRIVIAL EXPORT)
##############################################################################
def is_re_export(chunk: str) -> bool:
    """
    Return True if the snippet looks like a trivial re-export.
    e.g. `export * from '...'` or `export { Something } from '...'`.
    """
    # A naive regex that checks if the chunk is exactly a re-export statement.
    # e.g. export * from "foo"
    #      export { Something, Another } from 'bar'
    # We'll allow trailing semicolons/spaces, but no actual code block.
    pattern = r"^export\s+(\*|{[^}]+})\s+from\s+['\"].+['\"].*?$"
    stripped = chunk.strip().rstrip(";")
    return bool(re.match(pattern, stripped))

##############################################################################
# 5) LLM EXPLANATION (OPENAI>=1.0.0)
##############################################################################
def generate_usage_explanation(code_snippet: str, is_overview: bool = False) -> str:
    """
    Call the LLM to produce a short explanation for the snippet.
    If is_overview is True, generate a general overview of the UI kit.
    """
    if is_overview:
        prompt = """You are an expert UI developer.
        
Create a comprehensive overview of the WorldCoin Mini Apps UI Kit that includes:
1. A general introduction to the UI kit's purpose and design philosophy
2. A list of all the available components in the UI kit
3. Basic usage information and installation instructions
4. Any best practices for using the UI kit

IMPORTANT: Please include the following specific installation information in your overview:
- Installation command: `npm i @worldcoin/mini-apps-ui-kit-react@1.0.0-canary.4`
- Make sure to emphasize that users MUST import the CSS file in their layout.tsx or main component file: 
  `import "@worldcoin/mini-apps-ui-kit-react/styles.css";`
- Explain that this CSS import is critical for the components to display correctly

Format the response as a clear, concise guide that will help developers understand the overall value and components available in the WorldCoin Mini Apps UI Kit.
---
Now provide the overview:
"""
    else:
        prompt = f"""You are an expert UI developer.
Given the following TypeScript/React component or function code, produce a concise explanation:
1. What this component does or represents.
2. How to use it in a larger application (props, typical usage, example).
3. Any best practices or limitations.

IMPORTANT: In ALL examples, show importing components from '@worldcoin/mini-apps-ui-kit-react' 
rather than from relative paths. For example:
```tsx
import {{ Button }} from '@worldcoin/mini-apps-ui-kit-react';
```

Code Snippet:
{code_snippet}
---
Now provide the explanation:
"""

    response = openai.chat.completions.create(
        model=LLM_MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
    )
    explanation = response.choices[0].message.content.strip()
    return explanation

##############################################################################
# 6) CREATE & PERSIST LOCAL VECTOR STORE
##############################################################################
def create_vectorstore(documents: List[Document], storage_folder: str):
    """
    Creates an SKLearn vector store from the given documents, using
    the specified embedding model, and persists it as a .parquet file.
    """
    if not documents:
        print("No documents to process. Skipping vector store creation.")
        return None

    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL_NAME,
        openai_api_key=OPENAI_API_KEY
    )
    
    # Create a safer approach - embed texts separately first to validate dimensions
    print(f"Embedding {len(documents)} documents...")
    texts = [doc.page_content for doc in documents]
    metadatas = [doc.metadata for doc in documents]
    
    try:
        # Try to embed a single document first to get expected dimensions
        print("Testing embedding dimensionality...")
        sample_embedding = embeddings.embed_query(texts[0][:8000])  # Limit length to avoid token limits
        expected_dim = len(sample_embedding)
        print(f"Expected embedding dimension: {expected_dim}")
        
        # Process in smaller batches to avoid token limits and to handle errors
        batch_size = 10
        all_embeddings = []
        all_valid_texts = []
        all_valid_metadatas = []
        all_valid_ids = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_metadatas = metadatas[i:i+batch_size]
            
            print(f"Processing batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}...")
            
            # Truncate texts to avoid token limits
            truncated_batch = [text[:8000] for text in batch_texts]  # Limit to 8000 chars per doc
            
            try:
                # Embed each document separately to catch errors
                for j, text in enumerate(truncated_batch):
                    try:
                        embed = embeddings.embed_query(text)
                        # Validate dimension
                        if len(embed) == expected_dim:
                            all_embeddings.append(embed)
                            all_valid_texts.append(batch_texts[j])
                            all_valid_metadatas.append(batch_metadatas[j])
                            # Generate a unique ID for each document
                            unique_id = hashlib.md5(text.encode()).hexdigest()
                            all_valid_ids.append(unique_id)
                        else:
                            print(f"  Skipping document with incorrect embedding dimension: {len(embed)} != {expected_dim}")
                    except Exception as e:
                        print(f"  Error embedding document {i+j}: {str(e)}")
            except Exception as batch_err:
                print(f"  Error processing batch: {str(batch_err)}")
        
        print(f"Successfully embedded {len(all_valid_texts)} documents out of {len(texts)}")
        
        if not all_valid_texts:
            print("No valid embeddings generated. Skipping vector store creation.")
            return None
            
        # Now create vector store manually with valid embeddings
        persist_path = os.path.join(storage_folder, "sklearn_vectorstore_uikit.parquet")
        
        try:
            print("Creating vector store with pre-embedded vectors...")
            
            # Import numpy for array handling
            import numpy as np
            from sklearn.neighbors import NearestNeighbors
            
            # Convert embeddings to numpy array
            embeddings_array = np.array(all_embeddings)
            
            # Create and fit the nearest neighbors model with proper parameters
            nn_model = NearestNeighbors(n_neighbors=4, algorithm="ball_tree")
            nn_model.fit(embeddings_array)
            
            # Create a fresh SKLearnVectorStore
            vs = SKLearnVectorStore(
                embedding=embeddings,  # Still need the embedding object for future queries
                persist_path=persist_path,
                serializer="parquet"
            )
            
            # Manually add the pre-embedded vectors and neighbor model
            vs._embeddings = all_embeddings
            vs._texts = all_valid_texts
            vs._metadatas = all_valid_metadatas or [{}] * len(all_valid_texts)
            vs._ids = all_valid_ids
            vs._neighbors = nn_model
            
            # Persist to disk
            vs.persist()
            print(f"Vector store persisted at: {persist_path}")
            return vs
            
        except Exception as vs_err:
            print(f"Error creating vector store with manual initialization: {str(vs_err)}")
            
            # Fallback attempt: create new documents from valid data
            print("Trying alternative approach with Documents...")
            
            try:
                # Create new documents with valid content
                valid_documents = [
                    Document(page_content=text, metadata=metadata)
                    for text, metadata in zip(all_valid_texts, all_valid_metadatas)
                ]
                
                # Create a simple wrapper for embeddings to avoid re-embedding
                from langchain_core.embeddings.base import Embeddings
                
                class PrecomputedEmbeddings(Embeddings):
                    """Wrapper for precomputed embeddings."""
                    
                    def __init__(self, precomputed_embeddings: Dict[str, List[float]]):
                        self.precomputed = precomputed_embeddings
                        
                    def embed_documents(self, texts: List[str]) -> List[List[float]]:
                        """Return precomputed embeddings for the texts."""
                        return [self.precomputed[text] for text in texts if text in self.precomputed]
                        
                    def embed_query(self, text: str) -> List[float]:
                        """Use the real embedding function for queries."""
                        return embeddings.embed_query(text)
                
                # Create a mapping of text to its embedding
                precomputed = {text: embed for text, embed in zip(all_valid_texts, all_embeddings)}
                custom_embeddings = PrecomputedEmbeddings(precomputed)
                
                # Create vector store with precomputed embeddings
                vs = SKLearnVectorStore.from_documents(
                    documents=valid_documents,
                    embedding=custom_embeddings,  # Use our wrapper with precomputed embeddings
                    persist_path=persist_path,
                    serializer="parquet"
                )
                
                vs.persist()
                print(f"Vector store created and persisted with alternative method.")
                return vs
                
            except Exception as alt_err:
                print(f"Alternative approach also failed: {str(alt_err)}")
                
                # Last resort approach - save embeddings to a file to use directly with another system
                backup_path = os.path.join(storage_folder, "embeddings_backup.npz")
                try:
                    print(f"Saving embeddings to backup file at {backup_path}")
                    np.savez(
                        backup_path,
                        embeddings=np.array(all_embeddings),
                        texts=np.array(all_valid_texts),
                        metadatas=np.array([json.dumps(m) for m in all_valid_metadatas])
                    )
                    print("Embeddings saved for manual processing.")
                except Exception as save_err:
                    print(f"Error saving embeddings backup: {str(save_err)}")
                
                raise alt_err
        
    except Exception as e:
        print(f"Error creating vector store: {str(e)}")
        # Optional - Save the texts and metadata for later debugging
        debug_path = os.path.join(storage_folder, "failed_docs_uikit.json")
        try:
            with open(debug_path, "w", encoding="utf-8") as f:
                json.dump([{"content": t[:500], "metadata": m} for t, m in zip(texts, metadatas)], f, indent=2)
            print(f"Debug info saved to {debug_path}")
        except Exception as save_err:
            print(f"Error saving debug info: {str(save_err)}")
        return None

##############################################################################
# 7) MAIN WORKFLOW
##############################################################################
def build_vectorstore_from_json(json_file_path: str, storage_folder: str):
    """
    Build a vector store from an existing explanations JSON file
    without re-parsing or generating explanations.
    """
    print(f"Building vector store from existing explanations: {json_file_path}")
    
    try:
        with open(json_file_path, "r", encoding="utf-8") as f:
            explanations_list = json.load(f)
        
        print(f"Loaded {len(explanations_list)} component explanations from JSON")
        
        # Convert to langchain Document objects
        all_docs = []
        for item in explanations_list:
            file_path = item.get("file_path", "unknown")
            code_snippet = item.get("code_snippet", "")
            explanation = item.get("explanation", "")
            
            # Skip entries with missing data
            if not code_snippet or not explanation:
                continue
                
            # Format same as in main workflow
            doc_content = f"{code_snippet}\n\n=== USAGE EXPLANATION ===\n{explanation}"
            doc = Document(
                page_content=doc_content,
                metadata={"file_path": file_path}
            )
            all_docs.append(doc)
            
        print(f"Created {len(all_docs)} document objects for vector store")
        
        # Use the same function to create vectorstore
        return create_vectorstore(all_docs, storage_folder)
        
    except Exception as e:
        print(f"Error building vector store from JSON: {str(e)}")
        return None

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Build a vector store from UI components")
    parser.add_argument(
        "--from-json",
        dest="from_json",
        help="Build vector store from existing explanations_ukit.json file instead of regenerating explanations",
        action="store_true"
    )
    parser.add_argument(
        "--json-path",
        dest="json_path",
        help="Path to explanations_uikit.json file (defaults to script directory)",
        default=None
    )
    
    args = parser.parse_args()
    script_folder = os.path.dirname(os.path.abspath(__file__))
    
    # Create docs/uikit folder if it doesn't exist
    docs_folder = os.path.join(script_folder, "docs", "uikit")
    os.makedirs(docs_folder, exist_ok=True)
    
    # Using existing explanations_uikit.json
    if args.from_json:
        json_path = args.json_path if args.json_path else os.path.join(docs_folder, "explanations_uikit.json")
        if os.path.exists(json_path):
            build_vectorstore_from_json(json_path, docs_folder)
            return
        else:
            print(f"Error: Explanations file not found at {json_path}")
            print("Running full extraction and explanation instead.")
    
    # Original functionality for component extraction and explanation
    ts_files = get_ts_files(BASE_DIR)
    print(f"Found {len(ts_files)} TS/TSX files (excluding index).")

    all_docs = []
    # We'll accumulate raw code + explanations in a list, then dump to JSON
    explanations_list: List[Dict[str, Any]] = []

    # First, create a general overview of the UI kit
    print("Generating general UI kit overview...")
    try:
        overview_explanation = generate_usage_explanation("", is_overview=True)
        # Create a special doc for the overview
        doc_content = f"UI Kit Overview\n\n=== USAGE EXPLANATION ===\n{overview_explanation}"
        overview_doc = Document(
            page_content=doc_content,
            metadata={"file_path": "UI_KIT_OVERVIEW"}
        )
        all_docs.append(overview_doc)
        
        # Add to explanations list
        explanations_list.append({
            "file_path": "UI_KIT_OVERVIEW",
            "code_snippet": "UI Kit Overview",
            "explanation": overview_explanation
        })
        print("UI kit overview generated successfully.")
    except Exception as e_overview:
        print(f"Error generating UI kit overview: {e_overview}")

    for file_path in ts_files:
        print(f"Processing file: {file_path}")
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            chunks = split_into_components(content)

            # If no meaningful chunks, skip
            if not chunks:
                print(f"  - Skipped (no valid 'export' chunks found).")
                continue

            doc_count_for_file = 0
            for chunk in chunks:
                # Skip trivial re-exports
                if is_re_export(chunk):
                    continue

                try:
                    explanation = generate_usage_explanation(chunk)
                except Exception as e_llm:
                    print(f"  ! Error generating explanation for chunk: {e_llm}")
                    continue

                # Combine snippet + explanation
                doc_content = f"{chunk}\n\n=== USAGE EXPLANATION ===\n{explanation}"
                doc = Document(
                    page_content=doc_content,
                    metadata={"file_path": file_path}
                )
                all_docs.append(doc)
                doc_count_for_file += 1

                # Also store in explanations_list for offline reference
                # We'll store the raw code snippet + explanation separately
                # in a JSON structure so we can re-use it if needed.
                explanations_list.append({
                    "file_path": file_path,
                    "code_snippet": chunk,
                    "explanation": explanation
                })

            if doc_count_for_file == 0:
                print(f"  - Skipped (only re-exports or invalid chunks).")
        except Exception as e_file:
            print(f"Error processing file {file_path}: {str(e_file)}")

    print(f"Total documents prepared: {len(all_docs)}")

    # Save all code+explanations to a JSON file for future reuse
    script_folder = os.path.dirname(os.path.abspath(__file__))
    docs_folder = os.path.join(script_folder, "docs", "uikit")
    explanations_path = os.path.join(docs_folder, "explanations_uikit.json")
    try:
        with open(explanations_path, "w", encoding="utf-8") as f:
            json.dump(explanations_list, f, indent=2)
        print(f"All code snippets + explanations saved to {explanations_path}")
    except Exception as e_json:
        print(f"Error saving explanations to JSON: {e_json}")

    # Finally, build the vector store if we have docs
    if all_docs:
        create_vectorstore(all_docs, storage_folder=docs_folder)
        print("Done! Your local SKLearn Vector Store is ready.")
    else:
        print("No valid UI component documents found. Vector store not created.")

if __name__ == "__main__":
    main()