import os
import time
from dotenv import load_dotenv
from tqdm import tqdm
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Setup Environment
load_dotenv()
_script_dir = os.path.dirname(os.path.abspath(__file__))
directory_path = os.path.join(_script_dir, "data") # Ensure this is the correct path to your .docx files
persist_db_path = os.path.join(_script_dir, "vectorstore")
collection_name = "interview_documents"

# 2. Load Documents
def loader_textdocs():
    print("[1/3] Loading documents from disk...")
    loader = DirectoryLoader(
        directory_path,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True,
    )
    docs = loader.load()
    print(f"      Loaded {len(docs)} document(s).\n")
    return docs

# 3. Chunk Documents based on its structure
def split_documents(documents):
    print("[2/3] Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        is_separator_regex=True,
        separators=[
            r"\n\n+",  # Paragraph breaks
            r"\n+",    # Line breaks
            r"\s+",    # Whitespace
        ],
    )
    all_splits = text_splitter.split_documents(documents)
    print(f"Created {len(all_splits)} chunk(s).\n")
    return all_splits

# 4. The Correct Batch Loading Logic
def ingest_data():
    documents = loader_textdocs()
    all_splits = split_documents(documents)

    if not all_splits:
        print("No documents found to split. Check your data path.")
        return

    # 5. Initialize Embeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    print(f"[3/3] Embedding and storing {len(all_splits)} chunks into ChromaDB collection '{collection_name}'...")

    batch_size = 20

    # Initialize Chroma with the FIRST batch directly.
    # This ensures the collection is created and the dimension is set correctly.
    for attempt in range(5):
        try:
            vectorstore = Chroma.from_documents(
                documents=all_splits[:batch_size],
                embedding=embeddings,
                persist_directory=persist_db_path,
                collection_name=collection_name,
            )
            break
        except Exception as e:
            if attempt == 4:
                print(f"\nError on first batch: {e}")
                return None
            time.sleep(5)

    # Use tqdm for the REMAINING splits
    for i in tqdm(range(batch_size, len(all_splits), batch_size), desc="      Embedding batches"):
        batch = all_splits[i : i + batch_size]
        for attempt in range(5):
            try:
                vectorstore.add_documents(batch)
                break
            except Exception as e:
                if attempt == 4:
                    print(f"\nError at batch {i}: {e}")
                    return vectorstore
                time.sleep(5)

    print(f"\nIngestion complete. Data stored in '{persist_db_path}' under collection '{collection_name}'.")
    return vectorstore

if __name__ == "__main__":
    vectorstore = ingest_data()
