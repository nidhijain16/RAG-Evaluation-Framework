import os
import tarfile
import ssl
import arxiv
ssl._create_default_https_context = ssl._create_unverified_context
arxiv.arxiv.Client.query_url_format = 'https://export.arxiv.org/api/query?{}'

from langchain_community.document_loaders import ArxivLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

# Load env variables from .env if present
load_dotenv()

def build_and_save_index():
    print("Initializing Arxiv parser...")
    # List of papers: Attention, BERT, RAG, MRKL, Mistral, LLM-as-a-Judge, and a recent paper like GraphRAG
    paper_ids = [
        "1706.03762",  # Attention Is All You Need
        "1810.04805",  # BERT
        "2005.11401",  # RAG
        "2205.00445",  # MRKL
        "2310.06825",  # Mistral
        "2306.05685",  # LLM-as-a-Judge
        "2404.16130",  # GraphRAG (satisfies recent paper requirement)
    ]
    
    docs = []
    for pid in paper_ids:
        print(f"Fetching paper {pid} from Arxiv...")
        try:
            loader = ArxivLoader(query=pid, load_max_docs=1)
            docs.extend(loader.load())
        except Exception as e:
            print(f"Warning: Failed to fetch paper {pid}: {e}")
            
    if not docs:
        print("Error: No documents were fetched. Check your network connection.")
        return
        
    print(f"Loaded {len(docs)} papers. Chunking documents...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(docs)
    print(f"Created {len(chunks)} chunks.")
    
    print("Generating embeddings and constructing FAISS index...")
    # Verify NVIDIA_API_KEY
    if not os.environ.get("NVIDIA_API_KEY"):
        print("WARNING: NVIDIA_API_KEY environment variable is not set.")
        raise ValueError("NVIDIA_API_KEY environment variable is missing. Please set it or place it in a .env file.")
        
    embedder = NVIDIAEmbeddings(model="nvidia/nv-embed-v1", truncate="END")
    docstore = FAISS.from_documents(chunks, embedder)
    
    print("Saving FAISS index locally...")
    docstore.save_local("docstore_index")
    
    # Also compress it as docstore_index.tgz for evaluation compatibility
    print("Compressing index to docstore_index.tgz...")
    with tarfile.open("docstore_index.tgz", "w:gz") as tar:
        tar.add("docstore_index", arcname="docstore_index")
        
    print("Index build completed successfully!")

if __name__ == "__main__":
    build_and_save_index()
