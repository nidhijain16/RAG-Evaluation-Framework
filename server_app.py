# https://python.langchain.com/docs/langserve#server
from fastapi import FastAPI
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langserve import add_routes
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnableLambda
from langchain_community.document_transformers import LongContextReorder
from pydantic import BaseModel, Field
from typing import List, Optional
import os
import time
from dotenv import load_dotenv

# Load env variables from .env if present
load_dotenv()

## Embedder and LLM
embedder = NVIDIAEmbeddings(model="nvidia/nv-embed-v1", truncate="END")
instruct_llm = ChatNVIDIA(model="meta/llama-3.1-8b-instruct") # Updated model name to match notebook 8

app = FastAPI(
  title="LangChain Server",
  version="1.0",
  description="A simple api server using Langchain's Runnable interfaces",
)

## PRE-ASSESSMENT: Basic Chat
add_routes(
    app,
    instruct_llm,
    path="/basic_chat",
)

## ASSESSMENT: Implementation

# 1. Load the vector store
# Note: You must have 'docstore_index' in the current directory.
docstore_loaded = False
if os.path.exists("docstore_index"):
    try:
        docstore = FAISS.load_local("docstore_index", embedder, allow_dangerous_deserialization=True)
        retriever = docstore.as_retriever()
        docstore_loaded = True
    except Exception as e:
        print(f"Error loading docstore_index: {e}")
        retriever = RunnableLambda(lambda x: [])
else:
    print("WARNING: 'docstore_index' not found. Please ensure you have run the setup in notebook 8.")
    # Fallback to avoid crash, but will fail assessment
    retriever = RunnableLambda(lambda x: [])

# 2. Create the generator chain
# The generator expects receiving the input state, but standard LangServe adds routes for Runnables.
# The frontend client sends a dictionary input.
prompt = ChatPromptTemplate.from_template(
    "Answer the question based only on the following context:\n{context}\n\nQuestion: {input}"
)
generator = prompt | instruct_llm | StrOutputParser()

add_routes(
    app,
    generator,
    path="/generator",
)

add_routes(
    app,
    retriever,
    path="/retriever",
)

## PORTFOLIO IMPLEMENTATION: Unified RAG endpoint

# Context formatter helper
def docs2str(docs, title="Document"):
    out_str = ""
    for doc in docs:
        doc_name = getattr(doc, 'metadata', {}).get('Title', title)
        if doc_name:
            out_str += f"[Quote from {doc_name}] "
        if isinstance(doc, dict):
            out_str += doc.get('page_content', doc) + "\n"
        else: 
            out_str += getattr(doc, 'page_content', str(doc)) + "\n"
    return out_str

# Input and Output schemas for OpenAPI/LangServe Swagger UI
class RAGQuery(BaseModel):
    input: str = Field(..., description="The question to ask the RAG agent")
    k: int = Field(default=4, description="Number of documents to retrieve")
    corpus: str = Field(default="default", description="The name of the corpus to search (currently 'default')")

class RAGResponse(BaseModel):
    output: str = Field(..., description="The generated answer from the LLM")
    sources: List[str] = Field(default=[], description="List of source document titles used for reference")

# Combined RAG flow logic (generator for streaming support)
def rag_flow(state):
    start_time = time.time()
    question = state.get("input")
    k = state.get("k", 4)
    
    # Check if docstore is loaded
    if not docstore_loaded:
        print("[RAG Route Log] Query received but docstore_index is not loaded.")
        yield {
            "output": "The document database index is not loaded. Please initialize the FAISS database first.",
            "sources": []
        }
        return
        
    # Retrieve documents dynamically based on k
    docs = docstore.similarity_search(question, k=k)
    
    # If retriever returns an empty list, apply fallback guardrail
    if not docs:
        latency = time.time() - start_time
        print(f"[RAG Route Log] Input: '{question}' | Docs Retrieved: 0 | Latency: {latency:.3f}s (Fallback triggered)")
        yield {
            "output": "I couldn’t find relevant documents in the corpus for this question.",
            "sources": []
        }
        return
        
    # Reorder documents to fight 'lost in the middle' effect
    reordering = LongContextReorder()
    reordered_docs = reordering.transform_documents(docs)
    
    # Convert documents to a combined context string
    context_str = docs2str(reordered_docs)
    
    # Extract source titles
    sources = []
    for doc in docs:
        title = doc.metadata.get("Title", doc.metadata.get("title", "Unknown Source"))
        sources.append(title)
    unique_sources = list(dict.fromkeys(sources))
    
    # Yield sources first, with empty output
    yield {
        "output": "",
        "sources": unique_sources
    }
    
    # Stream the response tokens from the generator
    for chunk in generator.stream({
        "context": context_str,
        "input": question
    }):
        yield {
            "output": chunk,
            "sources": []
        }
        
    latency = time.time() - start_time
    print(f"[RAG Route Log] Input: '{question}' | Docs Retrieved: {len(docs)} | Latency: {latency:.3f}s (Streaming completed)")


# Wire the combined RAG chain with explicit types
rag_chain = RunnableLambda(rag_flow).with_types(input_type=RAGQuery, output_type=RAGResponse)

add_routes(
    app,
    rag_chain,
    path="/rag",
)

# Health endpoint for live monitoring and portfolio visibility
@app.get("/health")
async def health():
    global docstore_loaded
    # Recheck if directory is created in the meantime
    if not docstore_loaded and os.path.exists("docstore_index"):
        try:
            global docstore, retriever
            docstore = FAISS.load_local("docstore_index", embedder, allow_dangerous_deserialization=True)
            retriever = docstore.as_retriever()
            docstore_loaded = True
        except Exception as e:
            print(f"Error loading FAISS index on health check: {e}")
            
    return {
        "status": "ok",
        "docstore_loaded": docstore_loaded,
        "nvidia_key_configured": bool(os.environ.get("NVIDIA_API_KEY"))
    }

class KeyInput(BaseModel):
    key: str = Field(..., description="The NVIDIA API Key starting with nvapi-")

@app.post("/set_key")
async def set_key(payload: KeyInput):
    os.environ["NVIDIA_API_KEY"] = payload.key
    global embedder, instruct_llm, docstore, retriever, docstore_loaded
    try:
        embedder = NVIDIAEmbeddings(model="nvidia/nv-embed-v1", truncate="END")
        instruct_llm = ChatNVIDIA(model="meta/llama-3.1-8b-instruct")
        # Re-initialize docstore if it exists
        if os.path.exists("docstore_index"):
            docstore = FAISS.load_local("docstore_index", embedder, allow_dangerous_deserialization=True)
            retriever = docstore.as_retriever()
            docstore_loaded = True
        return {"status": "success", "message": "NVIDIA API key updated successfully."}
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9012)
