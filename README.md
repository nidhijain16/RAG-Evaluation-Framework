# NVIDIA-Powered FAISS RAG Agent Product

[![GitHub Repository](https://img.shields.io/badge/GitHub-RAG--Evaluation--Framework-blue?logo=github)](https://github.com/nidhijain16/RAG-Evaluation-Framework)

This repository is connected to the GitHub repository [RAG-Evaluation-Framework](https://github.com/nidhijain16/RAG-Evaluation-Framework) and houses a portfolio-ready **Retrieval-Augmented Generation (RAG)** application. It integrates the **NVIDIA AI Foundation Endpoints** with a local **FAISS Vector Store** to answer complex questions over modern AI literature (e.g., Attention Is All You Need, BERT, RAG, MRKL, Mistral, and GraphRAG publications) with source-backed citation tracking.

## Technical Architecture

The RAG application is architected as a set of modular services communicating over standard REST APIs, orchestrated with **LangChain (LCEL)**, and deployed via **FastAPI** and **LangServe**:

```
                                +-----------------------------------+
                                |            User Client            |
                                |  (Gradio Chat Interface / cURL)   |
                                +-----------------+-----------------+
                                                  |
                                                  |  1. POST /rag/invoke (input, k)
                                                  v
                                +-----------------------------------+
                                |       FastAPI / LangServe         |
                                |         Backend Server            |
                                +-----------------+-----------------+
                                                  |
                                                  |  2. Semantic Search (Similarity Search)
                                                  v
                                +-----------------------------------+
                                |        FAISS Vector Store         |
                                |       (docstore_index/ db)        |
                                +-----------------+-----------------+
                                                  |
                                                  |  3. Retrieve Chunks
                                                  v
                                +-----------------------------------+
                                |       LongContextReorder          |
                                |  (Combat 'Lost in the Middle')    |
                                +-----------------+-----------------+
                                                  |
                                                  |  4. Structured Context
                                                  v
                                +-----------------------------------+
                                |        NVIDIA LLM NIM             |
                                |  (meta/llama-3.1-8b-instruct)     |
                                +-----------------+-----------------+
                                                  |
                                                  |  5. Answer Synthesis
                                                  v
                                +-----------------------------------+
                                |         Output Response           |
                                |   (Text Answer + Sources List)    |
                                +-----------------------------------+
```

### Key Highlights
1. **Dynamic Configuration (`k` Tuning)**: Exposes the retrieval parameter `k` in the OpenAPI request payload, allowing clients to control latency vs. information recall at runtime.
2. **Context Optimization**: Employs `LongContextReorder` to place the most semantically relevant document chunks at the beginning and end of the prompt window, mitigating the "lost in the middle" LLM processing degradation.
3. **Robust Fallbacks & Guardrails**: Automatically detects empty retrieval scenarios and bypasses LLM inference entirely to output a structured guardrail message (*"I couldn’t find relevant documents in the corpus..."*), saving latency and API token costs.
4. **Interactive Environment Configuration**: Allows updating or setting the `NVIDIA_API_KEY` dynamically in memory at runtime via the `/set_key` endpoint, keeping credentials out of persistent file storage.
5. **Real-time Latency & Request Logging**: Automatically measures and outputs execution latency, retrieved document volume, and queries to stdout for production monitoring.

---

## Setup & Running Guide

Ensure you have Python installed and the fast package manager `uv` (Astral's tool) configured on your machine.

### 1. Install Dependencies
Install all backend and frontend packages via `uv`:
```bash
uv pip install -r requirements.txt
```

### 2. Configure NVIDIA API Key
Create a `.env` file in the root directory and add your NVIDIA AI Foundation API key:
```env
NVIDIA_API_KEY=nvapi-your-actual-api-key-here
```
*(Alternatively, you can paste this key directly into the sidebar of the Gradio web UI after launching.)*

### 3. Generate Knowledge Database Index
To fetch the PDF publications from Arxiv, split them into chunks, generate embeddings, and build the local FAISS vector store, run:
```bash
uv run build_index.py
```
This generates the `docstore_index` database folder and compiles `docstore_index.tgz` for assessment/autograder compatibility.

### 4. Start the Backend API Server
Launch the LangServe FastAPI server:
```bash
uv run python server_app.py
```
The backend server will spin up on **port `9012`**. You can verify and inspect the autogenerated interactive documentation at [http://localhost:9012/docs](http://localhost:9012/docs).

### 5. Launch the Web Interface (UI Client)
Start the Gradio client application in a separate terminal:
```bash
uv run python ui_client.py
```
Open your browser and navigate to the local host address: [http://localhost:8090](http://localhost:8090).

---

## API Documentation

You can communicate with the backend server programmatically:

### 1. Invoke the RAG Agent
Sends a question to the unified `/rag` route:
```bash
curl -X POST "http://localhost:9012/rag/invoke" \
     -H "Content-Type: application/json" \
     -d '{"input": {"input": "How does MRKL combine LLMs with external tools?", "k": 4}}'
```

### 2. Verify Server Health
Queries connection state and checks if the vector store index has loaded successfully:
```bash
curl -X GET "http://localhost:9012/health"
```

---

## Recruiter & Interviewer Demo Checklist

When demonstrating this project live or recording video demos:
- **Test Fallback Guardrails**: Ask an unrelated question (e.g. *"What is the recipe for chocolate chip cookies?"*). Show that the agent gracefully refuses to answer using the exact fallback string instead of hallucinating.
- **Tweak the Retrieval Size (`k`)**: Modify the retrieval size slider in the UI from `4` to `1`. Ask a complex question and show how the source citations decrease, proving active parameter tuning.
- **Dynamic Key Validation**: Wipe the key environment variable, open the UI (showing the red "NVIDIA API Key: Missing" status indicator), paste your key into the text input, and click "Apply" to see it instantly turn green and operational.
