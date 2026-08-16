# RAG Evaluation Framework (NVIDIA FAISS RAG & AgentEval-Latency-Suite)

[![GitHub Repository](https://img.shields.io/badge/GitHub-RAG--Evaluation--Framework-blue?logo=github)](https://github.com/nidhijain16/RAG-Evaluation-Framework)

This repository brings together two main components:
1. **NVIDIA FAISS RAG Agent**: An end-to-end Retrieval-Augmented Generation pipeline integrated with a local FAISS vector store and NVIDIA AI Foundation Endpoints (NIM) for high-throughput embeddings and LLM inference.
2. **AgentEval-Latency-Suite (Submodule)**: A production-grade multi-user latency benchmarker that evaluates LLM streaming performance under concurrent loads, tracking metrics such as Time to First Token (TTFT), Tokens Per Second (TPS), and P50/P95/P99 tail latencies.

---

## Technical Architecture

The unified framework is modularized into independent services communicating over standard REST APIs, orchestrated with **LangChain (LCEL)**, and managed with **FastAPI/LangServe** and **Gradio**:

```
                              +-----------------------------------+
                              |            User Client            |
                              |  (Gradio Chat Interface / cURL)   |
                              +-----------------+-----------------+
                                                |
                     +--------------------------+--------------------------+
                     | 1. Ask Question (Stream)                            | 2. Run Latency Benchmark
                     v                                                     v
      +------------------------------+                      +------------------------------+
      |      FastAPI / LangServe     |                      |      Latency Suite Runner    |
      |         Backend Server       |                      | (benchmarks.latency_test.py) |
      +--------------+---------------+                      +--------------+---------------+
                     |                                                     |
                     | 1.1 similarity_search(k)                            | 2.1 Concurrently Queries
                     v                                                     |     NVIDIA NIM (5 Users)
      +------------------------------+                                     | 
      |      FAISS Vector Store      |                                     |
      |     (docstore_index/ db)     |                                     |
      +--------------+---------------+                                     |
                     |                                                     |
                     | 1.2 Retrieve & Reorder (LongContextReorder)         |
                     v                                                     v
      +------------------------------+                      +------------------------------+
      |       NVIDIA NIM LLM         | <=================== |      latency_results.json    |
      |  (meta/llama-3.1-8b-instruct)|     Cached Timing    |  (Parsed and saved to task)  |
      +--------------+---------------+                      +------------------------------+
                     |
                     | 1.3 Answer Synthesis
                     v
      +------------------------------+
      |       Gradio UI Output       |
      |   (Text Answer + Sources)    |
      +------------------------------+
```

### Key Highlights
1. **Integrated Latency Benchmarking**: Real-time evaluation of streaming endpoints, tracking time-to-first-token (TTFT) and throughput (TPS) per user, with mathematical percentile modeling (P50/P95/P99 tail latencies).
2. **Dynamic Configuration (`k` Tuning)**: Exposes the retrieval parameter `k` in the OpenAPI request payload, allowing clients to control latency vs. information recall at runtime.
3. **Context Optimization**: Employs `LongContextReorder` to place the most semantically relevant document chunks at the beginning and end of the prompt window, mitigating the "lost in the middle" LLM processing degradation.
4. **Robust Fallbacks & Guardrails**: Automatically detects empty retrieval scenarios and bypasses LLM inference entirely to output a structured guardrail message (*"I couldn’t find relevant documents in the corpus..."*), saving latency and API token costs.
5. **Interactive Environment Configuration**: Allows updating or setting the `NVIDIA_API_KEY` dynamically in memory at runtime via the `/set_key` endpoint, keeping credentials out of persistent file storage.
6. **Unified UI Dashboard**: The custom Gradio dark-mode interface integrates RAG conversation, model configurations, real-time backend/FAISS/key health status, and live latency benchmark reporting inside a single UI.

---

## Directory Structure

```
task/
├── server_app.py        # FastAPI / LangServe backend server (port 9012)
├── ui_client.py         # Gradio premium dark-theme UI frontend client (port 8090)
├── build_index.py       # ArXiv publication parser & FAISS index builder
├── latency_helper.py    # Subprocess bridge: Server/UI <-> Latency Suite Submodule
├── latency_results.json # Cached performance benchmark results
├── .env                 # API Credentials (ignored by Git)
└── latency_suite/       # Submodule: AgentEval-Latency-Suite
    ├── benchmarks/
    │   ├── latency_test.py # Multi-user concurrent streaming benchmarker
    │   └── eval_framework.py
    └── agents/
        ├── researcher.py   # Stateful multi-agent graph (LangGraph)
        └── architect.py
```

---

## Setup & Running Guide

Ensure you have Python 3.10+ installed and Astral's fast package manager **`uv`** configured.

### 1. Install Dependencies
Run from the root directory to set up the dependencies in the active virtual environment:
```bash
uv pip install -r requirements.txt
```

### 2. Configure NVIDIA API Key
Create a `.env` file in the root directory and add your NVIDIA API key:
```env
NVIDIA_API_KEY=nvapi-your-actual-api-key-here
```
*(Alternatively, you can paste this key directly into the **🔑 NVIDIA API Key Override** sidebar panel in the Gradio web UI after launching.)*

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
The backend server will spin up on **port `9012`** (binding to `127.0.0.1`). Inspect the autogenerated interactive documentation at [http://127.0.0.1:9012/docs](http://127.0.0.1:9012/docs).

### 5. Launch the Web Interface (UI Client)
Start the Gradio client application in a separate terminal:
```bash
uv run python ui_client.py
```
Navigate your browser to the local host address: [http://127.0.0.1:8090](http://127.0.0.1:8090).

---

## Running the Latency Benchmark

### Standalone (Command Line)
To execute the benchmark directly against the NVIDIA completions endpoint under concurrent load:
```bash
cd latency_suite
python -m benchmarks.latency_test
```
*Note: If no `NVIDIA_API_KEY` is present in the environment, the benchmark falls back to a simulated mock mode for offline testing.*

### Integrated (Gradio UI)
1. Open the UI client at [http://127.0.0.1:8090](http://127.0.0.1:8090).
2. Expand the **📈 Performance** accordion at the bottom.
3. Click the **Refresh Latency** button. The server executes the benchmark in the background, writes to `latency_results.json`, and outputs the stats (mean TTFT, TPS, P50, P95, P99) directly in markdown format.

---

## REST API Reference

- **`GET /health`**: Returns system loading status, FAISS DB presence, and NVIDIA API key validity.
- **`POST /rag/invoke`**: Runs full RAG pipeline (Retrieval -> Reordering -> Generation) and returns answer + sources.
- **`POST /set_key`**: Dynamic runtime loading of the NVIDIA API Key into server memory.
- **`GET /latency`**: Returns the cached performance stats of the latest benchmark run.
- **`POST /basic_chat/invoke`**: Bypasses vector store retrieval to query the LLM directly.
- **`POST /retriever/invoke`**: Returns raw retrieved context chunks matching the search phrase.
