# NVIDIA RAG Agent — Complete Run & Test Guide

A step-by-step guide for setting up, launching, and benchmarking the
**NVIDIA FAISS RAG Agent** with the integrated **AgentEval-Latency-Suite**.

---

## Project Architecture

```
task/
├── server_app.py        ← FastAPI backend  (port 9012)
├── ui_client.py         ← Gradio frontend  (port 8090)
├── build_index.py       ← FAISS index builder (run once)
├── latency_helper.py    ← Bridge: server ↔ latency submodule
├── latency_suite/       ← Git submodule: AgentEval-Latency-Suite
│   ├── benchmarks/
│   │   └── latency_test.py   ← Async latency benchmarker (NVIDIA NIM)
│   └── agents/
│       ├── researcher.py     ← LangGraph multi-agent example
│       └── architect.py
├── .env                 ← NVIDIA API key (never commit to git)
├── docstore_index/      ← FAISS vector store (built by build_index.py)
├── latency_results.json ← Cached benchmark output (auto-generated)
└── requirements.txt
```

### Ports at a glance

| Service         | URL                             |
|-----------------|---------------------------------|
| FastAPI Server  | http://localhost:9012           |
| Swagger UI      | http://localhost:9012/docs      |
| Gradio UI       | http://localhost:8090           |
| Health endpoint | http://localhost:9012/health    |
| Latency endpoint| http://localhost:9012/latency   |

---

## Prerequisites

- **Python 3.10+** and **[`uv`](https://docs.astral.sh/uv/)** installed
- An **NVIDIA API Key** starting with `nvapi-`  
  → Get one free at [build.nvidia.com](https://build.nvidia.com/)
- Internet access for the first run (ArXiv paper download + NVIDIA API calls)

---

## One-Time Setup

### Step 1 — Install all dependencies

Open a **PowerShell** terminal in the project root
(`e:\Fraunhofer2024-26\Nvidia\RAG Agents\resources\task`):

```powershell
uv pip install -r requirements.txt
```

> All packages (`fastapi`, `langchain`, `faiss-cpu`, `gradio`, `httpx`, etc.)
> will be installed into the active virtual environment.

---

### Step 2 — Configure your NVIDIA API Key

A `.env` file already exists in the project root. Open it and ensure it contains:

```env
NVIDIA_API_KEY=nvapi-your-real-key-here
```

> **Note:** The `.env` file is listed in `.gitignore` — it will never be committed.
> If you prefer, you can skip this step and paste the key directly into the
> **🔑 API Key Override** accordion in the Gradio UI after launch.

---

### Step 3 — Build the FAISS Vector Index *(run once)*

This fetches 7 ArXiv papers, chunks them, generates NVIDIA embeddings, and
saves the local FAISS database to `docstore_index/`.

```powershell
uv run python build_index.py
```

**Expected output:**
```
Initializing Arxiv parser...
Fetching paper 1706.03762 from Arxiv...
...
Loaded 7 papers. Chunking documents...
Created 412 chunks.
Generating embeddings and constructing FAISS index...
Saving FAISS index locally...
Compressing index to docstore_index.tgz...
Index build completed successfully!
```

> ⏱ This takes **2–5 minutes** depending on network speed.  
> Once `docstore_index/` exists you never need to run this again unless you
> want to rebuild with different papers.

---

## Running the System

You need **two terminals** open simultaneously — one for the server, one for the UI.

### Terminal A — Start the Backend Server

```powershell
uv run python server_app.py
```

**Expected output:**
```
INFO:     Started server process [...]
INFO:     Uvicorn running on http://0.0.0.0:9012 (Press CTRL+C to quit)
```

Verify it is healthy:
```powershell
curl http://localhost:9012/health
# Expected: {"status":"ok","docstore_loaded":true,"nvidia_key_configured":true}
```

> Keep this terminal open. Do **not** close it while using the UI.

---

### Terminal B — Start the Gradio UI Client

Open a **second** PowerShell window in the same directory:

```powershell
uv run python ui_client.py
```

**Expected output:**
```
Running on local URL: http://0.0.0.0:8090
```

Open your browser at **[http://localhost:8090](http://localhost:8090)**.

---

## Using the UI

### Status Panel (top right)
The header shows three live indicators:
- **Backend: 🟢 Online** — server is reachable
- **FAISS Index: 🟢 Loaded** — vector store is ready
- **NVIDIA API Key: 🟢 Configured** — embedder/LLM can call the API

Click **🔄 Refresh Status** to re-poll at any time.

### Chatting
1. Type a question in the text box (or click one of the suggestion buttons)
2. Press **Send** or hit **Enter**
3. The answer streams token-by-token with **Sources used** listed below

### Retrieval Controls (left sidebar)
| Control | Description |
|---------|-------------|
| **Retrieval Size (k Chunks)** | How many FAISS chunks to retrieve (1–10) |
| **Knowledge Corpus** | Corpus selector (`default` = all 7 ArXiv papers) |

### API Key Override
If you did not set up `.env`, expand **🔑 NVIDIA API Key Override**, paste your
`nvapi-...` key, and click **Apply Key**. This hot-reloads all NVIDIA clients
without restarting the server.

---

## Running the Latency Benchmark

The latency suite benchmarks LLM streaming performance: TTFT, TPS, and tail
latencies (P50/P95/P99) across 5 concurrent simulated users.

### Option A — From the Gradio UI

1. In the **📈 Performance** accordion (bottom of the chat column), click
   **Refresh Latency**.
2. This triggers the benchmark, stores results to `latency_results.json`,
   and displays the summary in the UI.

### Option B — From the terminal (standalone)

```powershell
# Run directly from the latency_suite/ directory:
cd latency_suite
uv run python -m benchmarks.latency_test
cd ..
```

**Expected output (NVIDIA NIM mode):**
```
Starting benchmark [NVIDIA NIM (meta/llama-3.1-8b-instruct)]: 5 concurrent users...

======================================================================
User ID    | TTFT (ms)   | Total (ms)  | TPS      | Status
----------------------------------------------------------------------
0          |     312.44 |    2847.23 |    14.28 | success
1          |     289.11 |    2723.45 |    15.92 | success
...
----------------------------------------------------------------------
Mean       |     298.00 |            |    14.85 |
P50 (Median|     295.50 | (Tail Latency Analysis)
P95        |     330.10 | (Production Threshold)
P99        |     341.20 | (Extreme Case)
======================================================================
```

**If `NVIDIA_API_KEY` is missing** — the benchmark auto-falls back to
**MOCK mode** with simulated timings (useful for CI / offline testing):

```
[WARNING] NVIDIA_API_KEY is not set — running in MOCK mode.
Starting benchmark [MOCK]: 5 concurrent users...
```

### Option C — Via the REST API

```powershell
# Trigger a fresh benchmark and get results:
curl -X GET "http://localhost:9012/latency"
```

> ⚠️ If no cached results exist yet, this returns:
> `{"error": "No cached latency results. Run benchmark first."}`
> Run Option A or B first, then this endpoint returns the cached JSON.

---

## API Endpoint Reference

All endpoints are documented interactively at **http://localhost:9012/docs**.

### `GET /health`
Returns server status, FAISS load state, and API key presence.
```json
{"status": "ok", "docstore_loaded": true, "nvidia_key_configured": true}
```

### `POST /rag/invoke`
Full RAG pipeline: retrieve → reorder → generate.
```powershell
curl -X POST "http://localhost:9012/rag/invoke" `
     -H "Content-Type: application/json" `
     -d '{"input": {"input": "What is the Transformer architecture?", "k": 4}}'
```
**Response:**
```json
{
  "output": {
    "output": "The Transformer architecture relies on a self-attention mechanism...",
    "sources": ["Attention Is All You Need"]
  }
}
```

### `POST /rag/stream`
Same as `/rag/invoke` but returns a Server-Sent Events stream (used by the UI).

### `GET /retriever/invoke`
Retrieves raw document chunks without generating an answer.
```powershell
curl -X POST "http://localhost:9012/retriever/invoke" `
     -H "Content-Type: application/json" `
     -d '{"input": "attention mechanism"}'
```

### `POST /basic_chat/invoke`
Direct pass-through to `meta/llama-3.1-8b-instruct` with no RAG context.
```powershell
curl -X POST "http://localhost:9012/basic_chat/invoke" `
     -H "Content-Type: application/json" `
     -d '{"input": "Hello, what can you do?"}'
```

### `POST /set_key`
Hot-reload the NVIDIA API key without restarting the server.
```powershell
curl -X POST "http://localhost:9012/set_key" `
     -H "Content-Type: application/json" `
     -d '{"key": "nvapi-your-key-here"}'
```

### `GET /latency`
Return the latest cached benchmark results JSON.

---

## Validation Checklist

Run through these to confirm everything works end-to-end:

| # | Test | Expected result |
|---|------|-----------------|
| 1 | `curl http://localhost:9012/health` | `{"status":"ok","docstore_loaded":true,...}` |
| 2 | Ask **"How does MRKL architecture combine LLMs?"** | Detailed answer + sources from MRKL paper |
| 3 | Ask **"What is the recipe for chocolate cake?"** | *"I couldn't find relevant documents..."* |
| 4 | Set k=1, ask same question | Server log shows `Docs Retrieved: 1` |
| 5 | Click **Refresh Latency** in UI | Summary table appears with TTFT/TPS/P50/P95/P99 |
| 6 | `curl http://localhost:9012/latency` | JSON with `per_user` list and `summary` stats |
| 7 | Check `latency_results.json` exists | File created in `task/` directory |

---

## Troubleshooting

### `docstore_loaded: false` on health check
→ Run `build_index.py` first. If it already ran, check that `docstore_index/`
exists in the project root and re-start the server.

### `NVIDIA API Key: 🔴 Missing`
→ Add your key to `.env` and restart both processes, or use the API Key Override
accordion in the UI.

### `Connection error` in the Gradio UI
→ Confirm the server is running in Terminal A. The UI requires the server on
port `9012`. Check with `curl http://localhost:9012/health`.

### Latency benchmark returns `[WARNING] NVIDIA_API_KEY is not set`
→ The key isn't reaching the subprocess. Confirm `.env` exists in `task/`
with `NVIDIA_API_KEY=nvapi-...` and that `python-dotenv` is installed.

### `ModuleNotFoundError: No module named 'benchmarks'`
→ Run the benchmark from inside `latency_suite/` directory, not from `task/`.
The `benchmarks/` folder needs to be on the Python path.

---

## Submodule Reference

The `latency_suite/` directory is a Git submodule.

```powershell
# Clone the repo fresh with submodule populated:
git clone --recurse-submodules https://github.com/nidhijain16/RAG-Evaluation-Framework.git

# Update the submodule to the latest commit:
git submodule update --remote latency_suite

# Check submodule status:
git submodule status
```

---

*Last updated: 2026-08-16*
