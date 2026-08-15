# Step-by-Step Run & Test Guide

This document provides a clear walkthrough for setting up, launching, and validating the **NVIDIA-powered FAISS RAG Agent Product** on your local machine.

---

## GitHub Repository Connection (Git Setup)

To link this local workspace to your GitHub repository ([RAG-Evaluation-Framework](https://github.com/nidhijain16/RAG-Evaluation-Framework)) and push your code, open a Git-enabled terminal in this folder and run:

```bash
# 1. Initialize local Git repository
git init

# 2. Add the remote repository origin
git remote add origin https://github.com/nidhijain16/RAG-Evaluation-Framework.git

# 3. Add files and make initial commit
git branch -M main
git add .
git commit -m "Initial commit: Unified RAG endpoint, Gradio UI, FAISS index generator, and setup guide"

# 4. Push files to GitHub
git push -u origin main
```

---

## Prerequisites

Before starting, make sure you have:
1. **Astral's `uv` tool** installed. (We verified that `uv` is available on your system path).
2. An **NVIDIA API Key** (starts with `nvapi-`). You can get one from the [NVIDIA API Catalog](https://build.nvidia.com/).

---

## Step-by-Step Launch Procedure

Follow these steps in order:

### Step 1: Install Dependencies
Open a PowerShell terminal in the workspace directory (`e:\Fraunhofer2024-26\Nvidia\RAG Agents\resources\task`) and run:
```bash
uv pip install -r requirements.txt
```

### Step 2: Configure Your API Key
Choose **one** of these options:
* **Option A (Recommended)**: Create a file named `.env` in the root workspace directory with the content:
  ```env
  NVIDIA_API_KEY=nvapi-your-key-here
  ```
* **Option B**: Skip this step and paste your API key directly into the **API Key Override** textbox in the browser UI when you launch the client.

### Step 3: Build the Vector Store Index
Run the index builder script to fetch the Arxiv publications, chunk them, generate embeddings, and build the local FAISS database:
```bash
uv run build_index.py
```
* **Expected Output:** You should see console outputs showing each paper being loaded, text being chunked, and finally:
  `Successfully built and saved docstore_index!`
  This creates a local `docstore_index` directory and compiles `docstore_index.tgz`.

### Step 4: Run the Backend API Server
Launch the LangServe FastAPI server:
```bash
uv run python server_app.py
```
* **Expected Output:** Uvicorn starts on port `9012`.
* **API Documentation:** You can view the live Swagger documentation by opening [http://localhost:9012/docs](http://localhost:9012/docs) in your browser.

### Step 5: Run the Frontend UI Client
In a **separate** terminal tab/window in the same workspace directory, launch the UI client:
```bash
uv run python ui_client.py
```
* **Expected Output:** Gradio interface launches on port `8090`.
* **Browser Access:** Open your browser and navigate to: [http://localhost:8090](http://localhost:8090).

---

## How to Test the System

Here are the scenarios to verify everything is working as expected:

### Scenario 1: Interactive Chat Testing (UI)
1. Navigate to [http://localhost:8090](http://localhost:8090).
2. Check the **System Status** panel on the left sidebar:
   * **Backend Server**: Should be `🟢 Online`.
   * **FAISS Index**: Should be `🟢 Loaded`.
   * **NVIDIA API Key**: Should be `🟢 Configured`.
   *(If you did not use a `.env` file, paste your `nvapi-...` key into the input box and click **Apply Key** to turn the status indicator green).*
3. **Ask an In-Corpus Question**:
   * Type: *`How does MRKL architecture combine LLMs with external tools?`* and press Send.
   * **Verify**: The chatbot should generate a highly detailed answer based on the MRKL paper and append `Sources used: - MRKL Paper` (or similar) at the bottom.
4. **Test Dynamic Configuration**:
   * Adjust the **Retrieval Size (k Chunks)** slider on the left sidebar to `1`.
   * Ask the question again.
   * **Verify**: Look at the console logs of the backend server. The logs should print that only `1` document was retrieved, resulting in faster latency.
5. **Test Fallback Guardrails**:
   * Ask an out-of-corpus question like: *`What is the recipe for baking chocolate chip cookies?`*
   * **Verify**: The agent should respond with: *"I couldn’t find relevant documents in the corpus for this question."* and list no sources.

---

### Scenario 2: Programmatic API Testing (Terminal)
You can test the FastAPI endpoints directly from a PowerShell or Git Bash terminal using `curl`:

1. **Verify Health Check Endpoint**:
   ```bash
   curl -X GET "http://localhost:9012/health"
   ```
   * **Expected Response**:
     `{"status":"ok","docstore_loaded":true,"nvidia_key_configured":true}`

2. **Query the Unified RAG Route**:
   ```bash
   curl -X POST "http://localhost:9012/rag/invoke" \
        -H "Content-Type: application/json" \
        -d '{"input": {"input": "What is the main objective of the Attention Is All You Need paper?", "k": 3}}'
   ```
   * **Expected Response**: A JSON payload wrapping the response structure:
     ```json
     {
       "output": {
         "output": "The main objective of the Attention Is All You Need paper is to propose the Transformer architecture...",
         "sources": ["Attention Is All You Need Paper"]
       }
     }
     ```

3. **Query with Non-Corpus Question (Fallback validation)**:
   ```bash
   curl -X POST "http://localhost:9012/rag/invoke" \
        -H "Content-Type: application/json" \
        -d '{"input": {"input": "What is the capital of Italy?", "k": 2}}'
   ```
   * **Expected Response**:
     ```json
     {
       "output": {
         "output": "I couldn’t find relevant documents in the corpus for this question.",
         "sources": []
       }
     }
     ```
