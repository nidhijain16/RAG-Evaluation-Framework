import os
import requests
import gradio as gr

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:9012")

def check_backend_status():
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=2)
        if response.status_code == 200:
            data = response.json()
            status_md = "🟢 **Backend Server:** Online\n\n"
            if data.get("docstore_loaded"):
                status_md += "🟢 **FAISS Index:** Loaded\n\n"
            else:
                status_md += "🔴 **FAISS Index:** Missing / Unloaded\n\n"
                
            if data.get("nvidia_key_configured"):
                status_md += "🟢 **NVIDIA API Key:** Configured\n"
            else:
                status_md += "🔴 **NVIDIA API Key:** Missing\n"
            return status_md, data.get("docstore_loaded", False), data.get("nvidia_key_configured", False)
    except Exception as e:
        pass
    return "🔴 **Backend Server:** Offline (Is it running on port 9012?)\n\n🔴 **FAISS Index:** Unknown\n\n🔴 **NVIDIA API Key:** Unknown", False, False

def apply_api_key(key):
    if not key or not key.strip().startswith("nvapi-"):
        return "⚠️ Key must start with 'nvapi-'"
    try:
        response = requests.post(f"{BACKEND_URL}/set_key", json={"key": key}, timeout=5)
        if response.status_code == 200:
            res = response.json()
            if res.get("status") == "success":
                return "✅ API Key set successfully!"
            else:
                return f"❌ Error: {res.get('message')}"
        return f"❌ Server returned code {response.status_code}"
    except Exception as e:
        return f"❌ Connection Error: {e}"

def ask_question(message, history, k_val, corpus_val, key_override):
    # If the user supplied a key override, apply it first
    if key_override and key_override.strip().startswith("nvapi-"):
        apply_api_key(key_override)
        
    payload = {
        "input": message,
        "k": int(k_val),
        "corpus": corpus_val
    }
    
    try:
        response = requests.post(f"{BACKEND_URL}/rag/invoke", json={"input": payload}, timeout=60)
        # LangServe /invoke route wraps output in {"output": ...}
        # In our server_app, rag_flow returns {"output": "...", "sources": [...]}
        # So LangServe returns {"output": {"output": "...", "sources": [...]}}
        if response.status_code == 200:
            res_json = response.json()
            output_data = res_json.get("output", {})
            if isinstance(output_data, dict):
                answer = output_data.get("output", "No response text found.")
                sources = output_data.get("sources", [])
            else:
                answer = str(output_data)
                sources = []
                
            if sources:
                answer += "\n\n**Sources used:**\n" + "\n".join([f"- {s}" for s in sources])
                
            history.append((message, answer))
            return history, "", ""
        else:
            err_msg = f"Error: Server returned status code {response.status_code}"
            try:
                err_msg += f"\nDetails: {response.text}"
            except:
                pass
            history.append((message, err_msg))
            return history, "", ""
    except Exception as e:
        err_msg = f"Connection error: Could not connect to backend at {BACKEND_URL}/rag/invoke.\nDetails: {e}"
        history.append((message, err_msg))
        return history, "", ""

def init_ui():
    initial_status, index_loaded, key_configured = check_backend_status()
    
    # Custom CSS for modern styling (Glassmorphism, Indigo theme)
    custom_css = """
    .gradio-container {
        font-family: 'Inter', 'Outfit', sans-serif !important;
    }
    .header-box {
        background: linear-gradient(135deg, #1e1b4b 0%, #311042 100%);
        padding: 25px;
        border-radius: 12px;
        color: white;
        margin-bottom: 20px;
        border: 1px solid #3730a3;
    }
    .sidebar-panel {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 15px;
    }
    """
    
    with gr.Blocks(css=custom_css, title="NVIDIA FAISS RAG Agent") as demo:
        # Header Row
        with gr.Box(elem_classes="header-box"):
            gr.HTML("""
            <div style="display: flex; align-items: center; justify-content: space-between;">
                <div>
                    <h1 style="margin: 0; color: #818cf8; font-size: 28px; font-weight: 800;">NVIDIA-Powered RAG Agent</h1>
                    <p style="margin: 5px 0 0 0; color: #cbd5e1; font-size: 14px;">FAISS Vector Store Retrieval & meta/llama-3.1-8b-instruct Generation</p>
                </div>
                <div style="background-color: rgba(255,255,255,0.1); padding: 8px 15px; border-radius: 20px; border: 1px solid rgba(255,255,255,0.2);">
                    <span style="font-weight: 600; color: #a5b4fc; font-size: 13px;">Portfolio Demo</span>
                </div>
            </div>
            """)
            
        with gr.Row():
            # Left Sidebar - Configuration & Health
            with gr.Column(scale=1, min_width=300):
                with gr.Box(elem_classes="sidebar-panel"):
                    gr.Markdown("### ⚙️ System Status")
                    status_display = gr.Markdown(value=initial_status)
                    refresh_btn = gr.Button("🔄 Refresh Status", size="sm")
                    
                with gr.Box(elem_classes="sidebar-panel"):
                    gr.Markdown("### 🛠️ Configuration")
                    k_slider = gr.Slider(minimum=1, maximum=10, value=4, step=1, label="Retrieval Size (k Chunks)")
                    corpus_select = gr.Dropdown(choices=["default"], value="default", label="Knowledge Corpus")
                    
                with gr.Box(elem_classes="sidebar-panel"):
                    gr.Markdown("### 🔑 API Key Override")
                    gr.Markdown("*Set or update the key dynamically in the backend server memory without restarting:*")
                    api_key_box = gr.Textbox(placeholder="nvapi-...", type="password", show_label=False)
                    apply_btn = gr.Button("Apply Key", size="sm")
                    key_status = gr.Markdown(value="")
            
            # Right Area - Chatbot
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(label="Conversation", height=500)
                with gr.Row():
                    txt = gr.Textbox(
                        show_label=False,
                        placeholder="Ask a question about Attention, BERT, RAG, MRKL, Mistral, or GraphRAG...",
                        scale=9
                    )
                    submit_btn = gr.Button("Send", variant="primary", scale=1)
                    
        # Event bindings
        refresh_btn.click(fn=lambda: check_backend_status()[0], inputs=None, outputs=status_display)
        apply_btn.click(fn=apply_api_key, inputs=api_key_box, outputs=key_status)
        
        submit_event = submit_btn.click(
            fn=ask_question,
            inputs=[txt, chatbot, k_slider, corpus_select, api_key_box],
            outputs=[chatbot, txt, key_status]
        )
        txt.submit(
            fn=ask_question,
            inputs=[txt, chatbot, k_slider, corpus_select, api_key_box],
            outputs=[chatbot, txt, key_status]
        )
        
    return demo

if __name__ == "__main__":
    demo = init_ui()
    demo.launch(server_name="0.0.0.0", server_port=8090, share=False)
