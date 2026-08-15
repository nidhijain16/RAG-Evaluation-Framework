import os
import requests
import gradio as gr
from langserve import RemoteRunnable


BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:9012")

# Custom CSS for modern styling (Dark Mode, custom borders, suggestion buttons)
custom_css = """
.gradio-container {
    background: #0b0f19 !important; /* Dark background */
    color: #f1f5f9 !important;
    max-width: 1200px !important;
    margin: 0 auto !important;
    padding: 20px !important;
    
    /* CSS custom variables overrides to force dark mode rendering */
    --body-background-fill: #0b0f19 !important;
    --block-background-fill: #111827 !important;
    --background-fill-primary: #111827 !important;
    --background-fill-secondary: #1f2937 !important; /* Assistant bubble background */
    --border-color-primary: #1f2937 !important;
    --border-color-secondary: #374151 !important;
    --text-color-primary: #f1f5f9 !important;
    --text-color-secondary: #9ca3af !important;
    --input-background-fill: #0b0f19 !important;
    --input-border-color: #1f2937 !important;
}
.header-row {
    border-bottom: 1px solid #1e2937;
    padding-bottom: 15px;
    margin-bottom: 25px !important;
    align-items: center !important;
}
.header-title-box h1 {
    background: linear-gradient(to right, #818cf8, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.header-status-box {
    background-color: #111827;
    border: 1px solid #1f2937;
    border-radius: 10px;
    padding: 8px 16px;
    font-size: 13px !important;
    color: #9ca3af;
    display: inline-block;
}
.sidebar-panel {
    background-color: #111827 !important;
    border: 1px solid #1f2937 !important;
    border-radius: 12px !important;
    padding: 16px !important;
    margin-bottom: 15px !important;
}
.suggest-btn {
    border-radius: 8px !important;
    font-size: 12px !important;
    text-align: left !important;
    white-space: normal !important;
    height: auto !important;
    padding: 10px 14px !important;
    background-color: #111827 !important;
    border: 1px solid #1f2937 !important;
    color: #9ca3af !important;
    transition: all 0.2s ease !important;
}
.suggest-btn:hover {
    background-color: #1f2937 !important;
    border-color: #4f46e5 !important;
    color: #ffffff !important;
}

/* Chatbot container & message bubble overrides for Gradio 5/6 */
.chatbot, .message-wrap, .bubble-wrap, [data-testid="chatbot"] {
    background-color: #0b0f19 !important;
}

/* User bubbles (Indigo) */
.user, [data-testid="user-message"], .message-row.user-row .message, .message.user-message {
    background-color: #4f46e5 !important; /* Premium Indigo background */
    border: 1px solid #6366f1 !important;
    border-radius: 12px 12px 0 12px !important;
    padding: 10px 14px !important;
}
.user *, [data-testid="user-message"] *, .message-row.user-row .message *, .message.user-message * {
    color: #ffffff !important; /* Pure white text */
}

/* Assistant/Bot bubbles (Dark Gray/Slate) */
.bot, .assistant, [data-testid="bot-message"], .message-row.bot-row .message, .message.bot-message {
    background-color: #1f2937 !important; /* Dark Slate background */
    border: 1px solid #374151 !important;
    border-radius: 12px 12px 12px 0 !important;
    padding: 10px 14px !important;
}
.bot *, .assistant *, [data-testid="bot-message"] *, .message-row.bot-row .message *, .message.bot-message * {
    color: #f1f5f9 !important; /* Clean light gray/white text */
}
"""

# Modern dark indigo theme customization
theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="slate",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "sans-serif"]
).set(
    body_background_fill="*neutral_950",
    body_text_color="*neutral_100",
    block_background_fill="*neutral_900",
    block_border_width="1px",
    block_border_color="*neutral_800",
    input_background_fill="*neutral_950",
    input_border_color="*neutral_800",
    button_secondary_background_fill="*neutral_850",
    button_secondary_background_fill_hover="*neutral_800",
)


def check_backend_status():
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=2)
        if response.status_code == 200:
            data = response.json()
            # Modern horizontal status format
            status_parts = []
            status_parts.append("Backend: 🟢 Online")
            if data.get("docstore_loaded"):
                status_parts.append("FAISS Index: 🟢 Loaded")
            else:
                status_parts.append("FAISS Index: 🔴 Unloaded")
                
            if data.get("nvidia_key_configured"):
                status_parts.append("NVIDIA API Key: 🟢 Configured")
            else:
                status_parts.append("NVIDIA API Key: 🔴 Missing")
            
            status_md = " &nbsp;|&nbsp; ".join(status_parts)
            return status_md, data.get("docstore_loaded", False), data.get("nvidia_key_configured", False)
    except Exception:
        pass
    return "Backend: 🔴 Offline &nbsp;|&nbsp; FAISS: 🔴 Unknown &nbsp;|&nbsp; NVIDIA Key: 🔴 Unknown", False, False

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
    
    # Pre-append the messages to history so we can stream into the assistant slot
    user_msg_idx = len(history)
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": "⌛ Connecting to NVIDIA RAG pipeline..."})
    yield history, "", ""
    
    try:
        # Initialize LangServe RemoteRunnable
        client = RemoteRunnable(f"{BACKEND_URL}/rag")
        
        answer_accum = ""
        sources_accum = []
        
        # Stream the chunks dynamically
        for chunk in client.stream(payload):
            if isinstance(chunk, dict):
                # Update output/sources accumulator
                if "output" in chunk:
                    answer_accum += chunk["output"]
                if "sources" in chunk and chunk["sources"]:
                    sources_accum.extend(chunk["sources"])
                    
                # Format current accumulated answer
                full_answer = answer_accum
                if sources_accum:
                    unique_sources = list(dict.fromkeys(sources_accum))
                    full_answer += "\n\n**Sources used:**\n" + "\n".join([f"- {s}" for s in unique_sources])
                    
                # Update UI history and yield to Gradio
                history[user_msg_idx + 1]["content"] = full_answer or "⚡ Thinking..."
                yield history, "", ""
            else:
                # Fallback if chunk is raw string
                answer_accum += str(chunk)
                history[user_msg_idx + 1]["content"] = answer_accum
                yield history, "", ""
                
    except Exception as e:
        err_msg = f"Connection error: Could not stream from backend.\nDetails: {e}"
        history[user_msg_idx + 1]["content"] = err_msg
        yield history, "", ""


def init_ui():
    initial_status, index_loaded, key_configured = check_backend_status()
    
    with gr.Blocks(title="NVIDIA FAISS RAG Agent") as demo:
        # Header Row
        with gr.Row(elem_classes="header-row"):
            with gr.Column(scale=3, elem_classes="header-title-box"):
                gr.HTML("""
                <div style="display: flex; align-items: center; gap: 12px;">
                    <div style="background: linear-gradient(135deg, #4f46e5 0%, #a855f7 100%); width: 36px; height: 36px; border-radius: 8px; display: flex; align-items: center; justify-content: center;">
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                            <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path>
                        </svg>
                    </div>
                    <div>
                        <h1 style="margin: 0; font-size: 22px; font-weight: 800; letter-spacing: -0.5px;">NVIDIA RAG Agent</h1>
                        <p style="margin: 2px 0 0 0; color: #94a3b8; font-size: 13px;">FAISS Vector Store & meta/llama-3.1-8b-instruct</p>
                    </div>
                </div>
                """)
            with gr.Column(scale=2, min_width=200):
                with gr.Row(elem_classes="header-status-box"):
                    status_display = gr.Markdown(value=initial_status)
                refresh_btn = gr.Button("🔄 Refresh Status", size="sm", variant="secondary")
            
        with gr.Row():
            # Left Column: Configuration Accordions
            with gr.Column(scale=1, min_width=280):
                with gr.Accordion("⚙️ Retrieval Parameters", open=True):
                    k_slider = gr.Slider(minimum=1, maximum=10, value=4, step=1, label="Retrieval Size (k Chunks)")
                    corpus_select = gr.Dropdown(choices=["default"], value="default", label="Knowledge Corpus")
                    
                with gr.Accordion("🔑 NVIDIA API Key Override", open=False):
                    gr.Markdown("Set or update the key in the backend memory:")
                    api_key_box = gr.Textbox(placeholder="nvapi-...", type="password", show_label=False)
                    apply_btn = gr.Button("Apply Key", size="sm", variant="primary")
                    key_status = gr.Markdown(value="")
            
            # Right Column: Chatbot
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(label="Conversation History", height=500)
                
                # Chat Input Row
                with gr.Row():
                    txt = gr.Textbox(
                        show_label=False,
                        placeholder="Ask a question about Attention, BERT, RAG, MRKL, Mistral, or GraphRAG...",
                        scale=8
                    )
                    submit_btn = gr.Button("Send", variant="primary", scale=1)
                
                # Quick Suggestions Box
                gr.Markdown("💡 **Try asking:**")
                with gr.Row():
                    s1 = gr.Button("How does MRKL architecture combine LLMs with external tools?", variant="secondary", size="sm", elem_classes="suggest-btn")
                    s2 = gr.Button("What are the key concepts of the RAG framework?", variant="secondary", size="sm", elem_classes="suggest-btn")
                    s3 = gr.Button("Explain GraphRAG concept and its benefits.", variant="secondary", size="sm", elem_classes="suggest-btn")
                    
        # Event bindings
        refresh_btn.click(fn=lambda: check_backend_status()[0], inputs=None, outputs=status_display)
        apply_btn.click(fn=apply_api_key, inputs=api_key_box, outputs=key_status)
        
        # Submit action (send button click & enter press)
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
        
        # Suggestions click bindings
        s1.click(fn=lambda: "How does MRKL architecture combine LLMs with external tools?", outputs=txt).then(
            fn=ask_question,
            inputs=[txt, chatbot, k_slider, corpus_select, api_key_box],
            outputs=[chatbot, txt, key_status]
        )
        s2.click(fn=lambda: "What are the key concepts of the RAG framework?", outputs=txt).then(
            fn=ask_question,
            inputs=[txt, chatbot, k_slider, corpus_select, api_key_box],
            outputs=[chatbot, txt, key_status]
        )
        s3.click(fn=lambda: "Explain GraphRAG concept and its benefits.", outputs=txt).then(
            fn=ask_question,
            inputs=[txt, chatbot, k_slider, corpus_select, api_key_box],
            outputs=[chatbot, txt, key_status]
        )
        
    return demo

if __name__ == "__main__":
    demo = init_ui()
    demo.launch(server_name="0.0.0.0", server_port=8090, share=False, css=custom_css, theme=theme)
