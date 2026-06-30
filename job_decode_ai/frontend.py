import os
import gradio as gr
import requests

FASTAPI_URL = os.getenv("FASTAPI_URL", "http://127.0.0.1:8001")

CUSTOM_CSS = """
footer {display: none !important;}
"""

def _parse_api_response(response):
    try:
        data = response.json()
    except ValueError:
        text = response.text.strip()
        return {"status": "error", "message": text or f"HTTP {response.status_code}"}

    if isinstance(data, dict):
        return data

    return {"status": "error", "message": str(data)}


def _normalize_text(raw_text):
    if not raw_text:
        return None
    return raw_text.replace("\r\n", "\n").replace("\r", "\n").strip() or None


def load_job_description(raw_text, file_obj):
    file_path = None
    if file_obj is not None:
        file_path = getattr(file_obj, "name", str(file_obj))

    payload = {
        "file_path": file_path,
        "raw_text": _normalize_text(raw_text),
        "temperature": 0.0,
    }

    try:
        response = requests.post(
            f"{FASTAPI_URL}/load",
            json=payload,
            timeout=120
        )
        data = _parse_api_response(response)

        if response.ok:
            if data.get("status") == "success":
                return data.get("message", "Indexed successfully.")
            return f"Error: {data.get('message', 'Unknown error')}"

        return f"Error: {data.get('message', data.get('detail', str(data)))}"
    except Exception as exc:
        return f"Error: {str(exc)}"


def ask_question(message, history_text):
    message = (message or "").strip()
    history_text = history_text or ""

    if not message:
        return "", history_text

    payload = {"question": message}

    try:
        response = requests.post(
            f"{FASTAPI_URL}/ask",
            json=payload,
            timeout=120
        )
        data = _parse_api_response(response)
        answer = data.get("answer", data.get("message", "No answer returned."))
    except Exception as exc:
        answer = f"Error: {str(exc)}"

    updated_history = history_text.strip()
    if updated_history:
        updated_history += "\n\n"

    updated_history += f"User: {message}\nAssistant: {answer}"
    return "", updated_history


def clear_chat():
    return ""


with gr.Blocks(title="JobDecode AI") as demo:
    gr.Markdown("# JobDecode AI")
    gr.Markdown("Upload or paste a job description, index it, and ask role-specific questions.")

    with gr.Row():
        with gr.Column():
            raw_text = gr.Textbox(
                label="Paste Job Description",
                lines=14,
                placeholder="Paste the job description here..."
            )
            file_input = gr.File(
                label="Or Upload a PDF / TXT file",
                file_types=[".pdf", ".txt"]
            )
            load_button = gr.Button("Load / Index Job Description")
            load_status = gr.Textbox(label="Status", interactive=False)

        with gr.Column():
            chat_history = gr.Textbox(
                label="JobDecode Chat",
                lines=20,
                interactive=False,
                placeholder="Conversation will appear here..."
            )
            msg = gr.Textbox(
                label="Ask a question",
                placeholder="e.g. What skills are required?"
            )
            send_btn = gr.Button("Send")
            clear_btn = gr.Button("Clear Chat")

    load_button.click(
        fn=load_job_description,
        inputs=[raw_text, file_input],
        outputs=load_status
    )

    send_btn.click(
        fn=ask_question,
        inputs=[msg, chat_history],
        outputs=[msg, chat_history]
    )

    msg.submit(
        fn=ask_question,
        inputs=[msg, chat_history],
        outputs=[msg, chat_history]
    )

    clear_btn.click(
        fn=clear_chat,
        outputs=chat_history
    )

demo.launch(
    server_name="127.0.0.1",
    server_port=7861,
    inbrowser=True,
    css=CUSTOM_CSS
)