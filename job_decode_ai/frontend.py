import os
import gradio as gr
import requests

FASTAPI_URL = os.getenv("FASTAPI_URL", "http://127.0.0.1:8002")


def _parse_api_response(response):
    try:
        data = response.json()
    except ValueError:
        text = response.text.strip()
        return {"status": "error", "message": text or f"HTTP {response.status_code}"}

    if isinstance(data, dict):
        return data
    return {"status": "error", "message": str(data)}


def load_job_description(raw_text, file_obj):
    file_path = None
    if file_obj is not None:
        file_path = getattr(file_obj, "name", str(file_obj))

    payload = {
        "file_path": file_path,
        "raw_text": (raw_text or "").strip() or None,
        "temperature": 0.0,
    }

    try:
        response = requests.post(f"{FASTAPI_URL}/load", json=payload, timeout=30)
        data = _parse_api_response(response)
        if response.ok:
            return data.get("message", data.get("status", str(data)))
        return f"Error: {data.get('message', data.get('detail', str(data)))}"
    except Exception as exc:
        return f"Error: {str(exc)}"


def ask_question(message, history):
    payload = {"question": message}

    try:
        response = requests.post(f"{FASTAPI_URL}/ask", json=payload, timeout=30)
        data = _parse_api_response(response)
        answer = data.get("answer", data.get("message", "No answer returned."))
    except Exception as exc:
        answer = f"Error: {str(exc)}"

    history = history or []
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": answer})

    return "", history

with gr.Blocks(title="JobDecode AI") as demo:
    gr.Markdown("# JobDecode AI")
    gr.Markdown("Upload or paste a job description, index it, and ask role-specific questions.")

    with gr.Row():
        with gr.Column():
            raw_text = gr.Textbox(label="Paste Job Description", lines=12)
            file_input = gr.File(label="Or Upload a PDF / TXT file", file_types=[".pdf", ".txt"])
            load_button = gr.Button("Load / Index Job Description")
            load_status = gr.Textbox(label="Status", interactive=False)

        with gr.Column():
            chatbot = gr.Chatbot(label="JobDecode Chat")
            msg = gr.Textbox(label="Ask a question")
            send_btn = gr.Button("Send")
            clear_btn = gr.Button("Clear Chat")

    load_button.click(
        fn=load_job_description,
        inputs=[raw_text, file_input],
        outputs=load_status
    )

    send_btn.click(
        fn=ask_question,
        inputs=[msg, chatbot],
        outputs=[msg, chatbot]
    )

    msg.submit(
        fn=ask_question,
        inputs=[msg, chatbot],
        outputs=[msg, chatbot]
    )

    clear_btn.click(lambda: [], outputs=chatbot)

demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True)