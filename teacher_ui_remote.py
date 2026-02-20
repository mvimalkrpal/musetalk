#!/usr/bin/env python3
"""
Mac-side remote UI:
- Student text -> Gemini
- AI text -> pyttsx3 wav
- Send wav to Windows MuseTalk API
- Receive and display mp4 reply
"""

import argparse
import os
import tempfile
import time
from pathlib import Path
from typing import List, Tuple

import gradio as gr
import requests

from teacher_realtime import synthesize_pyttsx3


def chat_gemini(
    student_text: str,
    history: List[Tuple[str, str]],
    model: str,
    api_key: str,
    system_prompt: str,
    timeout_s: int = 60,
) -> str:
    if not api_key:
        raise RuntimeError("Missing Gemini API key. Set --gemini_api_key or GEMINI_API_KEY.")

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    prompt_parts = [system_prompt.strip(), "\n\nConversation so far:\n"]
    for u, a in history:
        prompt_parts.append(f"Student: {u}\nTeacher: {a}\n")
    prompt_parts.append(f"Student: {student_text}\nTeacher:")
    prompt = "".join(prompt_parts)

    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    resp = requests.post(url, params={"key": api_key}, json=payload, timeout=timeout_s)
    resp.raise_for_status()
    data = resp.json()

    candidates = data.get("candidates", [])
    if not candidates:
        raise RuntimeError(f"Gemini returned no candidates: {data}")
    parts = candidates[0].get("content", {}).get("parts", [])
    return "".join(p.get("text", "") for p in parts).strip()


def send_to_musetalk_api(server_url: str, wav_path: Path, output_prefix: str) -> Path:
    out_dir = Path(tempfile.mkdtemp(prefix="remote_musetalk_video_"))
    out_video = out_dir / "reply.mp4"
    url = server_url.rstrip("/") + "/generate"

    with open(wav_path, "rb") as f:
        files = {"audio": ("input.wav", f, "audio/wav")}
        data = {"output_prefix": output_prefix}
        resp = requests.post(url, files=files, data=data, timeout=1800)
    resp.raise_for_status()

    with open(out_video, "wb") as f:
        f.write(resp.content)
    return out_video


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Remote MuseTalk UI (Gemini + Windows API)")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7863)
    parser.add_argument("--musetalk_server", required=True, help="Example: http://192.168.1.10:8787")

    parser.add_argument("--gemini_model", default="gemini-2.5-flash")
    parser.add_argument("--gemini_api_key", default=os.getenv("GEMINI_API_KEY", ""))
    parser.add_argument(
        "--system_prompt",
        default=(
            "You are an English teacher. Keep replies short (1-3 sentences), "
            "friendly, and correct grammar naturally. Ask one follow-up question."
        ),
    )

    parser.add_argument("--tts_rate", type=int, default=170)
    parser.add_argument("--tts_voice", default=None)
    parser.add_argument("--output_prefix", default="teacher_reply")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    def generate_reply(student_text: str, history: List[List[str]]):
        history = history or []
        if not student_text or not student_text.strip():
            return history, None, "Enter student text."

        start = time.time()
        tuple_history = [(x[0], x[1]) for x in history if len(x) == 2]

        try:
            ai_text = chat_gemini(
                student_text=student_text.strip(),
                history=tuple_history,
                model=args.gemini_model,
                api_key=args.gemini_api_key,
                system_prompt=args.system_prompt,
            )
            if not ai_text:
                ai_text = "I could not generate a response. Please try again."

            wav_tmp = Path(tempfile.mkdtemp(prefix="remote_musetalk_tts_")) / "reply.wav"
            synthesize_pyttsx3(ai_text, wav_tmp, args.tts_rate, args.tts_voice)

            out_video = send_to_musetalk_api(args.musetalk_server, wav_tmp, args.output_prefix)
            history.append([student_text, ai_text])

            took = time.time() - start
            return history, str(out_video), f"Turn done in {took:.2f}s"
        except Exception as exc:
            return history, None, f"Error: {exc}"

    with gr.Blocks(title="AI English Teacher (Remote MuseTalk)") as demo:
        gr.Markdown("## AI English Teacher (Gemini on this machine, MuseTalk on Windows)")
        with gr.Row():
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(label="Conversation", height=420)
                student_text = gr.Textbox(
                    label="Student Input",
                    placeholder="Type what the student says...",
                    lines=3,
                )
                with gr.Row():
                    send_btn = gr.Button("Send")
                    clear_btn = gr.Button("Clear Chat")
                status = gr.Textbox(label="Status", interactive=False)
            with gr.Column(scale=1):
                video = gr.Video(label="Teacher Video Reply")
                gr.Markdown(
                    f"MuseTalk server: `{args.musetalk_server}`  \n"
                    f"Gemini model: `{args.gemini_model}`"
                )

        send_btn.click(
            fn=generate_reply,
            inputs=[student_text, chatbot],
            outputs=[chatbot, video, status],
        )
        student_text.submit(
            fn=generate_reply,
            inputs=[student_text, chatbot],
            outputs=[chatbot, video, status],
        )
        clear_btn.click(
            fn=lambda: ([], None, "Cleared."),
            inputs=None,
            outputs=[chatbot, video, status],
        )

    demo.launch(server_name=args.host, server_port=args.port)


if __name__ == "__main__":
    main()

