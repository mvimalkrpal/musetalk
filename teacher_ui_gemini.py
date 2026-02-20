#!/usr/bin/env python3
"""
Gradio UI for AI English Teacher with warm MuseTalk runtime + Gemini API backend.
"""

import argparse
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import List, Tuple

import gradio as gr
import requests

from teacher_realtime import (
    AvatarCache,
    RuntimeModels,
    load_runtime,
    resolve_output_path,
    synthesize_pyttsx3,
)


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

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ]
    }
    resp = requests.post(
        url,
        params={"key": api_key},
        json=payload,
        timeout=timeout_s,
    )
    resp.raise_for_status()
    data = resp.json()

    candidates = data.get("candidates", [])
    if not candidates:
        raise RuntimeError(f"Gemini returned no candidates: {data}")
    parts = candidates[0].get("content", {}).get("parts", [])
    text = "".join(p.get("text", "") for p in parts).strip()
    return text


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MuseTalk AI teacher UI (Gemini)")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7862)

    parser.add_argument("--avatar_id", required=True, help="Stable avatar id for caching")
    parser.add_argument("--video_path", required=True, help="Source video or frame directory")
    parser.add_argument("--prepare", action="store_true", help="Recreate avatar materials")

    parser.add_argument("--output_dir", default="./results/v15/teacher_out")
    parser.add_argument("--output_prefix", default="teacher_reply")

    parser.add_argument("--ffmpeg_path", type=str, default="./ffmpeg-4.4-amd64-static/")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--whisper_device", choices=["cpu", "gpu"], default="cpu")

    parser.add_argument("--vae_type", type=str, default="sd-vae")
    parser.add_argument("--unet_config", type=str, default="./models/musetalkV15/musetalk.json")
    parser.add_argument("--unet_model_path", type=str, default="./models/musetalkV15/unet.pth")
    parser.add_argument("--whisper_dir", type=str, default="./models/whisper")

    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--audio_padding_length_left", type=int, default=2)
    parser.add_argument("--audio_padding_length_right", type=int, default=2)
    parser.add_argument("--parsing_mode", default="jaw")
    parser.add_argument("--left_cheek_width", type=int, default=90)
    parser.add_argument("--right_cheek_width", type=int, default=90)
    parser.add_argument("--extra_margin", type=int, default=10)

    parser.add_argument("--gemini_model", default="gemini-1.5-flash")
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
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    print("Loading warm runtime...")
    t0 = time.time()
    models: RuntimeModels = load_runtime(args)
    print(f"Runtime loaded in {time.time() - t0:.2f}s")

    avatar = AvatarCache(
        base_dir=Path("./results/v15/avatars"),
        avatar_id=args.avatar_id,
        video_path=args.video_path,
        parsing_mode=args.parsing_mode,
        extra_margin=args.extra_margin,
    )
    print("Preparing/loading avatar...")
    t1 = time.time()
    avatar.prepare(models=models, batch_size=args.batch_size, force_recreate=args.prepare)
    print(f"Avatar ready in {time.time() - t1:.2f}s")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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

            tmp_dir = Path(tempfile.mkdtemp(prefix="musetalk_ui_tts_"))
            wav_path = tmp_dir / "reply.wav"
            synthesize_pyttsx3(ai_text, wav_path, args.tts_rate, args.tts_voice)

            out_path = resolve_output_path(output_dir, args.output_prefix)
            avatar.infer_audio(
                models=models,
                audio_path=str(wav_path),
                fps=args.fps,
                batch_size=args.batch_size,
                left_pad=args.audio_padding_length_left,
                right_pad=args.audio_padding_length_right,
                output_path=out_path,
            )
            shutil.rmtree(tmp_dir, ignore_errors=True)

            history.append([student_text, ai_text])
            took = time.time() - start
            return history, str(out_path), f"Turn done in {took:.2f}s"
        except Exception as exc:
            return history, None, f"Error: {exc}"

    with gr.Blocks(title="AI English Teacher (MuseTalk + Gemini)") as demo:
        gr.Markdown("## AI English Teacher (MuseTalk + Gemini)")
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
                    f"Gemini model: `{args.gemini_model}`  \n"
                    f"Whisper device: `{args.whisper_device}`"
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
