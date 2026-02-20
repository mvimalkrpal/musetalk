#!/usr/bin/env python3
"""
Mac-side remote UI (audio-only):
- Record/upload wav
- Send to Windows MuseTalk API /generate
- Receive and display mp4 reply
"""

import argparse
import socket
import tempfile
import time
from pathlib import Path
from urllib.parse import urlparse

import gradio as gr
import requests


def resolve_server_base_url(server_url: str) -> str:
    parsed = urlparse(server_url)
    if not parsed.scheme or not parsed.hostname:
        return server_url

    try:
        ip = socket.gethostbyname(parsed.hostname)
    except Exception:
        return server_url

    if parsed.port:
        return f"{parsed.scheme}://{ip}:{parsed.port}"
    return f"{parsed.scheme}://{ip}"


def send_to_musetalk_api(server_url: str, wav_path: Path, output_prefix: str) -> Path:
    out_dir = Path(tempfile.mkdtemp(prefix="remote_musetalk_video_"))
    out_video = out_dir / "reply.mp4"
    base = resolve_server_base_url(server_url)
    url = base.rstrip("/") + "/generate"

    with open(wav_path, "rb") as f:
        files = {"audio": (wav_path.name, f, "audio/wav")}
        data = {"output_prefix": output_prefix}
        resp = requests.post(url, files=files, data=data, timeout=1800)
    resp.raise_for_status()

    with open(out_video, "wb") as f:
        f.write(resp.content)
    return out_video


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Remote MuseTalk UI (audio-only)")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7863)
    parser.add_argument("--musetalk_server", required=True, help="Example: http://markinova.local:8787")
    parser.add_argument("--output_prefix", default="teacher_reply")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    def generate_from_audio(audio_path: str):
        if not audio_path:
            return None, "Record or upload a .wav file first."
        if not audio_path.lower().endswith(".wav"):
            return None, "Only .wav is supported by the current server."

        start = time.time()
        try:
            out_video = send_to_musetalk_api(
                server_url=args.musetalk_server,
                wav_path=Path(audio_path),
                output_prefix=args.output_prefix,
            )
            return str(out_video), f"Done in {time.time() - start:.2f}s"
        except Exception as exc:
            return None, f"Error: {exc}"

    css = """
    :root {
      --bg-a: #1b2735;
      --bg-b: #090a0f;
      --bg-c: #050507;
      --glass: rgba(12, 16, 24, 0.7);
      --line: rgba(255, 255, 255, 0.16);
      --text: #f5f7ff;
      --accent-a: #22d3ee;
      --accent-b: #5eead4;
    }
    body, .gradio-container {
      background: radial-gradient(1000px 600px at 10% 0%, var(--bg-a) 0%, var(--bg-b) 45%, var(--bg-c) 100%);
    }
    .gradio-container {
      max-width: 100vw !important;
      width: 100% !important;
      margin: 0 !important;
      padding: 10px !important;
      min-height: 100vh;
    }
    #call_shell {
      position: relative;
      height: 94vh;
      border-radius: 24px;
      overflow: hidden;
      border: 1px solid var(--line);
      background: #06070a;
    }
    #teacher_video {
      height: 100% !important;
      min-height: 100% !important;
    }
    #teacher_video video {
      width: 100% !important;
      height: 100% !important;
      object-fit: cover !important;
    }
    #controls_bar {
      position: absolute;
      left: 10px;
      right: 10px;
      bottom: 10px;
      z-index: 20;
      background: var(--glass);
      backdrop-filter: blur(10px);
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 8px;
    }
    #controls_bar label {display: none !important;}
    #send_btn button {
      border-radius: 10px !important;
      height: 42px !important;
      font-weight: 700 !important;
      background: linear-gradient(135deg, var(--accent-b) 0%, var(--accent-a) 100%) !important;
      color: #021015 !important;
    }
    #status_box textarea {
      font-size: 12px !important;
      min-height: 44px !important;
    }

    /* Small screens: WhatsApp/FaceTime style */
    @media (max-width: 767px) {
      .gradio-container {max-width: 430px !important; margin: 0 auto !important; padding: 6px !important;}
      #call_shell {height: 95vh; border-radius: 22px;}
    }

    /* Large screens: Zoom/Meet style */
    @media (min-width: 1024px) {
      .gradio-container {padding: 18px !important;}
      #call_shell {
        height: 88vh;
        border-radius: 18px;
        display: grid;
        grid-template-columns: 1.75fr 0.9fr;
      }
      #teacher_video {
        height: 100% !important;
        border-right: 1px solid var(--line);
      }
      #controls_bar {
        position: static;
        margin: 14px;
        align-self: end;
      }
    }
    """

    with gr.Blocks(title="Remote MuseTalk (Audio to Video)", css=css) as demo:
        with gr.Column(elem_id="call_shell"):
            video_out = gr.Video(label="Teacher", elem_id="teacher_video")
            with gr.Column(elem_id="controls_bar"):
                audio_in = gr.Audio(
                    sources=["microphone", "upload"],
                    type="filepath",
                    label="Audio",
                )
                run_btn = gr.Button("Send", elem_id="send_btn")
                status = gr.Textbox(label="Status", interactive=False, elem_id="status_box")

        run_btn.click(
            fn=generate_from_audio,
            inputs=[audio_in],
            outputs=[video_out, status],
        )

    demo.launch(server_name=args.host, server_port=args.port)


if __name__ == "__main__":
    main()
