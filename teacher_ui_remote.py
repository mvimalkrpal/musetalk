#!/usr/bin/env python3
"""
Mac-side remote UI (audio-only):
- Record/upload wav
- Send to Windows MuseTalk API /generate
- Receive and display mp4 reply
"""

import argparse
import audioop
import ipaddress
import socket
import tempfile
import time
import wave
from pathlib import Path
from urllib.parse import urlparse, unquote

import gradio as gr
import requests


def ts() -> str:
    return time.strftime("%H:%M:%S")


def resolve_server_base_url(server_url: str) -> str:
    parsed = urlparse(server_url)
    if not parsed.scheme or not parsed.hostname:
        return server_url

    try:
        ipaddress.ip_address(parsed.hostname)
        return server_url
    except ValueError:
        pass

    try:
        ip = socket.gethostbyname(parsed.hostname)
    except Exception:
        return server_url

    if parsed.port:
        return f"{parsed.scheme}://{ip}:{parsed.port}"
    return f"{parsed.scheme}://{ip}"


def normalize_wav_for_upload(wav_path: Path) -> tuple[Path, str]:
    out_dir = Path(tempfile.mkdtemp(prefix="remote_musetalk_wav_"))
    out_wav = out_dir / "input_16k_mono.wav"
    with wave.open(str(wav_path), "rb") as r:
        channels = r.getnchannels()
        sampwidth = r.getsampwidth()
        framerate = r.getframerate()
        frames = r.readframes(r.getnframes())

    if channels > 1:
        frames = audioop.tomono(frames, sampwidth, 0.5, 0.5)
    if framerate != 16000:
        frames, _ = audioop.ratecv(frames, sampwidth, 1, framerate, 16000, None)

    with wave.open(str(out_wav), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(sampwidth)
        w.setframerate(16000)
        w.writeframes(frames)
    return out_wav, f"wav_norm: {channels}ch@{framerate}Hz -> 1ch@16000Hz"


def send_to_musetalk_api(http: requests.Session, server_url: str, wav_path: Path, output_prefix: str, reply_backend: str, session_id: str):
    out_dir = Path(tempfile.mkdtemp(prefix="remote_musetalk_video_"))
    out_video = out_dir / f"reply_{reply_backend}.mp4"
    base = resolve_server_base_url(server_url)
    url = base.rstrip("/") + "/converse"
    print(
        f"[{ts()}] [client] POST {url} file={wav_path.name} session_id={session_id} "
        f"prefix={output_prefix} backend={reply_backend}"
    )

    t_post = time.time()
    with open(wav_path, "rb") as f:
        files = {"audio": (wav_path.name, f, "audio/wav")}
        data = {"output_prefix": output_prefix, "session_id": session_id, "reply_backend": reply_backend}
        resp = http.post(url, files=files, data=data, timeout=1800)
    resp.raise_for_status()
    print(
        f"[{ts()}] [client] response status={resp.status_code} content-type={resp.headers.get('content-type')} "
        f"roundtrip_ms={(time.time() - t_post) * 1000:.0f}"
    )
    student_text = unquote(resp.headers.get("X-Student-Text", ""))
    teacher_reply = unquote(resp.headers.get("X-Teacher-Reply", ""))
    if student_text:
        print(f"[{ts()}] [client] student={student_text}")
    if teacher_reply:
        print(f"[{ts()}] [client] teacher={teacher_reply}")

    with open(out_video, "wb") as f:
        f.write(resp.content)
    meta = {
        "backend": resp.headers.get("X-Reply-Backend", reply_backend),
        "student": student_text,
        "reply": teacher_reply,
        "stt_ms": resp.headers.get("X-Stt-Ms", ""),
        "llm_ms": resp.headers.get("X-Llm-Ms", ""),
        "tts_ms": resp.headers.get("X-Tts-Ms", ""),
        "render_ms": resp.headers.get("X-Render-Ms", ""),
        "total_ms": resp.headers.get("X-Total-Ms", ""),
    }
    return out_video, meta


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Remote MuseTalk UI (audio-only)")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7863)
    parser.add_argument("--musetalk_server", required=True, help="Example: http://markinova.local:8787")
    parser.add_argument("--output_prefix", default="teacher_reply")
    parser.add_argument("--normalize_audio", action="store_true", help="Normalize client audio to mono 16k before upload")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    http = requests.Session()

    def generate_from_audio(audio_path: str, backends: list[str]):
        if not audio_path:
            return None, "Error: Record or upload a .wav file first."
        if not audio_path.lower().endswith(".wav"):
            return None, "Error: Only .wav is supported by the current server."
        if not backends:
            return None, "Error: Select at least one backend."

        start = time.time()
        normalized_wav = None
        try:
            if args.normalize_audio:
                print(f"[{ts()}] [client] normalize:start")
                normalized_wav, norm_msg = normalize_wav_for_upload(Path(audio_path))
                print(f"[{ts()}] [client] normalize:end")
                upload_wav = normalized_wav
            else:
                norm_msg = "wav_norm: skipped (fast path)"
                upload_wav = Path(audio_path)
            primary_video = None
            logs = [norm_msg]
            for b in backends:
                out_video, meta = send_to_musetalk_api(
                    http=http,
                    server_url=args.musetalk_server,
                    wav_path=upload_wav,
                    output_prefix=f"{args.output_prefix}_{b}",
                    reply_backend=b,
                    session_id=f"default_{b}",
                )
                if primary_video is None:
                    primary_video = str(out_video)
                if meta["backend"] == "gemini_live":
                    logs.append(
                        f"[{meta['backend']}] ok total={meta['total_ms']}ms render={meta['render_ms']}ms (audio-in/audio-out)"
                    )
                else:
                    logs.append(
                        f"[{meta['backend']}] ok total={meta['total_ms']}ms render={meta['render_ms']}ms reply={meta['reply'][:140]}"
                    )
            logs.append(f"Done in {time.time() - start:.2f}s")
            return primary_video, "\n".join(logs)
        except Exception as exc:
            return None, f"Error: {exc}"
        finally:
            if normalized_wav is not None:
                try:
                    parent = normalized_wav.parent
                    if normalized_wav.exists():
                        normalized_wav.unlink()
                    parent.rmdir()
                except Exception:
                    pass

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
      min-height: 94vh;
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
      position: fixed;
      left: 12px;
      right: 12px;
      bottom: 10px;
      z-index: 50;
      background: var(--glass);
      backdrop-filter: blur(10px);
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 8px;
    }
    #send_btn button {
      border-radius: 10px !important;
      height: 42px !important;
      font-weight: 700 !important;
      background: linear-gradient(135deg, var(--accent-b) 0%, var(--accent-a) 100%) !important;
      color: #021015 !important;
    }
    #status_box textarea {font-size: 12px !important; line-height: 1.35 !important;}

    /* Small screens: WhatsApp/FaceTime style */
    @media (max-width: 767px) {
      .gradio-container {max-width: 430px !important; margin: 0 auto !important; padding: 6px !important;}
      #call_shell {min-height: 130vh; border-radius: 22px; padding-bottom: 220px;}
      #teacher_video {height: 44vh !important;}
    }

    /* Large screens: Zoom/Meet style */
    @media (min-width: 1024px) {
        .gradio-container {padding: 18px !important;}
      #call_shell {
        min-height: 88vh;
        border-radius: 18px;
        display: grid;
        grid-template-columns: 1fr 360px;
        align-items: stretch;
      }
      #teacher_video {
        height: 88vh !important;
        border-right: 1px solid var(--line);
      }
      #controls_bar {
        position: sticky;
        top: 14px;
        right: 0;
        margin: 14px;
        height: fit-content;
        align-self: start;
        z-index: 20;
      }
    }
    """

    with gr.Blocks(title="Remote MuseTalk (Audio to Video)", css=css) as demo:
        with gr.Column(elem_id="call_shell"):
            teacher_video = gr.Video(label="Teacher", elem_id="teacher_video")
            with gr.Column(elem_id="controls_bar"):
                backend_sel = gr.CheckboxGroup(
                    choices=["gemini_live", "gemini", "ollama"],
                    value=["gemini_live"],
                    label="Brain",
                )
                audio_in = gr.Audio(
                    sources=["microphone", "upload"],
                    type="filepath",
                    label="Audio",
                )
                run_btn = gr.Button("Send", elem_id="send_btn")
                status = gr.Textbox(label="Response", interactive=False, lines=8, elem_id="status_box")

        run_btn.click(
            fn=generate_from_audio,
            inputs=[audio_in, backend_sel],
            outputs=[teacher_video, status],
        )

    demo.launch(server_name=args.host, server_port=args.port)


if __name__ == "__main__":
    main()
