#!/usr/bin/env python3
"""
Windows-side MuseTalk API server.

Runs warm MuseTalk runtime and exposes:
- GET /health
- POST /generate (multipart/form-data with wav file field: audio)
"""

import argparse
import base64
import os
import tempfile
import threading
import time
import traceback
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from urllib.parse import quote

from fastapi import FastAPI, File, Form, UploadFile
from fastapi import Request
from fastapi.responses import FileResponse, JSONResponse
import uvicorn
import requests

from teacher_realtime import AvatarCache, RuntimeModels, load_runtime, resolve_output_path


def _log_exception(prefix: str, exc: BaseException) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {prefix}: {type(exc).__name__}: {exc}")
    traceback.print_exception(type(exc), exc, exc.__traceback__)


def _sys_excepthook(exc_type, exc_value, exc_tb):
    print(f"[{time.strftime('%H:%M:%S')}] UNCAUGHT EXCEPTION")
    traceback.print_exception(exc_type, exc_value, exc_tb)


def _threading_excepthook(args):
    print(f"[{time.strftime('%H:%M:%S')}] UNCAUGHT THREAD EXCEPTION in {args.thread.name}")
    traceback.print_exception(args.exc_type, args.exc_value, args.exc_traceback)


sys.excepthook = _sys_excepthook
threading.excepthook = _threading_excepthook


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MuseTalk API server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8787)

    parser.add_argument("--avatar_id", required=True)
    parser.add_argument("--video_path", required=True)
    parser.add_argument("--prepare", action="store_true")

    parser.add_argument("--output_dir", default="./results/v15/api_out")
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
    parser.add_argument("--gemini_api_key", type=str, default="")
    parser.add_argument("--gemini_model", type=str, default="gemini-2.5-flash")
    parser.add_argument("--ollama_endpoint", type=str, default="http://127.0.0.1:11434")
    parser.add_argument("--ollama_model", type=str, default="llama3.2:1b")
    parser.add_argument(
        "--system_prompt",
        type=str,
        default=(
            "You are an English teacher. Reply in 1-2 short sentences and ask one short follow-up question."
        ),
    )
    parser.add_argument("--tts_rate", type=int, default=170)
    parser.add_argument("--tts_voice", type=str, default="")
    return parser


def create_app(args: argparse.Namespace) -> FastAPI:
    app = FastAPI(title="MuseTalk Teacher API", version="1.0")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("MuseTalk Teacher API starting...")
    print(f"Host: {args.host}:{args.port}")
    print(f"Avatar ID: {args.avatar_id}")
    print(f"Video path: {args.video_path}")
    print(f"Output dir: {output_dir.resolve()}")
    print(f"Whisper device: {args.whisper_device}")
    print("=" * 72)

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

    infer_lock = threading.Lock()
    chat_histories: Dict[str, List[Tuple[str, str]]] = {}

    gemini_key = args.gemini_api_key or os.environ.get("GEMINI_API_KEY", "")

    def transcribe_with_gemini(wav_path: Path) -> str:
        if not gemini_key:
            raise RuntimeError("GEMINI_API_KEY missing on server for transcription.")
        wav_bytes = wav_path.read_bytes()
        b64 = base64.b64encode(wav_bytes).decode("utf-8")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{args.gemini_model}:generateContent"
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": "Transcribe this audio into plain text only. No extra words."},
                        {"inline_data": {"mime_type": "audio/wav", "data": b64}},
                    ]
                }
            ]
        }
        resp = requests.post(url, params={"key": gemini_key}, json=payload, timeout=90)
        resp.raise_for_status()
        data = resp.json()
        candidates = data.get("candidates", [])
        if not candidates:
            raise RuntimeError(f"Gemini transcription returned no candidates: {data}")
        parts = candidates[0].get("content", {}).get("parts", [])
        return "".join(p.get("text", "") for p in parts).strip()

    def reply_with_ollama(student_text: str, session_id: str) -> str:
        hist = chat_histories.get(session_id, [])
        messages = [{"role": "system", "content": args.system_prompt}]
        for u, a in hist[-8:]:
            messages.append({"role": "user", "content": u})
            messages.append({"role": "assistant", "content": a})
        messages.append({"role": "user", "content": student_text})

        url = args.ollama_endpoint.rstrip("/") + "/api/chat"
        resp = requests.post(
            url,
            json={"model": args.ollama_model, "messages": messages, "stream": False},
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        reply = data.get("message", {}).get("content", "").strip()
        if not reply:
            raise RuntimeError("Ollama returned empty response.")
        hist.append((student_text, reply))
        chat_histories[session_id] = hist
        return reply

    def synthesize_tts_windows(text: str, out_wav: Path) -> None:
        import pyttsx3

        engine = pyttsx3.init()
        if args.tts_rate > 0:
            engine.setProperty("rate", args.tts_rate)
        if args.tts_voice:
            engine.setProperty("voice", args.tts_voice)
        engine.save_to_file(text, str(out_wav))
        engine.runAndWait()
        if not out_wav.exists() or out_wav.stat().st_size <= 44:
            raise RuntimeError("TTS output wav is empty.")

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        _log_exception(f"UNHANDLED {request.method} {request.url.path}", exc)
        return JSONResponse(status_code=500, content={"error": "Internal server error"})

    @app.get("/health")
    def health():
        print(f"[{time.strftime('%H:%M:%S')}] GET /health")
        return {"status": "ok"}

    @app.post("/generate")
    async def generate(
        audio: UploadFile = File(...),
        output_prefix: str = Form(default=args.output_prefix),
    ):
        req_start = time.time()
        print(
            f"[{time.strftime('%H:%M:%S')}] POST /generate "
            f"filename={audio.filename} prefix={output_prefix}"
        )
        if not audio.filename.lower().endswith(".wav"):
            print(f"[{time.strftime('%H:%M:%S')}] REJECT non-wav upload")
            return JSONResponse(
                status_code=400,
                content={"error": "Only wav files are supported for now."},
            )

        temp_dir = Path(tempfile.mkdtemp(prefix="musetalk_api_"))
        wav_path = temp_dir / "input.wav"
        out_path = resolve_output_path(output_dir, output_prefix or args.output_prefix)

        try:
            with open(wav_path, "wb") as f:
                f.write(await audio.read())

            with infer_lock:
                start = time.time()
                avatar.infer_audio(
                    models=models,
                    audio_path=str(wav_path),
                    fps=args.fps,
                    batch_size=args.batch_size,
                    left_pad=args.audio_padding_length_left,
                    right_pad=args.audio_padding_length_right,
                    output_path=out_path,
                )
                print(f"Inference done in {time.time() - start:.2f}s -> {out_path}")
            print(f"[{time.strftime('%H:%M:%S')}] POST /generate done in {time.time() - req_start:.2f}s")

            return FileResponse(
                str(out_path),
                media_type="video/mp4",
                filename=out_path.name,
            )
        except Exception as exc:
            _log_exception("POST /generate error", exc)
            return JSONResponse(status_code=500, content={"error": str(exc)})
        finally:
            try:
                if wav_path.exists():
                    wav_path.unlink()
                temp_dir.rmdir()
            except Exception:
                pass

    @app.post("/converse")
    async def converse(
        audio: UploadFile = File(...),
        session_id: str = Form(default="default"),
        output_prefix: str = Form(default=args.output_prefix),
    ):
        req_start = time.time()
        print(
            f"[{time.strftime('%H:%M:%S')}] POST /converse "
            f"filename={audio.filename} session_id={session_id} prefix={output_prefix}"
        )
        if not audio.filename.lower().endswith(".wav"):
            return JSONResponse(status_code=400, content={"error": "Only wav files are supported."})

        temp_dir = Path(tempfile.mkdtemp(prefix="musetalk_converse_"))
        student_wav = temp_dir / "student.wav"
        reply_wav = temp_dir / "reply.wav"
        out_path = resolve_output_path(output_dir, output_prefix or args.output_prefix)

        try:
            with open(student_wav, "wb") as f:
                f.write(await audio.read())

            with infer_lock:
                t0 = time.time()
                student_text = transcribe_with_gemini(student_wav)
                stt_ms = (time.time() - t0) * 1000

                t1 = time.time()
                teacher_reply = reply_with_ollama(student_text, session_id)
                llm_ms = (time.time() - t1) * 1000

                t2 = time.time()
                synthesize_tts_windows(teacher_reply, reply_wav)
                tts_ms = (time.time() - t2) * 1000

                t3 = time.time()
                avatar.infer_audio(
                    models=models,
                    audio_path=str(reply_wav),
                    fps=args.fps,
                    batch_size=args.batch_size,
                    left_pad=args.audio_padding_length_left,
                    right_pad=args.audio_padding_length_right,
                    output_path=out_path,
                )
                render_ms = (time.time() - t3) * 1000

            print(
                f"[{time.strftime('%H:%M:%S')}] /converse timings: "
                f"stt={stt_ms:.0f}ms llm={llm_ms:.0f}ms tts={tts_ms:.0f}ms render={render_ms:.0f}ms"
            )
            print(f"[{time.strftime('%H:%M:%S')}] student='{student_text}'")
            print(f"[{time.strftime('%H:%M:%S')}] teacher='{teacher_reply}'")
            print(f"[{time.strftime('%H:%M:%S')}] POST /converse done in {time.time() - req_start:.2f}s")

            headers = {
                "X-Student-Text": quote(student_text[:1000]),
                "X-Teacher-Reply": quote(teacher_reply[:1000]),
            }
            return FileResponse(
                str(out_path),
                media_type="video/mp4",
                filename=out_path.name,
                headers=headers,
            )
        except Exception as exc:
            _log_exception("POST /converse error", exc)
            return JSONResponse(status_code=500, content={"error": str(exc)})
        finally:
            try:
                for p in [student_wav, reply_wav]:
                    if p.exists():
                        p.unlink()
                temp_dir.rmdir()
            except Exception:
                pass

    return app


def main() -> None:
    args = build_parser().parse_args()
    try:
        app = create_app(args)
        uvicorn.run(app, host=args.host, port=args.port, access_log=True)
    except Exception as exc:
        _log_exception("FATAL server startup crash", exc)
        raise


if __name__ == "__main__":
    main()
