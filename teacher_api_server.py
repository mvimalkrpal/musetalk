#!/usr/bin/env python3
"""
Windows-side MuseTalk API server.

Runs warm MuseTalk runtime and exposes:
- GET /health
- POST /generate (multipart/form-data with wav file field: audio)
"""

import argparse
import tempfile
import threading
import time
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse
import uvicorn

from teacher_realtime import AvatarCache, RuntimeModels, load_runtime, resolve_output_path


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
    return parser


def create_app(args: argparse.Namespace) -> FastAPI:
    app = FastAPI(title="MuseTalk Teacher API", version="1.0")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/generate")
    async def generate(
        audio: UploadFile = File(...),
        output_prefix: str = Form(default=args.output_prefix),
    ):
        if not audio.filename.lower().endswith(".wav"):
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

            return FileResponse(
                str(out_path),
                media_type="video/mp4",
                filename=out_path.name,
            )
        except Exception as exc:
            return JSONResponse(status_code=500, content={"error": str(exc)})
        finally:
            try:
                if wav_path.exists():
                    wav_path.unlink()
                temp_dir.rmdir()
            except Exception:
                pass

    return app


def main() -> None:
    args = build_parser().parse_args()
    app = create_app(args)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

