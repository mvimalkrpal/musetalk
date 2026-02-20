#!/usr/bin/env python3
"""
Warm MuseTalk runtime for low-latency teacher replies.

Single-file helper:
- One-time model load
- One-time avatar preparation (or reuse cached avatar)
- Repeated generation from text (pyttsx3) or wav

Examples:
  python teacher_realtime.py --avatar_id teacher1 --video_path data/video/yongen.mp4 --prepare
  python teacher_realtime.py --avatar_id teacher1 --video_path data/video/yongen.mp4 --audio_path data/audio/eng.wav
  python teacher_realtime.py --avatar_id teacher1 --video_path data/video/yongen.mp4 --interactive --tts_engine pyttsx3
"""

import argparse
import glob
import json
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from transformers import WhisperModel

from musetalk.utils.audio_processor import AudioProcessor
from musetalk.utils.blending import get_image_blending, get_image_prepare_material
from musetalk.utils.face_parsing import FaceParsing
from musetalk.utils.preprocessing import get_landmark_and_bbox, read_imgs
from musetalk.utils.utils import datagen, load_all_model


def fast_check_ffmpeg() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except Exception:
        return False


def ensure_ffmpeg(ffmpeg_path: str) -> None:
    if fast_check_ffmpeg():
        return
    sep = ";" if sys.platform == "win32" else ":"
    os.environ["PATH"] = f"{ffmpeg_path}{sep}{os.environ.get('PATH', '')}"
    if not fast_check_ffmpeg():
        raise RuntimeError(
            "ffmpeg not found. Install ffmpeg or pass --ffmpeg_path correctly."
        )


def video2imgs(vid_path: str, save_path: str, max_frames: int = 10_000_000) -> None:
    cap = cv2.VideoCapture(vid_path)
    count = 0
    while True:
        if count > max_frames:
            break
        ok, frame = cap.read()
        if not ok:
            break
        cv2.imwrite(f"{save_path}/{count:08d}.png", frame)
        count += 1
    cap.release()


@dataclass
class RuntimeModels:
    device: torch.device
    vae: object
    unet: object
    pe: object
    timesteps: torch.Tensor
    whisper: WhisperModel
    whisper_device: torch.device
    whisper_dtype: torch.dtype
    audio_processor: AudioProcessor
    face_parser: FaceParsing


class AvatarCache:
    def __init__(
        self,
        base_dir: Path,
        avatar_id: str,
        video_path: str,
        parsing_mode: str,
        extra_margin: int,
    ) -> None:
        self.base_dir = base_dir
        self.avatar_id = avatar_id
        self.video_path = video_path
        self.parsing_mode = parsing_mode
        self.extra_margin = extra_margin

        self.avatar_path = self.base_dir / avatar_id
        self.full_imgs_path = self.avatar_path / "full_imgs"
        self.coords_path = self.avatar_path / "coords.pkl"
        self.latents_path = self.avatar_path / "latents.pt"
        self.mask_path = self.avatar_path / "mask"
        self.mask_coords_path = self.avatar_path / "mask_coords.pkl"
        self.meta_path = self.avatar_path / "avatar_info.json"

        self.frame_list_cycle: List[np.ndarray] = []
        self.coord_list_cycle: List[List[int]] = []
        self.mask_list_cycle: List[np.ndarray] = []
        self.mask_coords_list_cycle: List[Tuple[int, int, int, int]] = []
        self.input_latent_list_cycle: List[torch.Tensor] = []

    def _load_existing(self) -> None:
        self.input_latent_list_cycle = torch.load(self.latents_path)
        with open(self.coords_path, "rb") as f:
            self.coord_list_cycle = pickle.load(f)
        with open(self.mask_coords_path, "rb") as f:
            self.mask_coords_list_cycle = pickle.load(f)

        img_files = sorted(glob.glob(str(self.full_imgs_path / "*.[jpJP][pnPN]*[gG]")))
        mask_files = sorted(glob.glob(str(self.mask_path / "*.[jpJP][pnPN]*[gG]")))
        self.frame_list_cycle = read_imgs(img_files)
        self.mask_list_cycle = read_imgs(mask_files)

    def _write_meta(self) -> None:
        payload = {
            "avatar_id": self.avatar_id,
            "video_path": self.video_path,
            "parsing_mode": self.parsing_mode,
            "extra_margin": self.extra_margin,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def prepare(self, models: RuntimeModels, batch_size: int, force_recreate: bool) -> None:
        needs_prepare = force_recreate or not self.avatar_path.exists()
        if not needs_prepare:
            try:
                self._load_existing()
                return
            except Exception:
                needs_prepare = True

        if self.avatar_path.exists():
            shutil.rmtree(self.avatar_path)
        self.full_imgs_path.mkdir(parents=True, exist_ok=True)
        self.mask_path.mkdir(parents=True, exist_ok=True)

        if os.path.isfile(self.video_path):
            video2imgs(self.video_path, str(self.full_imgs_path))
        else:
            files = sorted(glob.glob(os.path.join(self.video_path, "*.[jpJP][pnPN]*[gG]")))
            if not files:
                raise RuntimeError(f"No frames found in directory: {self.video_path}")
            for src in files:
                shutil.copy(src, self.full_imgs_path / os.path.basename(src))

        input_img_list = sorted(glob.glob(str(self.full_imgs_path / "*.[jpJP][pnPN]*[gG]")))
        coord_list, frame_list = get_landmark_and_bbox(input_img_list, 0)

        coord_placeholder = (0.0, 0.0, 0.0, 0.0)
        input_latents = []
        normalized_coords = []

        for bbox, frame in zip(coord_list, frame_list):
            if bbox == coord_placeholder:
                continue
            x1, y1, x2, y2 = [int(v) for v in bbox]
            y2 = min(y2 + self.extra_margin, frame.shape[0])
            normalized_coords.append([x1, y1, x2, y2])

            crop = frame[y1:y2, x1:x2]
            crop = cv2.resize(crop, (256, 256), interpolation=cv2.INTER_LANCZOS4)
            latents = models.vae.get_latents_for_unet(crop)
            input_latents.append(latents)

        if not input_latents:
            raise RuntimeError("Avatar preparation failed: no valid face crops found.")

        self.frame_list_cycle = frame_list + frame_list[::-1]
        self.coord_list_cycle = normalized_coords + normalized_coords[::-1]
        self.input_latent_list_cycle = input_latents + input_latents[::-1]
        self.mask_list_cycle = []
        self.mask_coords_list_cycle = []

        print(f"[avatar:{self.avatar_id}] prepare_masks:start total={len(self.frame_list_cycle)}")
        for i, frame in enumerate(self.frame_list_cycle):
            cv2.imwrite(str(self.full_imgs_path / f"{i:08d}.png"), frame)
            x1, y1, x2, y2 = self.coord_list_cycle[i]
            mask, crop_box = get_image_prepare_material(
                frame,
                [x1, y1, x2, y2],
                fp=models.face_parser,
                mode=self.parsing_mode,
            )
            cv2.imwrite(str(self.mask_path / f"{i:08d}.png"), mask)
            self.mask_list_cycle.append(mask)
            self.mask_coords_list_cycle.append(crop_box)
        print(f"[avatar:{self.avatar_id}] prepare_masks:end")

        with open(self.coords_path, "wb") as f:
            pickle.dump(self.coord_list_cycle, f)
        with open(self.mask_coords_path, "wb") as f:
            pickle.dump(self.mask_coords_list_cycle, f)
        torch.save(self.input_latent_list_cycle, self.latents_path)
        self._write_meta()

    @torch.no_grad()
    def infer_audio(
        self,
        models: RuntimeModels,
        audio_path: str,
        fps: int,
        batch_size: int,
        left_pad: int,
        right_pad: int,
        output_path: Path,
    ) -> Path:
        whisper_input_features, librosa_length = models.audio_processor.get_audio_feature(
            audio_path, weight_dtype=models.whisper_dtype
        )
        whisper_chunks = models.audio_processor.get_whisper_chunk(
            whisper_input_features,
            models.whisper_device,
            models.whisper_dtype,
            models.whisper,
            librosa_length,
            fps=fps,
            audio_padding_length_left=left_pad,
            audio_padding_length_right=right_pad,
        )

        gen = datagen(whisper_chunks, self.input_latent_list_cycle, batch_size)
        frames = []
        total_gen_steps = int(np.ceil(float(len(whisper_chunks)) / batch_size))
        print(f"[avatar:{self.avatar_id}] generate_latents:start steps={total_gen_steps}")
        for whisper_batch, latent_batch in gen:
            audio_feature_batch = models.pe(
                whisper_batch.to(device=models.device, dtype=models.unet.model.dtype)
            )
            latent_batch = latent_batch.to(device=models.device, dtype=models.unet.model.dtype)
            pred_latents = models.unet.model(
                latent_batch, models.timesteps, encoder_hidden_states=audio_feature_batch
            ).sample
            pred_latents = pred_latents.to(device=models.device, dtype=models.vae.vae.dtype)
            recon = models.vae.decode_latents(pred_latents)
            frames.extend(recon)
        print(f"[avatar:{self.avatar_id}] generate_latents:end frames={len(frames)}")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_dir = Path(tempfile.mkdtemp(prefix="musetalk_teacher_"))
        silent_video = tmp_dir / "silent.mp4"

        try:
            print(f"[avatar:{self.avatar_id}] blending:start frames={len(frames)}")
            for idx, res_frame in enumerate(frames):
                bbox = self.coord_list_cycle[idx % len(self.coord_list_cycle)]
                ori = self.frame_list_cycle[idx % len(self.frame_list_cycle)].copy()
                x1, y1, x2, y2 = bbox

                resized = cv2.resize(res_frame.astype(np.uint8), (x2 - x1, y2 - y1))
                mask = self.mask_list_cycle[idx % len(self.mask_list_cycle)]
                mask_crop_box = self.mask_coords_list_cycle[idx % len(self.mask_coords_list_cycle)]
                combined = get_image_blending(ori, resized, bbox, mask, mask_crop_box)
                cv2.imwrite(str(tmp_dir / f"{idx:08d}.png"), combined)
            print(f"[avatar:{self.avatar_id}] blending:end")

            print(f"[avatar:{self.avatar_id}] ffmpeg_img2video:start")
            cmd_img2video = [
                "ffmpeg", "-y", "-v", "warning",
                "-r", str(fps),
                "-f", "image2",
                "-i", str(tmp_dir / "%08d.png"),
                "-vcodec", "libx264",
                "-vf", "format=yuv420p",
                "-crf", "18",
                str(silent_video),
            ]
            subprocess.run(cmd_img2video, check=True)
            print(f"[avatar:{self.avatar_id}] ffmpeg_img2video:end")

            print(f"[avatar:{self.avatar_id}] ffmpeg_mux:start")
            cmd_mux = [
                "ffmpeg", "-y", "-v", "warning",
                "-i", audio_path,
                "-i", str(silent_video),
                "-shortest",
                str(output_path),
            ]
            subprocess.run(cmd_mux, check=True)
            print(f"[avatar:{self.avatar_id}] ffmpeg_mux:end")
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        return output_path


def synthesize_pyttsx3(text: str, out_wav: Path, rate: int, voice: Optional[str]) -> Path:
    try:
        import pyttsx3
    except ImportError as exc:
        raise RuntimeError(
            "pyttsx3 is not installed. Install with: pip install pyttsx3"
        ) from exc

    engine = pyttsx3.init()
    if rate > 0:
        engine.setProperty("rate", rate)
    if voice:
        engine.setProperty("voice", voice)
    engine.save_to_file(text, str(out_wav))
    engine.runAndWait()
    return out_wav


def load_runtime(args: argparse.Namespace) -> RuntimeModels:
    ensure_ffmpeg(args.ffmpeg_path)
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

    vae, unet, pe = load_all_model(
        unet_model_path=args.unet_model_path,
        vae_type=args.vae_type,
        unet_config=args.unet_config,
        device=device,
    )
    timesteps = torch.tensor([0], device=device)

    pe = pe.half().to(device)
    vae.vae = vae.vae.half().to(device)
    unet.model = unet.model.half().to(device)

    whisper_device = torch.device(
        "cpu" if args.whisper_device == "cpu" else (f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    )
    whisper_dtype = torch.float32 if whisper_device.type == "cpu" else unet.model.dtype

    whisper = WhisperModel.from_pretrained(args.whisper_dir)
    whisper = whisper.to(device=whisper_device, dtype=whisper_dtype).eval()
    whisper.requires_grad_(False)

    audio_processor = AudioProcessor(feature_extractor_path=args.whisper_dir)
    face_parser = FaceParsing(
        left_cheek_width=args.left_cheek_width,
        right_cheek_width=args.right_cheek_width,
    )

    return RuntimeModels(
        device=device,
        vae=vae,
        unet=unet,
        pe=pe,
        timesteps=timesteps,
        whisper=whisper,
        whisper_device=whisper_device,
        whisper_dtype=whisper_dtype,
        audio_processor=audio_processor,
        face_parser=face_parser,
    )


def resolve_output_path(output_dir: Path, prefix: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return output_dir / f"{prefix}_{ts}.mp4"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Warm MuseTalk runtime for teacher replies.")
    parser.add_argument("--avatar_id", required=True, help="Stable avatar id for caching.")
    parser.add_argument("--video_path", required=True, help="Source video or frame directory for avatar.")
    parser.add_argument("--prepare", action="store_true", help="Force re-create avatar materials.")
    parser.add_argument("--interactive", action="store_true", help="Start interactive loop.")
    parser.add_argument("--audio_path", default=None, help="One-shot wav path to generate from.")
    parser.add_argument("--text", default=None, help="One-shot text; requires --tts_engine pyttsx3.")
    parser.add_argument("--tts_engine", default="none", choices=["none", "pyttsx3"])
    parser.add_argument("--tts_rate", type=int, default=170, help="pyttsx3 speech rate.")
    parser.add_argument("--tts_voice", default=None, help="pyttsx3 voice id (optional).")
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("Loading runtime models...")
    t0 = time.time()
    models = load_runtime(args)
    print(f"Model warmup finished in {time.time() - t0:.2f}s.")

    cache_root = Path(f"./results/v15/avatars")
    avatar = AvatarCache(
        base_dir=cache_root,
        avatar_id=args.avatar_id,
        video_path=args.video_path,
        parsing_mode=args.parsing_mode,
        extra_margin=args.extra_margin,
    )
    print("Preparing/loading avatar cache...")
    t1 = time.time()
    avatar.prepare(models=models, batch_size=args.batch_size, force_recreate=args.prepare)
    print(f"Avatar ready in {time.time() - t1:.2f}s.")

    output_dir = Path(args.output_dir)

    def run_one(audio_path: str, prefix: Optional[str] = None) -> Path:
        out_path = resolve_output_path(output_dir, prefix or args.output_prefix)
        start = time.time()
        out = avatar.infer_audio(
            models=models,
            audio_path=audio_path,
            fps=args.fps,
            batch_size=args.batch_size,
            left_pad=args.audio_padding_length_left,
            right_pad=args.audio_padding_length_right,
            output_path=out_path,
        )
        print(f"Saved: {out} | turn time: {time.time() - start:.2f}s")
        return out

    # One-shot from wav
    if args.audio_path:
        run_one(args.audio_path)
        return

    # One-shot from text via pyttsx3
    if args.text:
        if args.tts_engine != "pyttsx3":
            raise RuntimeError("For --text, pass --tts_engine pyttsx3")
        tmp_wav = Path(tempfile.mkdtemp(prefix="musetalk_tts_")) / "reply.wav"
        try:
            synthesize_pyttsx3(args.text, tmp_wav, args.tts_rate, args.tts_voice)
            run_one(str(tmp_wav))
        finally:
            shutil.rmtree(tmp_wav.parent, ignore_errors=True)
        return

    if not args.interactive:
        raise RuntimeError("Provide one of --audio_path, --text, or --interactive.")

    print("Interactive mode.")
    print("Type plain text to synthesize+generate video.")
    print("Or type: /wav <path-to-wav>")
    print("Or type: /quit")

    while True:
        user_in = input("\nTeacher reply > ").strip()
        if not user_in:
            continue
        if user_in.lower() in {"/quit", "quit", "exit"}:
            break
        if user_in.startswith("/wav "):
            wav = user_in[5:].strip().strip('"').strip("'")
            if not os.path.exists(wav):
                print(f"File not found: {wav}")
                continue
            run_one(wav, prefix="reply_wav")
            continue

        if args.tts_engine != "pyttsx3":
            print("Set --tts_engine pyttsx3 for text input mode.")
            continue

        tmp_wav = Path(tempfile.mkdtemp(prefix="musetalk_tts_")) / "reply.wav"
        try:
            synthesize_pyttsx3(user_in, tmp_wav, args.tts_rate, args.tts_voice)
            run_one(str(tmp_wav), prefix="reply_txt")
        except Exception as exc:
            print(f"TTS failed: {exc}")
        finally:
            shutil.rmtree(tmp_wav.parent, ignore_errors=True)


if __name__ == "__main__":
    main()
