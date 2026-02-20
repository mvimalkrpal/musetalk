#!/usr/bin/env python3
import argparse
import math
import struct
import wave
from pathlib import Path

import requests


def make_tone_wav(path: Path, sr: int = 16000, dur_s: float = 1.0, hz: float = 440.0) -> None:
    with wave.open(str(path), "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        for i in range(int(sr * dur_s)):
            sample = int(12000 * math.sin(2 * math.pi * hz * i / sr))
            w.writeframes(struct.pack("<h", sample))


def main() -> None:
    p = argparse.ArgumentParser(description="Test Mac -> Windows MuseTalk /generate endpoint")
    p.add_argument("--server", required=True, help="Example: http://markinova.local:8787")
    p.add_argument("--out", default="/tmp/reply.mp4")
    p.add_argument("--prefix", default="manual")
    args = p.parse_args()

    wav_path = Path("/tmp/test_remote_tone.wav")
    make_tone_wav(wav_path)

    url = args.server.rstrip("/") + "/generate"
    with open(wav_path, "rb") as fp:
        resp = requests.post(
            url,
            files={"audio": ("test.wav", fp, "audio/wav")},
            data={"output_prefix": args.prefix},
            timeout=1800,
        )

    print("status:", resp.status_code)
    print("content-type:", resp.headers.get("content-type"))
    out_path = Path(args.out)
    out_path.write_bytes(resp.content)
    print("saved:", out_path)


if __name__ == "__main__":
    main()
