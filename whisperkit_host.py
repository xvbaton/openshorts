"""Loopback bridge to a one-shot WhisperKit CLI on the Mac."""
import argparse
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import subprocess
import tempfile
import threading
import time

from transcript_cache import valid_transcript


_MAX_WAV_BYTES = 128 * 1024 * 1024


class TranscriptionHandler(BaseHTTPRequestHandler):
    timeout = 30

    def __init__(self, *args, cli, model_path, inference_timeout, slot, **kwargs):
        self.cli = cli
        self.model_path = model_path
        self.inference_timeout = inference_timeout
        self.slot = slot
        super().__init__(*args, **kwargs)

    def do_POST(self):
        if self.path != "/transcribe":
            return self.send_error(404)
        if self.headers.get_content_type() != "audio/wav":
            return self.send_error(415)
        try:
            remaining = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            return self.send_error(400)
        if remaining <= 0:
            return self.send_error(411)
        if remaining > _MAX_WAV_BYTES:
            return self.send_error(413)
        if not self.slot.acquire(blocking=False):
            return self.send_error(503, "WhisperKit is already transcribing")
        try:
            self._respond_with_transcript(remaining)
        finally:
            self.slot.release()

    def _respond_with_transcript(self, remaining):
        with tempfile.TemporaryDirectory(prefix="clipin-whisperkit-") as directory:
            wav = Path(directory) / "input.wav"
            try:
                with wav.open("wb") as audio:
                    while remaining:
                        chunk = self.rfile.read(min(remaining, 128 * 1024))
                        if not chunk:
                            return self.send_error(400)
                        audio.write(chunk)
                        remaining -= len(chunk)
                started = time.monotonic()
                subprocess.run([
                    self.cli, "transcribe", "--model-path", self.model_path,
                    "--audio-path", str(wav), "--report", "--report-path", directory,
                    "--word-timestamps", "--skip-special-tokens", "--incremental-loading",
                    "--audio-encoder-compute-units", "cpuAndGPU",
                    "--text-decoder-compute-units", "cpuAndGPU",
                ], check=True, timeout=self.inference_timeout, stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                report = json.loads(wav.with_suffix(".json").read_text(encoding="utf-8"))
                if not valid_transcript(report):
                    return self.send_error(502, "Invalid WhisperKit report")
                report["metadata"] = {
                    "backend": "whisperkit", "model": Path(self.model_path).name,
                    "compute_units": "cpuAndGPU", "audio_loading": "incremental",
                    "wall_seconds": time.monotonic() - started,
                }
                body = json.dumps(report, ensure_ascii=False, allow_nan=False).encode("utf-8")
            except subprocess.TimeoutExpired:
                return self.send_error(504, "WhisperKit deadline exceeded")
            except (OSError, subprocess.CalledProcessError, ValueError):
                return self.send_error(502, "WhisperKit transcription failed")

        # subprocess.run has reaped the model process before the caller can score.
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def serve(cli, model_path, port=50060, timeout=600):
    handler = partial(TranscriptionHandler, cli=cli, model_path=model_path,
                      inference_timeout=timeout, slot=threading.Lock())
    return ThreadingHTTPServer(("127.0.0.1", port), handler)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cli", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--port", type=int, default=50060)
    args = parser.parse_args()
    if not args.cli.is_file() or not args.model_path.is_dir():
        parser.error("--cli must be a file and --model-path must be a directory")
    with serve(args.cli.resolve(), args.model_path.resolve(), args.port) as server:
        print(f"WhisperKit listening on http://127.0.0.1:{server.server_port}/transcribe", flush=True)
        server.serve_forever()
