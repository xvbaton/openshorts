"""Measure clip selection from a verified transcript, without ASR or rendering."""
import argparse
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

import httpx

import main
from jsonfile import write_json_atomic


def benchmark():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("transcript_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--seconds", type=int, default=240,
                        help="Prefix duration in seconds, or 0 for the full transcript")
    parser.add_argument("--think", action="store_true", help="Enable reasoning in scoring")
    args = parser.parse_args()
    if args.seconds < 0:
        parser.error("--seconds must be zero or positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("use an empty output directory for fresh measurements")

    os.environ["LLM_PROVIDER"] = "ollama"
    os.environ["LLM_MODEL"] = main.llm_backend.QWEN_MODEL
    os.environ["LLM_BASE_URL"] = os.environ.get("LLM_BASE_URL") or "http://127.0.0.1:11434"
    os.environ["LLM_SCORE_THINK"] = "1" if args.think else "0"

    duration = float(subprocess.check_output([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(args.video),
    ], text=True))
    transcript = main.load_transcript_checkpoint(args.transcript_dir, args.video, duration)
    if transcript is None:
        parser.error("no valid transcript checkpoint for the supplied video")
    if args.seconds and args.seconds < duration:
        duration = float(args.seconds)
        transcript["segments"] = [s for s in transcript["segments"] if s["end"] <= duration]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    result = main.get_viral_clips(transcript, duration, output_dir=args.output_dir)
    elapsed = time.monotonic() - started

    base = main.llm_backend.base_url().removesuffix("/v1")
    response = httpx.get(f"{base}/api/ps", timeout=5)
    response.raise_for_status()
    model = next((entry for entry in response.json()["models"]
                  if entry["name"] == main.llm_backend.QWEN_MODEL), None)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    stages = {}
    shortlist = []
    scores = {}
    for path in sorted((args.output_dir / ".clip-analysis").glob("*.json")):
        rows, costs = json.loads(path.read_text())
        stage = path.stem.rsplit("-", 1)[0]
        if stage == "shortlist":
            shortlist = [row["id"] for row in rows]
            continue
        if stage == "score":
            scores.update({row["id"]: row["score"] for row in rows})
        stages.setdefault(stage, []).extend(costs)
    report = {
        "duration_seconds": duration, "seconds": elapsed, "score_think": args.think,
        "scores": dict(sorted(scores.items())), "shortlist": shortlist,
        "clips": result["shorts"] if result else [], "stages": stages,
        "ollama_loaded_model": model,
        "python_peak_rss_bytes": rss if sys.platform == "darwin" else rss * 1024,
    }
    write_json_atomic(args.output_dir / "measurements.json", report, indent=2, ensure_ascii=False)
    print(f"Selection: {elapsed:.2f}s, {len(report['clips'])} clip(s), "
          f"report: {args.output_dir / 'measurements.json'}")


if __name__ == "__main__":
    benchmark()
