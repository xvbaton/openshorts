# Clip selection

## How many clips a job returns (`clip_selection.py`, `main.get_viral_clips`)

The count is not a setting, it is derived. `get_viral_clips` builds ~90 s scoring
windows over the transcript, scores them in batches, shortlists the best
`shortlist_target` (`min(10, duration//90 + 2)`, floor 3) and asks the detail pass
for `clip_count_targets(len(shortlist))` clips. The floor matters: jobs that
return only 1-3 clips retain far worse than jobs that return 4-9.

Two invariants keep that chain from leaking:

- **The scoring pass ranks, it does not select.** The prompt scores **every**
  window and the shortlist is the global top N. A prompt that says "choose up to
  3 windows from this batch" caps the shortlist at `3 * n_batches` instead of the
  target and starves every source under ~30 min. Batches are near-equal
  (`score_batches`): a trailing batch of one window would return that window
  whatever its score.
- **The clip floor is enforced in code, not just in the prompt.** When the detail
  pass comes back under `min_clips`, the shortlist windows that produced nothing
  get one more call for the difference. An empty answer is accepted: padding is
  worse than a short list.

`target_clips` on `/api/process` (dashboard: advanced options) pins both ends via
`CLIP_TARGET_MIN` / `CLIP_TARGET_MAX` when the user wants an exact number.

## Silent footage: the vision fallback (`main.get_visual_clips`)

`transcribe_video` raising `NoAudioError`, **or** `speech_is_sparse()` returning
true, both set `transcript = None` and route to `get_visual_clips`. Sparse means
under `MIN_SPEECH_WORDS` (8) in total or under `MIN_SPEECH_WORDS_PER_MIN` (5).
Music-only footage and mic-muted recordings transcribe to a handful of stray
words; without the second test the picker scores those words and cuts around
them.

The vision pass uploads the video, Gemini returns the same `{"shorts"}` shape
(`gemini_worker.VisualResponse`) in the same 15-60 s band, so every later stage
is identical. `CLIP_TARGET_MIN`/`MAX` apply directly (there are no scoring
windows). The transcript is stored as `{"language": "none", "segments": []}`, so
the clips have no subtitles, which is correct.

**This is the one stage that sends Gemini the video instead of frames, on
purpose**: frames can say what kind of video it is, not which 40 seconds to cut.
Gemini counts video at ~300 tokens/second, so an hour is ~1.08M tokens, past a 1M
window. A source longer than `VISION_SEGMENT_SECONDS` (2400) is therefore split by
`clip_selection.vision_segments` into near-equal stretches (no one-minute tail),
each re-encoded to a 360p / 1 fps proxy (`_vision_proxy`: re-encoded, not
stream-copied, or every timestamp would be off by up to a GOP), watched on its
own, offset back to source time, then merged with `dedupe_overlapping` +
`trim_to_best` to `CLIP_TARGET_MAX`. One failed stretch is logged and skipped.
`VISION_MAX_SEGMENTS` (8, ~5 h 20) bounds the spend of one job; past it only the
first stretches are watched. A source that fits one stretch is one stretch: it
goes through the same proxy, never as the raw file (a broken capture, such as an
HLS VOD whose audio will not decode, made Gemini answer 400 INVALID_ARGUMENT).
When the proxy's audio cannot be encoded it is retried without audio. Do not
port the frame trick here. Gemini-only: a
text-only `LLM_BASE_URL` server cannot see footage, and without `GEMINI_API_KEY`
the function logs one line and returns None, which fails the job.

The public `/gta-5-clips`, `/gta-6-clips`, `/wow-forever-clips` and `/platforms/twitch` pages state
these thresholds and the stretch sizes; if the behaviour changes, change
`dashboard/seo/pages.js` and `dashboard/seo/clip-tools.js` too.

## Local LLM for the moment picker (`llm_backend.py`)

`LLM_BASE_URL` (+ `LLM_MODEL`, `LLM_API_KEY`) routes the two transcript passes of
`get_viral_clips` to any OpenAI-compatible `/chat/completions` instead of Gemini;
the response is validated with the same pydantic schemas Gemini enforces
server-side, so `main.py` sees one shape. `main.score_batch_size` drops to 3
windows per call there (local contexts are 4-8k; a truncated prompt scores garbage
silently). Self-host `/api/process` then accepts a request without
`X-Gemini-Key` and `/api/config.localLlm` tells the dashboard not to demand one.
Frame-based stages (`layout_picker`, `screencast_layout`, `get_visual_clips`)
stay on Gemini. Never wired in cloud mode: `BILLING_ENABLED` ignores it.

### CLIPIN Qwen 4B

In the backend environment, set `LLM_PROVIDER=ollama`,
`LLM_MODEL=qwen3.5:4b`, and `LLM_BASE_URL=http://host.docker.internal:11434`
for Docker or `http://127.0.0.1:11434` on the host. An existing URL ending in
`/v1` works with this profile. The model defaults to `qwen3.5:4b` in the
Ollama profile, including when the older untagged `qwen3.5` is configured.
Requests use Ollama's native chat endpoint. The backend logs both tags and
fails if Ollama reports a different tag, omits it, or truncates the response.

The context is 4,096 tokens. Scoring ranks every window, up to three per
request, with a default output budget of 384 tokens. `LLM_SCORE_BATCH` can
lower that batch size. Detail requests carry at most two windows and produce
up to three accepted clips per batch, with a 1,536-token output budget.
`LLM_SCORE_THINK=1` enables experimental reasoning for scoring only and raises
its default budget to 1,024 tokens. `LLM_SCORE_MAX_TOKENS` and
`LLM_DETAIL_MAX_TOKENS` can adjust the output budgets up to 2,048.
`LLM_TIMEOUT` controls the HTTPX read timeout (600 seconds by default).
Connect and pool timeouts are 10 seconds, and the write timeout is 30 seconds.

Completed score batches, the shortlist, and detail batches are written
atomically under the job's `.clip-analysis/` directory. A restarted or
retried job uses completed batches from the same prompts and settings. HTTPX
timeouts and transient server failures get up to three attempts per call.
The backend can requeue the job once more using its existing retry policy.
An explicit token-limit failure requires reviewing the budget or disabling
reasoning before retrying.

The Zelda benchmark uses the existing source-verified transcript checkpoint.
From the repository root, run the command below with a fresh output
directory. The default excerpt is 240 seconds. Add `--think` for the
reasoning run or `--seconds 0` for full transcript selection.

```bash
PYTHONPATH=. LLM_BASE_URL=http://127.0.0.1:11434 .venv/bin/python bin/benchmark_qwen.py \
  .cache/asr-benchmark/zelda.mp4 .cache/asr-benchmark/native-1 \
  .cache/qwen-benchmark/short-no-think
```

The benchmark saves the per-call timing and token counts, scores, selected
clips, and Ollama model allocation in `measurements.json`. Ollama counts
reasoning within its generated-token total. The Python memory measurement
is peak RSS and includes startup imports.

#### Zelda measurements

Measured on a MacBook Pro M1 Pro with 16 GiB RAM and Ollama 0.40.16 (MLX).
All served requests reported `qwen3.5:4b`. Ollama reported digest
`7f3251fa6878` and a 4,096-token context. Times exclude transcript
verification and Python startup. Model memory is the allocation reported by
`/api/ps` after selection, a point measurement.

| Run | Selection | Scoring | Scoring tokens in / out | Total tokens in / out | Clips | Ollama allocation | Peak Python RSS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 240 s excerpt, reasoning off | 68.02 s | 18.12 s | 2,852 / 300 | 8,574 / 1,689 | 3 | 4.68 GiB | 408 MiB |
| 734.49 s source, reasoning off | 115.68 s | 36.25 s | 7,693 / 825 | 18,437 / 3,233 | 9 | 6.52 GiB | 415 MiB |

On the excerpt, the first scoring batch took 11.19 s without reasoning,
including 3.73 s of model load, and produced 148 tokens and two scores.
With reasoning, the same batch exhausted 1,024 tokens in 28.82 s and 2,048
tokens in 54.49 s. Neither request produced usable scores. These runs do
not establish a ranking-quality comparison.

The full pass scored 11 windows, shortlisted 10, and completed five detail
batches in 79.22 s. It discarded one overlapping proposal. A checkpoint-only
replay took 13 ms and reproduced all nine clips exactly with zero LLM calls.
Transcript review confirmed concrete hooks in several proposals and found
titles with unsupported claims. Editorial quality and cut boundaries still
need review against the source video.
