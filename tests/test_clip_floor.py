"""The clip floor must survive a model that ignores it.

`min_clips` lived only in the detail prompt, so a run that came back with
half the requested clips shipped half the clips: the measured 9:10 source
asked for 6-12 and delivered 3-4. The windows the detail pass skipped get
one more call before the job settles for less.
"""
import json

import pytest

main = pytest.importorskip("main")


def _transcript(n_segments=40):
    segments = []
    for i in range(n_segments):
        start = i * 15.0
        segments.append({
            "start": start, "end": start + 15.0,
            "text": f"sentence number {i} with enough words to score",
            "words": [{"word": f"w{i}", "start": start, "end": start + 0.4}],
        })
    return {"language": "es", "segments": segments}


def _clip(start, window_id, score=80):
    return {
        "start": start, "end": start + 30.0, "source_window_id": window_id,
        "predicted_score": score, "video_description_for_tiktok": "d",
        "video_description_for_instagram": "d",
        "video_title_for_youtube_short": "t", "viral_hook_text": "h",
        "why": "because",
    }


def _run(monkeypatch, detail_responses):
    """Drive get_viral_clips with canned stage results; returns (clips, calls)."""
    calls = []

    def fake_stage(client, model, prompt, schema):
        payload = json.loads(prompt.split("WINDOWS_JSON:")[-1].split("Return only:")[0]
                             if "WINDOWS_JSON:" in prompt else "[]")
        if schema is main.gemini_worker.ScoreResponse:
            ids = [w["id"] for w in payload]
            calls.append(("score", ids))
            # The prompt asks for every window to be scored.
            return {"windows": [{"id": i, "start": 0, "end": 90, "score": 90 - n,
                                 "reason": "r"}
                                for n, i in enumerate(ids)]}, None
        windows = json.loads(prompt.split("CANDIDATE_WINDOWS_JSON:")[-1]
                             .split("Return only:")[0])
        ids = [w["id"] for w in windows]
        calls.append(("detail", ids))
        return {"shorts": detail_responses.pop(0)}, None

    monkeypatch.setattr(main, "_run_gemini_stage", fake_stage)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(main.genai, "Client", lambda api_key=None: object())
    out = main.get_viral_clips(_transcript(), 600.0)
    return (out or {}).get("shorts", []), calls


def test_a_short_detail_result_asks_the_unused_windows(monkeypatch):
    # Two clips back on a shortlist that asked for six: the four windows that
    # produced nothing get a second call, and its clips are kept.
    first = [_clip(0, "window_001"), _clip(120, "window_003")]
    second = [_clip(240, "window_005"), _clip(300, "window_006"),
              _clip(360, "window_007"), _clip(420, "window_008")]
    clips, calls = _run(monkeypatch, [first, second])

    detail_calls = [c for c in calls if c[0] == "detail"]
    assert len(detail_calls) == 2, "the floor never triggered a second call"
    # The retry only revisits windows the first pass returned nothing for.
    assert "window_001" not in detail_calls[1][1]
    assert "window_003" not in detail_calls[1][1]
    assert len(clips) >= 6
    assert [c["start"] for c in clips] == sorted(c["start"] for c in clips)


def test_no_second_call_when_the_floor_is_met(monkeypatch):
    enough = [_clip(i * 60, f"window_{i + 1:03d}") for i in range(6)]
    clips, calls = _run(monkeypatch, [enough])
    assert len([c for c in calls if c[0] == "detail"]) == 1
    assert len(clips) == 6


def test_an_empty_retry_is_accepted_rather_than_padded(monkeypatch):
    # Material that genuinely holds two clips still returns two, not filler.
    clips, calls = _run(monkeypatch, [[_clip(0, "window_001")], []])
    assert len([c for c in calls if c[0] == "detail"]) == 2
    assert len(clips) == 1


def test_scoring_fills_the_shortlist_it_targets(monkeypatch):
    # 600s -> target 8. Every window is scored, so the shortlist is the top 8
    # globally instead of the 3-per-batch a constant cap allowed.
    clips, calls = _run(monkeypatch, [[_clip(i * 60, f"window_{i + 1:03d}")
                                       for i in range(6)]])
    detail_windows = [c[1] for c in calls if c[0] == "detail"][0]
    assert len(detail_windows) == 8


def test_qwen_resumes_after_a_failed_detail_batch(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("LLM_BASE_URL", "http://llm.test")
    monkeypatch.setenv("LLM_MODEL", "qwen3.5:4b")
    monkeypatch.setenv("LLM_SCORE_BATCH", "99")
    fail_once = True

    def stage(client, model, prompt, schema):
        nonlocal fail_once
        scoring = schema is main.gemini_worker.ScoreResponse
        windows = json.loads(prompt.split("WINDOWS_JSON:")[-1].split("Return only:")[0])
        calls.append(("score" if scoring else "detail", tuple(w["id"] for w in windows)))
        assert len(windows) <= (3 if scoring else 2)
        if scoring:
            return {"windows": [dict(w, score=80, reason="hook") for w in windows]}, {"input_tokens": 10}
        if fail_once and sum(stage == "detail" for stage, _ in calls) == 2:
            fail_once = False
            raise main.TransientLLMError("Transient LLM error: ReadTimeout")
        return {"shorts": [_clip(w["start"], w["id"]) for w in windows]}, {"input_tokens": 20}

    monkeypatch.setattr(main, "_run_gemini_stage", stage)
    with pytest.raises(main.TransientLLMError):
        main.get_viral_clips(_transcript(), 600.0, output_dir=tmp_path)
    checkpoints = tmp_path / ".clip-analysis"
    assert len(list(checkpoints.glob("score-*.json"))) == 4
    assert len(list(checkpoints.glob("shortlist-*.json"))) == 1
    assert len(list(checkpoints.glob("detail-*.json"))) == 1
    failed = calls[-1]
    calls.clear()
    result = main.get_viral_clips(_transcript(), 600.0, output_dir=tmp_path)
    assert calls[0] == failed
    assert all(kind == "detail" for kind, _ in calls)
    assert len(result["shorts"]) == 8
    assert result["cost_analysis"]["input_tokens"] == 120

    calls.clear()
    assert main.get_viral_clips(_transcript(), 600.0, output_dir=tmp_path) == result
    assert calls == []

    next(checkpoints.glob("detail-*.json")).write_text("null")
    assert main.get_viral_clips(_transcript(), 600.0, output_dir=tmp_path) == result
    assert [kind for kind, _ in calls] == ["detail"]

    calls.clear()
    monkeypatch.setenv("LLM_SCORE_THINK", "1")
    main.get_viral_clips(_transcript(), 600.0, output_dir=tmp_path)
    assert any(kind == "score" for kind, _ in calls)
