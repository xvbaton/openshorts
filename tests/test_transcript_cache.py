from copy import deepcopy

import pytest

from transcript_cache import normalize_whisperkit_timestamps, valid_transcript


def test_whisperkit_boundary_repair_keeps_every_word_and_the_unaffected_times():
    transcript = {
        "language": "fr", "text": "texte complet", "metadata": {"backend": "whisperkit"},
        "segments": [
            {"start": 1, "end": 6, "text": "premier passage", "words": [
                {"word": " bon", "start": 1, "end": 2},
                {"word": " début", "start": 3, "end": 4},
                {"word": " de", "start": 4, "end": 4.5},
                {"word": " phrase", "start": 4.5, "end": 6, "probability": 0.9},
            ]},
            {"start": 3.26, "end": 7, "text": "suite", "words": [
                {"word": " suite", "start": 3.26, "end": 4},
                {"word": " fin", "start": 5, "end": 7},
            ]},
        ],
    }
    original = deepcopy(transcript)
    fixed = normalize_whisperkit_timestamps(transcript)
    previous, following = fixed["segments"]

    assert valid_transcript(fixed)
    assert following == original["segments"][1]
    assert previous["words"][0] == original["segments"][0]["words"][0]
    assert previous["end"] == previous["words"][-1]["end"] == following["start"]
    assert [w["word"] for w in previous["words"]] == [w["word"] for w in original["segments"][0]["words"]]
    assert all(w["end"] > w["start"] for w in previous["words"])
    assert previous["words"][-1]["probability"] == 0.9
    assert previous["words"][-1]["raw_start"] == pytest.approx(4.5)
    assert fixed["text"] == original["text"]
    once = deepcopy(fixed)
    assert normalize_whisperkit_timestamps(fixed) == once


def test_other_engines_keep_their_timestamps():
    transcript = {"metadata": {"backend": "faster-whisper"}, "segments": [
        {"start": 1, "end": 3, "words": [{"start": 1, "end": 3}]},
        {"start": 2, "end": 4, "words": [{"start": 2, "end": 4}]},
    ]}
    original = deepcopy(transcript)
    assert normalize_whisperkit_timestamps(transcript) == original
