"""Identity and validation shared by the two transcript caches."""
import hashlib
import math


SCHEMA_VERSION = 1


def media_sha256(path):
    with open(path, "rb") as media:
        return hashlib.file_digest(media, "sha256").hexdigest()


def valid_span(item):
    if not isinstance(item, dict):
        return False
    start, end = item.get("start"), item.get("end")
    return (type(start) in (int, float) and type(end) in (int, float)
            and 0 <= start <= end < math.inf)


def valid_transcript(transcript):
    if not isinstance(transcript, dict):
        return False
    text, language = transcript.get("text"), transcript.get("language")
    segments = transcript.get("segments")
    if (not isinstance(text, str) or not isinstance(language, str) or not language
            or not isinstance(segments, list) or (not segments and text.strip())):
        return False
    for segment in segments:
        if not valid_span(segment) or not isinstance(segment.get("text"), str):
            return False
        words = segment.get("words")
        if not isinstance(words, list) or not all(
            valid_span(word) and isinstance(word.get("word"), str) for word in words
        ):
            return False
    return True


def normalize_whisperkit_timestamps(transcript):
    """Resolve overlaps between segments without changing the recognized words."""
    metadata = transcript.get("metadata")
    if not isinstance(metadata, dict) or metadata.get("backend") != "whisperkit":
        return transcript

    segments = transcript["segments"]
    for previous, following in zip(segments, segments[1:]):
        words, next_words = previous["words"], following["words"]
        if not words or not next_words:
            continue
        boundary = min(following["start"], next_words[0]["start"])
        if boundary <= max(previous["start"], words[0]["start"]):
            continue  # The preceding segment has no room before this boundary.

        word_end = words[-1]["end"]
        if word_end > boundary:
            first = next(i for i, word in enumerate(words) if word["end"] > boundary)
            while words[first]["start"] >= boundary:
                first -= 1
            anchor = words[first]["start"]
            # Keep the following segment fixed. If several earlier words
            # cross its boundary, compress only that suffix in text order.
            scale = (boundary - anchor) / (word_end - anchor)
            for word in words[first:]:
                start, end = word["start"], word["end"]
                word.setdefault("raw_start", start)
                word.setdefault("raw_end", end)
                word["start"] = min(boundary, anchor + (start - anchor) * scale)
                word["end"] = min(boundary, anchor + (end - anchor) * scale)
            words[-1]["end"] = boundary
        if previous["end"] > boundary:
            previous.setdefault("raw_end", previous["end"])
            previous["end"] = boundary
    return transcript
