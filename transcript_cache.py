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
