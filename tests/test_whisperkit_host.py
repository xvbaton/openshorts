import json
import os
import sys
import threading
import time

import httpx
import pytest

import transcribe_backends as tb


REPORT = {
    "text": "Salut Zelda",
    "language": "fr",
    "segments": [{"text": "Salut Zelda", "start": 0.0, "end": 1.5,
                  "words": [{"word": " Salut", "start": 0.1, "end": 0.4, "probability": 0.9},
                            {"word": " Zelda", "start": 0.5, "end": 1.35}]}],
    "timings": {"modelLoading": 0.5},
}


def test_native_response_waits_for_process_exit_and_keeps_full_report(tmp_path, monkeypatch):
    from whisperkit_host import serve

    pid_file = tmp_path / "pid"
    cli = tmp_path / "argmax-cli"
    cli.write_text(f"""#!{sys.executable}
import os, pathlib, sys
pathlib.Path({str(pid_file)!r}).write_text(str(os.getpid()))
audio = pathlib.Path(sys.argv[sys.argv.index("--audio-path") + 1])
assert audio.read_bytes() == b"test WAV"
assert "--word-timestamps" in sys.argv
assert "--incremental-loading" in sys.argv
audio.with_suffix(".json").write_text({json.dumps(REPORT)!r})
import time
time.sleep(0.2)
""")
    cli.chmod(0o700)
    wav = tmp_path / "input.wav"
    wav.write_bytes(b"test WAV")
    monkeypatch.setattr(tb, "_has_audio_stream", lambda _: True)
    monkeypatch.setattr(tb, "_extract_wav", lambda _: wav)
    monkeypatch.setenv("TRANSCRIBE_BACKEND", "whisperkit")
    monkeypatch.setenv("WHISPERKIT_MODEL", "small")
    monkeypatch.setenv("WHISPERKIT_LANGUAGE", "fr")
    with serve(cli, tmp_path / "small", port=0, timeout=5) as server:
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        try:
            monkeypatch.setenv("WHISPERKIT_URL", f"http://127.0.0.1:{server.server_port}/transcribe")
            result = tb.transcribe_media("video.mp4")
        finally:
            server.shutdown()
            thread.join()
    with pytest.raises(ProcessLookupError):
        os.kill(int(pid_file.read_text()), 0)
    assert result["segments"] == REPORT["segments"]
    assert result["timings"] == REPORT["timings"]
    assert result["metadata"]["backend"] == "whisperkit"
    assert result["metadata"]["task"] == "transcribe"
    assert result["metadata"]["requested_language"] == "fr"
    assert not wav.exists()


def test_server_reaps_a_timed_out_cli_before_replying(tmp_path):
    from whisperkit_host import serve

    pid_file = tmp_path / "pid"
    cli = tmp_path / "argmax-cli"
    cli.write_text(f"""#!{sys.executable}
import os, pathlib, time
pathlib.Path({str(pid_file)!r}).write_text(str(os.getpid()))
time.sleep(30)
""")
    cli.chmod(0o700)
    with serve(cli, tmp_path / "small", port=0, timeout=1) as server:
        thread = threading.Thread(target=server.handle_request)
        thread.start()
        try:
            response = httpx.post(f"http://127.0.0.1:{server.server_port}/transcribe",
                                  content=b"test WAV", headers={"Content-Type": "audio/wav"},
                                  trust_env=False, timeout=5)
        finally:
            thread.join()
    assert response.status_code == 504
    with pytest.raises(ProcessLookupError):
        os.kill(int(pid_file.read_text()), 0)


def test_busy_server_does_not_queue_a_second_native_process(tmp_path):
    from whisperkit_host import serve

    started, release = tmp_path / "started", tmp_path / "release"
    cli = tmp_path / "argmax-cli"
    cli.write_text(f"""#!{sys.executable}
import pathlib, sys, time
pathlib.Path({str(started)!r}).touch()
while not pathlib.Path({str(release)!r}).exists():
    time.sleep(0.02)
audio = pathlib.Path(sys.argv[sys.argv.index("--audio-path") + 1])
audio.with_suffix(".json").write_text({json.dumps(REPORT)!r})
""")
    cli.chmod(0o700)
    with serve(cli, tmp_path / "small", port=0, timeout=5) as server:
        server_thread = threading.Thread(target=server.serve_forever)
        server_thread.start()
        url = f"http://127.0.0.1:{server.server_port}/transcribe"
        responses = []

        def post(timeout):
            return httpx.post(url, content=b"test WAV", headers={"Content-Type": "audio/wav"},
                              trust_env=False, timeout=timeout)

        first = threading.Thread(target=lambda: responses.append(post(6)))
        first.start()
        try:
            deadline = time.monotonic() + 3
            while not started.exists() and time.monotonic() < deadline:
                time.sleep(0.02)
            assert started.exists()
            assert post(0.5).status_code == 503
        finally:
            release.touch()
            first.join()
            server.shutdown()
            server_thread.join()
    assert responses[0].status_code == 200


@pytest.mark.parametrize("response", [httpx.Response(503), httpx.Response(200, json={
    **REPORT, "segments": [{**REPORT["segments"][0], "words": []}],
})])
def test_native_failure_falls_back_to_faster_whisper_cpu(response, monkeypatch, tmp_path):
    wav = tmp_path / "input.wav"
    wav.write_bytes(b"test WAV")
    monkeypatch.setenv("TRANSCRIBE_BACKEND", "whisperkit")
    monkeypatch.setenv("WHISPER_DEVICE", "cuda")
    monkeypatch.setattr(tb, "_extract_wav", lambda _: wav)
    monkeypatch.setattr(tb, "_has_audio_stream", lambda _: True)
    client = httpx.Client(transport=httpx.MockTransport(lambda _: response))
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: client)
    seen = []

    def cpu(path, *, force_cpu=False):
        seen.append((path, force_cpu))
        return REPORT

    monkeypatch.setattr(tb, "_transcribe_with_whisper", cpu)
    assert tb.transcribe_media("video.mp4") == REPORT
    assert seen == [("video.mp4", True)]
    assert not wav.exists()
