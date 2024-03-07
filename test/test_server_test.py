import argparse
import json
import shutil
import subprocess
import sys
import time
from typing import Iterator

import requests


def stream_ffplay(audio_stream):

    ffplay_cmd = ["ffplay", "-nodisp", "-probesize", "1024", "-autoexit", "-"]


    ffplay_proc = subprocess.Popen(ffplay_cmd, stdin=subprocess.PIPE)
    for chunk in audio_stream:
        if chunk is not None:
            ffplay_proc.stdin.write(chunk)

    # close on finish
    ffplay_proc.stdin.close()
    ffplay_proc.wait()


def tts(text) -> Iterator[bytes]:
    start = time.perf_counter()
    speaker = dict()
    speaker["text"] = text
    speaker["language"] = "en"
    speaker["stream_chunk_size"] = "20"  # you can reduce it to get faster response, but degrade quality
    res = requests.post(
        "http://localhost:8000/tts_stream",
        json=speaker,
        stream=True,
    )
    end = time.perf_counter()
    print(f"Time to make POST: {end-start}s", file=sys.stderr)

    if res.status_code != 200:
        print("Error:", res.text)
        sys.exit(1)

    first = True
    for chunk in res.iter_content(chunk_size=512):
        if first:
            end = time.perf_counter()
            print(f"Time to first chunk: {end-start}s", file=sys.stderr)
            first = False
        if chunk:
            yield chunk

    print("⏱️ response.elapsed:", res.elapsed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--text",
        default="It took me quite a long time to develop a voice and now that I have it I am not going to be silent.",
        help="text input for TTS"
    )
    args = parser.parse_args()
    audio = stream_ffplay(
        tts(
            args.text
        )
    )
