import base64
import numpy as np
import audioop
import webrtcvad
from collections import deque
import asyncio

from faster_whisper import WhisperModel

model_size = "large-v3"

# Initialize the VAD.
vad = webrtcvad.Vad()
vad.set_mode(0)  # Set VAD's aggressiveness mode.

# Create a queue for audio chunks.
audio_queue = deque()
frames = 0

async def process_speech_chunk(base64_audio):
    """
    Decode base64 audio, convert it to linear PCM, and add it to the queue.
    """
    # Decode base64 audio to bytes.
    audio_bytes = base64.b64decode(base64_audio)
    # Convert µ-law to linear PCM.
    audio_pcm = audioop.ulaw2lin(audio_bytes, 2)
    # Add audio chunk to the queue.
    audio_queue.append(audio_pcm)
    # Process audio chunks from the queue.
    await analyze_audio_queue()

async def analyze_audio_queue():
    """
    Analyze audio chunks in the queue for voice activity.
    """
    # queue_len = len(audio_queue)
    # print(f"the queue length is currently {queue_len}")
    while audio_queue:
        global frames
        audio_chunk = audio_queue.popleft()
        is_speech = vad.is_speech(audio_chunk, 8000)  # Sample rate must match the audio data.
        if is_speech:
            frames += 1
            if frames > 20:
                print("Voice activity detected.")
                frames = 0
        else:
            frames = 0
