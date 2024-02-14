import os
import sys
import time
import torch
import torchaudio
import queue
import pyaudio
import threading

import numpy as np

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

print("Loading model...")
config = XttsConfig()
print(f"Sample rate: {config.audio.output_sample_rate}")
config.load_json("xtts_models/v2.0.2/config.json")
model = Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_dir="xtts_models/v2.0.2/", use_deepspeed=True)
model.cuda()

print("Computing speaker latents...")
gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=["isa_sample.wav"])

buffer = queue.Queue()

def postprocess_wave(chunk):
    """Post process the output waveform"""
    if isinstance(chunk, list):
        chunk = torch.cat(chunk, dim=0)
    chunk = chunk.clone().detach().cpu().numpy()
    chunk = chunk[None, : int(chunk.shape[0])]
    chunk = np.clip(chunk, -1, 1)
    chunk = chunk.astype(np.float32)
    return chunk

def tts_stream(buffer, text, sub_chunk_size):
    print("Inference...")
    t0 = time.time()
    chunks = model.inference_stream(
        text,
        "en",
        gpt_cond_latent,
        speaker_embedding
    )

    for i, chunk in enumerate(chunks):
        if i == 0:
            print(f"Time to first chunk: {time.time() - t0}")
        print(f"Received chunk {i} of audio length {chunk.shape[-1]}")
        chunk = postprocess_wave(chunk)
        buffer.put(chunk)

p = pyaudio.PyAudio()

stream = p.open(format=pyaudio.paFloat32,  # This should match the dtype of `chunk` (np.float32)
                channels=1,  # Mono audio
                rate=24000,  # This should match the sample rate of your model's output
                output=True)

def play_audio(buffer):
    while True:
        chunk = buffer.get()
        if chunk is None:
            break
        stream.write(chunk.tobytes(), exception_on_underflow=False)
        buffer.task_done()


synthesis = threading.Thread(target=tts_stream, args=(buffer, "This is a test. I am synthesizing audio.", 1024))
audio_player = threading.Thread(target=play_audio, args=(buffer,))

synthesis.start()
audio_player.start()

synthesis.join()
#since queue.get() is blocking,
buffer.put(None)
audio_player.join()

stream.stop_stream()
stream.close()