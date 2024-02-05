import os
import time
import torch
import torchaudio
import queue
import threading
import sounddevice as sd
import numpy as np
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

buffer_size = 1000
audio_buffer = queue.Queue(maxsize=buffer_size)

print("Loading model...")
config = XttsConfig()
config.load_json("xtts_models/v2.0.2/config.json")
model = Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_dir="xtts_models/v2.0.2/", use_deepspeed=True)
model.cuda()

print("Computing speaker latents...")
gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=["isa_sample.wav"])

def audio_callback(outdata, frames, time, status):
    try:
        chunk = audio_buffer.get_nowait()
        if chunk is None:  # End of data signal
            raise sd.CallbackAbort
        # Reshape the chunk for stereo output by duplicating the mono channel
        chunk_stereo = np.stack((chunk, chunk), axis=-1)
        if len(chunk_stereo) < len(outdata):
            # If the reshaped chunk is smaller than the expected size, fill the rest with zeros
            outdata[:len(chunk_stereo)] = chunk_stereo
            outdata[len(chunk_stereo):] = 0
            raise sd.CallbackStop
        else:
            outdata[:] = chunk_stereo[:len(outdata)]
    except queue.Empty:
        # If the buffer is empty, fill with zeros (silence)
        outdata.fill(0)
        print("Buffer underflow!")



def producer(audio_buffer, text_to_synthesize):
    print("Inference...")
    t0 = time.time()
    chunks = model.inference_stream(
        text_to_synthesize,
        "en",
        gpt_cond_latent,
        speaker_embedding
    )

    for i, chunk in enumerate(chunks):
        if i == 0:
            print(f"Time to first chunck: {time.time() - t0}")
        print(f"Received chunk {i} of audio length {chunk.shape[-1]}")
        audio_numpy = chunk.cpu().numpy().astype("float32")
        audio_buffer.put(audio_numpy)
    audio_buffer.put(None)

def consumer():
    """
    Sets up and starts the OutputStream for audio playback.
    """
    with sd.OutputStream(callback=audio_callback, samplerate=24000, blocksize=24000, dtype='float32'):
        print("Streaming started.")
        input("Press Enter to stop playback...")

def main(text_to_synthesize):
    producer_thread = threading.Thread(target=producer, args=(audio_buffer, text_to_synthesize))
    consumer_thread = threading.Thread(target=consumer)

    producer_thread.start()
    consumer_thread.start()

    producer_thread.join()
    consumer_thread.join()

if __name__ == "__main__":
    text = "I see the world being slowly transformed into a wilderness; I hear the approaching thunder that, one day, will destroy us too. I feel the suffering of millions."
    main(text)