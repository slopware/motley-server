import os
import time
import torch
import torchaudio
import queue
import threading
import sounddevice as sd
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

buffer_size = 10
audio_buffer = queue.Queue(maxsize=buffer_size)

print("Loading model...")
config = XttsConfig()
config.load_json("xtts_models/v2.0.2/config.json")
model = Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_dir="xtts_models/v2.0.2/", use_deepspeed=True)
model.cuda()

print("Computing speaker latents...")
gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=["isa_sample.wav"])

def producer(audio_buffer, text_to_synthesize):
    print("Inference...")
    t0 = time.time()
    chunks = model.inference_stream(
        text_to_synthesize,
        "en",
        gpt_cond_latent,
        speaker_embedding,
        stream_chunk_size=10
    )

    for i, chunk in enumerate(chunks):
        if i == 0:
            print(f"Time to first chunck: {time.time() - t0}")
        print(f"Received chunk {i} of audio length {chunk.shape[-1]}")
        audio_numpy = chunk.cpu().numpy().astype("float32")
        audio_buffer.put(audio_numpy)
    audio_buffer.put(None)

def consumer(audio_buffer):
    while True:
        chunk = audio_buffer.get()
        if chunk is None:
            break
        sd.play(chunk, 24000)
        sd.wait()

def main(text_to_synthesize):
    producer_thread = threading.Thread(target=producer, args=(audio_buffer, text_to_synthesize))
    consumer_thread = threading.Thread(target=consumer, args=(audio_buffer,))

    producer_thread.start()
    consumer_thread.start()

    producer_thread.join()
    consumer_thread.join()

if __name__ == "__main__":
    text = "I see the world being slowly transformed into a wilderness; I hear the approaching thunder that, one day, will destroy us too. I feel the suffering of millions. And yet, when I look up at the sky, I somehow feel that everything will change for the better, that this cruelty too shall end, that peace and tranquility will return once more."
    main(text)