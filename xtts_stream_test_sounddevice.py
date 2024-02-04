import os
import sys
import time
import torch
import torchaudio
import sounddevice as sd
import queue
import threading
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
import numpy as np
# print(sd.query_devices())
# print("Default Input Device:", sd.default.device['input'])
# print("Default Output Device:", sd.default.device['output'])
audio_buffer = queue.Queue(maxsize=2000)

print("Loading model...")
config = XttsConfig()
config.load_json("xtts_models/v2.0.2/config.json")
model = Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_dir="xtts_models/v2.0.2/", use_deepspeed=True)
model.cuda()

print("Computing speaker latents...")
gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=["isa_sample.wav"])

def tts_stream(audio_buffer, text_to_synthesize, sub_chunk_size=1024):
    print("Inference...")
    t0 = time.time()
    chunks = model.inference_stream(
        "It took me quite a long time to develop a voice and now that I have it I am not going to be silent.",
        "en",
        gpt_cond_latent,
        speaker_embedding
    )

    for i, chunk in enumerate(chunks):
        shape = chunk.shape[-1]
        if i == 0:
            print(f"Time to first chunk: {time.time() - t0}")
        print(f"Received chunk {i} of audio length {chunk.shape[-1]}")
        audio_numpy = chunk.cpu().numpy().astype("float32")
        
        # Calculate the number of sub-chunks needed
        num_sub_chunks = int(np.ceil(len(audio_numpy) / sub_chunk_size))
        
        for j in range(num_sub_chunks):
            start_idx = j * sub_chunk_size
            end_idx = start_idx + sub_chunk_size
            
            # If this is the last sub-chunk and it's not a full sub_chunk_size, pad it with zeros
            if end_idx > len(audio_numpy):
                pad_length = end_idx - len(audio_numpy)
                sub_chunk = np.pad(audio_numpy[start_idx:], (0, pad_length), 'constant')
            else:
                sub_chunk = audio_numpy[start_idx:end_idx]
            
            audio_buffer.put(sub_chunk.flatten())

    # After the last chunk, put a None to signal the end
    audio_buffer.put(None)

def callback(outdata, frames, time, status):
    global audio_buffer  # Ensure audio_buffer is accessible
    if status.output_underflow:
        print("Output underflow: increase buffer?", file=sys.stderr)
        raise sd.CallbackAbort
    try:
        # Attempt to fetch data from the queue
        data = audio_buffer.get_nowait()
        if data is None:  # Check for the end of data signal
            raise sd.CallbackAbort
        # Reshape data to match outdata shape
        data = data.reshape(-1, 1)  # Assuming mono output
        if len(data) < len(outdata):
            # If fetched data is smaller than outdata, pad the rest with zeros
            outdata[:len(data)] = data
            outdata[len(data):] = 0  # Pad the remaining with zeros
        else:
            # If fetched data fits outdata, use it directly
            outdata[:] = data[:len(outdata)]
    except queue.Empty:
        print("Buffer is empty: increase buffersize?", file=sys.stderr)
        outdata.fill(0)  # Fill with zeros to prevent underrun



producer_thread = threading.Thread(target=tts_stream, args=(audio_buffer, "This is a test. I am synthesizing audio."))
producer_thread.start()

with sd.OutputStream(dtype='float32', samplerate=24000, channels=1, callback=callback, blocksize=1024):
    print("stream started")
    input()

# producer_thread.join()
# print("streaming and synthesis completed")