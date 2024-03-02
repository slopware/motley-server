import os
import sys
import time
import torch
import torchaudio
import queue
import pyaudio
import threading
import numpy as np
import subprocess
import base64


from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from TTS.config import load_config

class XttsEngine:
    def __init__(self):
        self.config = XttsConfig()
        #self.config = load_config("xtts_models/v2.0.2/config.json")
        self.config.load_json("xtts_models/v2.0.2/config.json")
        self.model = Xtts.init_from_config(self.config)
        self.model.load_checkpoint(self.config, checkpoint_dir="xtts_models/v2.0.2/", use_deepspeed=True, eval=True)
        self.model.cuda()
        self.gpt_cond_latent, self.speaker_embedding = self.model.get_conditioning_latents(audio_path=["isa_denoised.wav"])
        print(f"sample rate: {self.config.audio.output_sample_rate}")
        self.resampler = torchaudio.transforms.Resample(orig_freq=24000, new_freq=8000)
        self.mu_law_encoder = torchaudio.transforms.MuLawEncoding(quantization_channels=246)
        self.audio_buffer = queue.Queue()
        self.text_buffer = queue.Queue()

        self.synthesis_thread = threading.Thread(target=self.synthesize, daemon=True)
        #self.send_thread = threading.Thread(target=self.send_audio, daemon=True)
        
        # Start threads
        self.synthesis_thread.start()
        #self.send_thread.start()

    def postprocess_and_encode(self, chunk):
        chunk_cpu = chunk.to('cpu')
        resampled_audio = self.resampler(chunk_cpu)
        mulaw_encoded_audio = self.mu_law_encoder(resampled_audio)
        audio_bytes = mulaw_encoded_audio.numpy().astype(np.uint8).tobytes()
        base64_encoded_audio = base64.b64encode(audio_bytes).decode('utf-8')
        return base64_encoded_audio

    def synthesize(self):
        while True:
            text = self.text_buffer.get()
            if text is None:
                break
            t0 = time.time()
            chunks = self.model.inference_stream(
                text,
                "en",
                self.gpt_cond_latent,
                self.speaker_embedding
            )
            for i, chunk in enumerate(chunks):
                chunk = self.postprocess_and_encode(chunk)
                self.audio_buffer.put(chunk)
                print(chunk)
            self.text_buffer.task_done()

    def send_chunks(self):
        while True:
            chunk = self.audio_buffer.get()
            if chunk is None:
                break
            self.audio_buffer.task_done()
            return chunk
    
    def get_chunk(self):
        chunk = self.audio_buffer.get()
        self.audio_buffer.task_done()
        return chunk

    def add_text_for_synthesis(self, text):
        self.text_buffer.put(text)

    def stop(self):
        # Signal to stop processing and playback
        self.text_buffer.put(None)
        self.audio_buffer.put(None)
        # Wait for threads to finish
        self.synthesis_thread.join()
        #self.playback_thread.join()
        # Clean up the audio stream
        self.stream.stop_stream()
        self.stream.close()

if __name__ == "__main__":
    import asyncio
    tts_reader = XttsEngine()
    try:
        while True:
            user_input = input("User: ")
            tts_reader.add_text_for_synthesis(user_input)
    except KeyboardInterrupt:
        tts_reader.stop()
        print("exiting...")
    
