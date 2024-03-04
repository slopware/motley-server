import os
import sys
import time
import torch
import queue
import threading
import base64
import torchaudio
import torchaudio.transforms as T
import numpy as np
import audioop
from fastapi import WebSocket
import asyncio
import json

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

        self.gpt_cond_latent, self.speaker_embedding = self.model.get_conditioning_latents(audio_path=["speakers/isa_denoised.wav"])
        #self.config.audio.output_sample_rate = 8000
        print(f"sample rate: {self.config.audio.output_sample_rate}") #24000
        self.resampler = T.Resample(orig_freq=24000, new_freq=8000)
        self.mulaw_encoder = T.MuLawEncoding(quantization_channels=256)

        self.audio_buffer = queue.Queue()
        self.text_buffer = queue.Queue()

        self.websocket = None
        self.sid = None

        self.synthesis_thread = threading.Thread(target=self.synthesize, daemon=True)
        self.synthesis_thread.start()

    def postprocess_chunk(self, chunk):
        """Post process the output waveform, converting to mulaw format before finally encoding in base64."""
        if isinstance(chunk, list):
            chunk = torch.cat(chunk, dim=0)
        chunk = chunk.detach().cpu()
        chunk = torch.clamp(chunk, min=-1, max=1)
        if chunk.ndim == 1:
            chunk = chunk.unsqueeze(0)
        chunk = self.resampler(chunk)

        pcm_array = np.int16(chunk*32768).tobytes()
        mulaw_array = audioop.lin2ulaw(pcm_array, 2)
        base64_encoded_chunk = base64.b64encode(mulaw_array).decode('utf-8')
        return base64_encoded_chunk
    
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
                print(f"Received chunk {i} of audio length {chunk.shape[-1]}")
                chunk = self.postprocess_chunk(chunk)
                self.audio_buffer.put(chunk)
            self.text_buffer.task_done()


    def add_text_for_synthesis(self, text):
        self.text_buffer.put(text)
            
    def stop(self):
        self.text_buffer.put(None)
        self.audio_buffer.put(None)
        self.synthesis_thread.join()

    def reset(self):
          # Clear queues without putting None, prepare for new data
        while not self.text_buffer.empty():
            self.text_buffer.get_nowait()
        while not self.audio_buffer.empty():
            self.audio_buffer.get_nowait()
        if not self.synthesis_thread.is_alive():
            self.synthesis_thread = threading.Thread(target=self.synthesize, daemon=True)
            self.synthesis_thread.start()
    
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
    
