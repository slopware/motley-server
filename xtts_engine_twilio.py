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
        self.audio_buffer = queue.Queue()
        self.text_buffer = queue.Queue()
        self.pyaudio_instance = pyaudio.PyAudio()
        self.stream = self.pyaudio_instance.open(format=pyaudio.paFloat32,
                             channels=1,
                             rate=self.config.audio.output_sample_rate,
                             output=True)
        # Thread management
        self.synthesis_thread = threading.Thread(target=self.synthesize, daemon=True)
        self.playback_thread = threading.Thread(target=self.play_audio, daemon=True)
        
        # Start threads
        self.synthesis_thread.start()
        self.playback_thread.start()


    def postprocess_wave(self, chunk):
        """Post process the output waveform"""
        if isinstance(chunk, list):
            chunk = torch.cat(chunk, dim=0)
        chunk = chunk.clone().detach().cpu().numpy()
        chunk = chunk[None, : int(chunk.shape[0])]
        chunk = np.clip(chunk, -1, 1)
        chunk = chunk.astype(np.float32)
        return chunk
    
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
                # if i == 0:
                #     print(f"Time to first chunk: {time.time() - t0}")
                # print(f"Received chunk {i} of audio length {chunk.shape[-1]}")
                chunk = self.postprocess_wave(chunk)
                self.audio_buffer.put(chunk)
            self.text_buffer.task_done()

    def play_audio(self):
        while True:
            chunk = self.audio_buffer.get()
            if chunk is None:
                break
            self.stream.write(chunk.tobytes(), exception_on_underflow=False)
            self.audio_buffer.task_done()
    
    def add_text_for_synthesis(self, text):
        self.text_buffer.put(text)

    def stop(self):
        # Signal to stop processing and playback
        self.text_buffer.put(None)
        self.audio_buffer.put(None)
        # Wait for threads to finish
        self.synthesis_thread.join()
        self.playback_thread.join()
        # Clean up the audio stream
        self.stream.stop_stream()
        self.stream.close()
