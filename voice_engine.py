import asyncio
import numpy as np
import pyaudio
import webrtcvad
import base64
import audioop
import librosa
import io

from faster_whisper import WhisperModel

class VoiceRecognitionEngine:
    def __init__(self, default_model_size="medium"):
        self.model_size = default_model_size
        self.model = WhisperModel(self.model_size, device="cuda", compute_type="float16")
        self.audio_buffer = io.BytesIO()
        self.speech_frames_count = 0
        self.silence_frames_count = 0
        # Initialize PyAudio
        self.pyaudio = pyaudio.PyAudio()
        self.stream = self.pyaudio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=8000,
            output=True,
            frames_per_buffer=1024
        )
    async def process_speech_chunk(self, payload):
        # Decode the base64-encoded audio data
        audio_bytes = base64.b64decode(payload)

        # Convert µ-law audio to PCM format
        pcm_data = audioop.ulaw2lin(audio_bytes, 2)

        vad = webrtcvad.Vad(2)
        is_speech = vad.is_speech(pcm_data, 8000)
        
        if is_speech:
            print('writing data')
            self.audio_buffer.write(pcm_data)
            self.speech_frames_count += 1
        else:
            self.silence_frames_count += 1
            if self.silence_frames_count >= 10 and self.speech_frames_count > 10:
                print('sending to stream')
                #self.write_stream(self.audio_buffer.getvalue())
                await asyncio.to_thread(self.transcribe, self.audio_buffer.getvalue())
                self.audio_buffer = io.BytesIO()
                self.speech_frames_count = 0
                self.silence_frames_count = 0

                print('.', end="", flush=True)


    def write_stream(self, data):
        self.stream.write(data)



    def transcribe(self, audio):
        audio_array = np.frombuffer(audio, dtype=np.int16)
        audio_float32 = audio_array.astype(np.float32) / 32768.0
        segments, info = self.model.transcribe(audio_float32, vad_filter=True)
        #segments = list(segments)
        #print(f"segments:{segments}\n info:{info}")
        #print(detected_language)
        for segment in segments:
            print(segment)
        
    async def stop(self):
        # Stop and close the PyAudio stream
        self.stream.stop_stream()
        self.stream.close()
        self.pyaudio.terminate()