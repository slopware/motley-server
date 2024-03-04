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
    def __init__(self, default_model_size="large-v3"):
        self.model_size = default_model_size
        self.model = WhisperModel(self.model_size, device="cuda", compute_type="float16")
        self.audio_buffer = io.BytesIO()
        self.speech_frames_count = 0
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

        # Convert the PCM data to a numpy array
        #audio_array = np.frombuffer(pcm_data, dtype=np.int16)
        #self.stream.write(pcm_data)
        # Perform voice activity detection
        self.audio_buffer.write(pcm_data)
        vad = webrtcvad.Vad(3)  # Adjust the aggressiveness level as needed
        is_speech = vad.is_speech(pcm_data, 8000)
        
        if is_speech:
            self.speech_frames_count += 1

        if self.speech_frames_count > 20:
            await self.write_stream(self.audio_buffer.getvalue())

    async def write_stream(self, data):
        self.stream.write(data)
        self.audio_buffer = io.BytesIO()
        self.speech_frames_count = 0


    async def transcribe(self, audio):
        segments, info = self.model.transcribe(audio)
        segments = list(segments)
        print(info)
        #print(detected_language)
        english_segments = []
        for segment in segments:
            if segment.no_speech_prob < 0.9:
                pass
                # if detected_language == "en":
                #     print(segment.text.lower().strip())
                

        self.speech_frames_count = 0
        self.frame_accumulator = []
    
    async def stop(self):
        # Stop and close the PyAudio stream
        self.stream.stop_stream()
        self.stream.close()
        self.pyaudio.terminate()