import numpy as np
import pyaudio
import webrtcvad
import base64
import audioop
import librosa
import io
import threading
import queue
from faster_whisper import WhisperModel

class VoiceRecognitionEngine:
    def __init__(self, default_model_size="large-v3"):
        self.model_size = default_model_size
        self.model = WhisperModel(self.model_size, device="cuda", compute_type="float16")
        self.audio_buffer = io.BytesIO()
        self.speech_frames_count = 0
        self.silence_frames_count = 0
        self.silent = True
        # Initialize PyAudio
        self.pyaudio = pyaudio.PyAudio()
        self.stream = self.pyaudio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=8000,
            output=True,
            frames_per_buffer=1024
        )
        self.chunk_queue = queue.Queue()
        self.text_queue = queue.Queue()
        self.transcription_thread = threading.Thread(target=self.transcribe, daemon=True)
        self.transcription_thread.start()

    def process_speech_chunk(self, payload):
        # Decode the base64-encoded audio data
        audio_bytes = base64.b64decode(payload)

        # Convert µ-law audio to PCM format
        pcm_data = audioop.ulaw2lin(audio_bytes, 2)

        vad = webrtcvad.Vad(1)
        is_speech = vad.is_speech(pcm_data, 8000)
        self.chunk_queue.put(pcm_data)
        if is_speech:
            self.silence_frames_count = 0
            self.silent = False
        else:
            self.silence_frames_count += 1
            if self.silence_frames_count >= 5 and self.silent == False:
                #print('...')
                self.silence_frames_count = 0
                self.silent = True
                self.chunk_queue.put('END')
        
    def write_stream(self, data):
        self.stream.write(data)

    def transcribe(self):
        while True:
            chunk = self.chunk_queue.get()
            if chunk is None:
                break
            if chunk != 'END':
                self.audio_buffer.write(chunk)
            if chunk == 'END':
                audio_data = self.audio_buffer.getvalue()
                self.audio_buffer = io.BytesIO()
                #self.stream.write(audio_data)
                #self.chunk_queue.empty()
                audio_array = np.frombuffer(audio_data, dtype=np.int16)
                audio_float32 = audio_array.astype(np.float32) / 32768.0
                segments, info = self.model.transcribe(audio_float32, vad_filter=True, beam_size=5, language="en")
                #segments = list(segments)
                for segment in segments:
                    #print(segment.text)
                    self.text_queue.put(segment.text)
                while not self.chunk_queue.empty():
                    self.chunk_queue.get_nowait()
            self.chunk_queue.task_done()


    def stop(self):
        # Stop and close the PyAudio stream
        self.chunk_queue.put(None)
        self.transcription_thread.join()
        self.stream.stop_stream()
        self.stream.close()
        self.pyaudio.terminate()

    def reset(self):
        while not self.text_queue.empty():
            self.text_queue.get_nowait()
        while not self.chunk_queue.empty():
            self.chunk_queue.get_nowait()
        if not self.transcription_thread.is_alive():
            self.transcription_thread = threading.Thread(target=self.transcribe, daemon=True)
            self.transcription_thread.start()