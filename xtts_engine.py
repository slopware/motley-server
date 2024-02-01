import torch
from TTS.api import TTS

if torch.cuda.is_available():
    print("gpu detected")

device = "cuda" if torch.cuda.is_available() else "cpu"
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

def generate_speech_file(text, index):
    tts.tts_to_file(text=text, file_path=f"output_{index}.wav", speaker_wav="isa_sample.wav", language="en")