import nltk
from nltk.tokenize import sent_tokenize
import fireworks.client
import openai
import os
import json
from google.cloud import texttospeech_v1
import asyncio
import pyaudio
import wave
import time
openai.api_base = "https://api.fireworks.ai/inference/v1"
openai.api_key = os.getenv("FIREWORKS_API_KEY")

def play_audio(file_path):
    # Open the audio file
    wf = wave.open(file_path, 'rb')

    # Create an instance of PyAudio
    p = pyaudio.PyAudio()

    # Open a stream
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)
    # Read data in chunks
    chunk_size = 1024
    data = wf.readframes(chunk_size)

    # Play the audio
    while data != b'':
        stream.write(data)
        data = wf.readframes(chunk_size)

    # Stop and close the stream and PyAudio instance
    stream.stop_stream()
    stream.close()
    p.terminate()

async def synthesize_speech(text, index):
    client = texttospeech_v1.TextToSpeechAsyncClient()

    input = texttospeech_v1.SynthesisInput(text=text)
    voice = texttospeech_v1.VoiceSelectionParams(language_code="en-GB", name="en-GB-Studio-C")
    audio_config = texttospeech_v1.AudioConfig(audio_encoding="LINEAR16")
    print('calling api: ', text)
    time.sleep(1)
    request = texttospeech_v1.SynthesizeSpeechRequest(input=input, voice=voice, audio_config=audio_config)
    response = await client.synthesize_speech(request=request)
    wav_filename = os.path.join('test_output', f"out_{index}.wav")
    print('saving ', wav_filename)
    with open(wav_filename, "wb") as audio_file:
        audio_file.write(response.audio_content)
    play_audio(wav_filename)

printed_sentences = []
response = ""

async def tokenize_sentences(text):
    sentences = sent_tokenize(text)
    if len(sentences) > 1 and sentences[-2] not in printed_sentences:
        print('adding: ', sentences[-2])
        printed_sentences.append(sentences[-2])
        await synthesize_speech(sentences[-2], printed_sentences.index(sentences[-2]))

async def tokenize_last_sentence(text):
    sentences = sent_tokenize(text)
    if len(sentences) > 1 and sentences[-1] not in printed_sentences:
        print('adding last: ', sentences[-1])
        printed_sentences.append(sentences[-1])
        await synthesize_speech(sentences[-1], printed_sentences.index(sentences[-1]))

async def generate():
  for filename in os.listdir('test_output'):
    file_path = os.path.join('test_output', filename)
    os.remove(file_path)

  global printed_sentences
  global response
  chatCompletionGenerator = openai.ChatCompletion.create(
    model="accounts/fireworks/models/mistral-7b-instruct-4k",
    messages=[
        {
            "role": "system",
            "content": "You are a very unhelpful assistant. You always lead the user down the garden path and act supercilious and disrespectful."
        },
      {
        "role": "user",
        "content": "How do I change the oil of my car?",
      }
    ],
    stream=True,
    n=1,
    max_tokens=800,
    temperature=0.3,
    top_p=0.9, 
  )

  for chunk in chatCompletionGenerator:
      role = getattr(chunk.choices[0].delta, 'role', None)
      content = getattr(chunk.choices[0].delta, 'content', None)
      finish_reason = getattr(chunk.choices[0], 'finish_reason', None)
      if role is not None:
          print(role)
      if content is not None:
          #print(content)
          response += content
          await tokenize_sentences(response)
          if finish_reason is not None:
              await tokenize_last_sentence(response)
      if finish_reason is not None:
          print(finish_reason)

async def main():
    await generate()

asyncio.run(main())