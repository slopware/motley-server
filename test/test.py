import nltk
from nltk.tokenize import sent_tokenize
import fireworks.client
from openai import AsyncOpenAI
import os
import json
from google.cloud import texttospeech_v1
import asyncio
import pyaudio
import wave
import time

client = AsyncOpenAI(
    api_key=os.getenv("FIREWORKS_API_KEY"),
    api_base="https://api.fireworks.ai/inference/v1"
)

processing = False

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
    print('playing audio: ', file_path)
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
    request = texttospeech_v1.SynthesizeSpeechRequest(input=input, voice=voice, audio_config=audio_config)
    response = await client.synthesize_speech(request=request)
    wav_filename = os.path.join('test_output', f"out_{index}.wav")
    print('saving ', text, ' as ', wav_filename)
    with open(wav_filename, "wb") as audio_file:
        audio_file.write(response.audio_content)
    #play_audio(wav_filename)

printed_sentences = []
response = ""

def tokenize_sentences(text):
    sentences = sent_tokenize(text)
    if len(sentences) > 1 and sentences[-2] not in printed_sentences:
        printed_sentences.append(sentences[-2])
        print('adding: ', printed_sentences.index(sentences[-2]), ': ', sentences[-2])
        #task = asyncio.create_task(synthesize_speech(sentences[-2], printed_sentences.index(sentences[-2])))

def tokenize_last_sentence(text):
    sentences = sent_tokenize(text)
    if len(sentences) > 1 and sentences[-1] not in printed_sentences:
        print('adding last: ', printed_sentences.index(sentences[-1]), ': ', sentences[-1])
        printed_sentences.append(sentences[-1])
        #task = asyncio.create_task(synthesize_speech(sentences[-1], printed_sentences.index(sentences[-1])))

def generate(input):
    global processing

    for filename in os.listdir('test_output'):
        file_path = os.path.join('test_output', filename)
        os.remove(file_path)

    global printed_sentences
    global response
    processing = True
    chatCompletionGenerator = openai.ChatCompletion.create(
    model="accounts/fireworks/models/mistral-7b-instruct-4k",
    messages=[
        {
            "role": "system",
            "content": "You are an AI Jerry Seinfeld."
        },
        {
        "role": "user",
        "content": input,
        }
    ],
    stream=True,
    n=1,
    max_tokens=800,
    temperature=0.3,
    top_p=0.9,
    stop=[]
    )

    for chunk in chatCompletionGenerator:
        role = getattr(chunk.choices[0].delta, 'role', None)
        content = getattr(chunk.choices[0].delta, 'content', None)
        finish_reason = getattr(chunk.choices[0], 'finish_reason', None)
        if role is not None:
            print(role)
        if content is not None:
            response += content
            yield tokenize_sentences(response)
    tokenize_last_sentence(response)
    processing = False

async def periodic_task():
    index = 0
    while True:
        if len(printed_sentences) > 0:
            #print(f'before reading: {printed_sentences}')
            await synthesize_speech(printed_sentences[0], index)
            printed_sentences.pop(0)
            index += 1
            #print(f'after reading: {printed_sentences}')
        if len(printed_sentences) == 0:
            await asyncio.sleep(1)
            if len(printed_sentences) == 0:
                break

async def main():
    print("Main start")
    task = asyncio.create_task(generate("Tell me a story."))
    await periodic_task()
    print("Main end.")


asyncio.run(main())
#try using decorators