import asyncio
from openai import AsyncOpenAI
from nltk.tokenize import sent_tokenize
import json
import requests
client = AsyncOpenAI()

generating = False
messages = [{"role": "system", "content": "You are an unhelpful and sarcastic assistant."}]
printed_sentences = []
response = ""
i = 0

async def tokenize_sentences(text):

    url = 'http://localhost:8020/tts_to_audio/'
    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json'
    }
    data = {
        #"text": "Let the whole world know you as you really are.",
        "speaker_wav": "isa_sample.wav",
        "language": "en"
    }
    global printed_sentences
    global response
    global i
    response += text
    sentences = sent_tokenize(response)
    
    if len(sentences) > 1 and sentences[-2] not in printed_sentences:
        printed_sentences.append(sentences[-2])
        print(f"adding sentence {printed_sentences.index(sentences[-2])}: {sentences[-2]}")
        data["text"] = sentences[-2]
        requests.post(url, headers=headers, json=data)
    if text == "" and len(sentences) > 1:
        printed_sentences.append(sentences[-1])
        print(f"adding (last) sentence {printed_sentences.index(sentences[-1])}: {sentences[-1]}")
        print(f"finished: {printed_sentences}")
        data["text"] = sentences[-1]
        requests.post(url, headers=headers, json=data)
    if text == "" and len(sentences) == 1:
        printed_sentences.append(sentences[-1])
        print(f"adding (only) sentence {printed_sentences.index(sentences[-1])}: {sentences[-1]}")
        print(f"finished: {printed_sentences}")
        data["text"] = sentences[-1]
        requests.post(url, headers=headers, json=data)

async def generate(input):
    global messages
    global generating
    messages.append({"role": "user", "content": input})
    print(f"messages: {messages}")
    generating = True
    resp = ""
    stream = await client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=messages,
        stream=True,
    )
    async for chunk in stream:
        resp += chunk.choices[0].delta.content or ""
        await tokenize_sentences(chunk.choices[0].delta.content or "")
    messages.append({"role": "assistant", "content": resp})
    generating = False
    print(f"stream finished\nmessages: {messages}")
    
async def main():
    generation = asyncio.create_task(generate("hello"))
    await generation

asyncio.run(main())
# python -m xtts_api_server --deepspeed --speaker-folder speakers --model-folder xtts_models --streaming-mode --stream-play-sync