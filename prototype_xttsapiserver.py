import asyncio
from openai import AsyncOpenAI
from nltk.tokenize import sent_tokenize
import json
import requests
import httpx

client = AsyncOpenAI()

generating = False
messages = [{"role": "system", "content": "You are a terse and sarcastic assistant."}]
printed_sentences = []
response = ""
i = 0

async def xtts_api_server_request(text):
    """run with xtts-api-server"""
    url = 'http://localhost:8020/tts_to_audio/'
    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json'
    }
    data = {
        "text": text,
        "speaker_wav": "isa_sample.wav",
        "language": "en"
    }
    print(f"requesting: {text}")
    async with httpx.AsyncClient(timeout=1000) as client:
        req = asyncio.create_task(client.post(url, headers=headers, json=data))
        await req
    #asyncio.create_task(requests.post(url, headers=headers, json=data))

async def tokenize_sentences(text):
    global printed_sentences
    global response
    global i
    response += text
    sentences = sent_tokenize(response)
    
    if len(sentences) > 1 and sentences[-2] not in printed_sentences:
        printed_sentences.append(sentences[-2])
        print(f"adding sentence {printed_sentences.index(sentences[-2])}: {sentences[-2]}")
        asyncio.create_task(xtts_api_server_request(sentences[-2]))
    if text == "" and len(sentences) > 1:
        printed_sentences.append(sentences[-1])
        print(f"adding (last) sentence {printed_sentences.index(sentences[-1])}: {sentences[-1]}")
        print(f"finished: {printed_sentences}")
        printed_sentences.clear()
        response = ""
        task = asyncio.create_task(xtts_api_server_request(sentences[-1]))
        await task
    if text == "" and len(sentences) == 1:
        printed_sentences.append(sentences[-1])
        print(f"adding (only) sentence {printed_sentences.index(sentences[-1])}: {sentences[-1]}")
        print(f"finished: {printed_sentences}")
        response = ""
        printed_sentences.clear()
        task = asyncio.create_task(xtts_api_server_request(sentences[-1]))
        await task
        
async def generate(input):
    global messages
    global generating
    messages.append({"role": "user", "content": input})
    print(f"messages: {messages}")
    generating = True
    resp = ""
    stream = await client.chat.completions.create(
        model="gpt-4-turbo-preview",
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
    while True:
        generation = asyncio.create_task(generate(input()))
        await generation

asyncio.run(main())
# python -m xtts_api_server --deepspeed --speaker-folder speakers --model-folder xtts_models --streaming-mode --stream-play-sync