from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.responses import Response, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import asyncio
from google.cloud import texttospeech_v1
from pydantic import BaseModel
from io import BytesIO
import ffmpeg
import os
import subprocess
import threading
import json
import math
import openai
import fireworks.client
import nltk
from nltk.tokenize import sent_tokenize
import requests
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, PatternMatchingEventHandler
import websockets
from starlette.websockets import WebSocketDisconnect
from typing import List
import httpx
app = FastAPI()
openai.api_key = os.getenv("OPENAI_API_KEY")
fireworks.client.api_key = os.getenv("FIREWORKS_API_KEY")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SynthesizeRequest(BaseModel):
    text: str

HLS_DIRECTORY = "hls"
# Ensure HLS directory exists
if not os.path.exists(HLS_DIRECTORY):
    os.makedirs(HLS_DIRECTORY)

async def synthesize_speech(text, segment_index):
    client = texttospeech_v1.TextToSpeechAsyncClient()

    input = texttospeech_v1.SynthesisInput(text=text)
    voice = texttospeech_v1.VoiceSelectionParams(language_code="en-GB", name="en-GB-Studio-C")
    audio_config = texttospeech_v1.AudioConfig(audio_encoding="LINEAR16")

    request = texttospeech_v1.SynthesizeSpeechRequest(input=input, voice=voice, audio_config=audio_config)
    response = await client.synthesize_speech(request=request)

    # Save the synthesized audio as a WAV file
    wav_filename = os.path.join(HLS_DIRECTORY, f"output_{segment_index}.wav")
    with open(wav_filename, "wb") as audio_file:
        audio_file.write(response.audio_content)

    return wav_filename

@app.post("/chat")
async def chat_completions(request_body: dict):
    body = request_body.copy()
    if body.get('model') == "mistral":
        return StreamingResponse(fireworks_event_generator(body), media_type="text/event-stream")
    else:
        return StreamingResponse(openai_event_generator(request_body), media_type="text/event-stream")

main_event_loop = asyncio.get_event_loop()

def fireworks_event_generator(body):
    body.pop('model', None)
    body.pop('temperature', None)
    openai.api_base = "https://api.fireworks.ai/inference/v1"
    openai.api_key = os.getenv("FIREWORKS_API_KEY")
    printed_sentences = set()
    accumulator = ""
    index = 0
    response = openai.ChatCompletion.create(
        model="accounts/fireworks/models/mixtral-8x7b-instruct",
        temperature=0.5,
        max_tokens=800,
        **body)
    for message in response:
        #print(message)
        delta = message.choices[0].delta
        content = getattr(delta, 'content', None)
        finish_reason = getattr(message.choices[0], 'finish_reason', None)
        if content is not None:
            #print(content)
            accumulator += content
            sentences = sent_tokenize(accumulator)
            if len(sentences) > 1 and sentences[-2] not in printed_sentences:
                print('adding sentence: ', sentences[-2])
                printed_sentences.add(sentences[-2])
                send_synthesize_request_async(sentences[-2], index, main_event_loop)
                index += 1
        #if finish_reason == "stop":
        yield f"data: {json.dumps(message)}\n\n"
    sentences = sent_tokenize(accumulator)
    if sentences[-1] not in printed_sentences:
        print('adding last sentence: ', sentences[-1])
        printed_sentences.add(sentences[-1])
        send_synthesize_request_async(sentences[-1], index, main_event_loop)
    yield "data: [DONE]\n\n"

SERVER_URL = "http://localhost:8000/synthesize/"

active_websockets: List[WebSocket] = []

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_websockets.append(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            await websocket.send_text(f"Message received was: {data}")
    except WebSocketDisconnect:
        active_websockets.remove(websocket)  # Remove the WebSocket from the list upon disconnection
        print(f"WebSocket disconnected and removed. Current active sockets: {len(active_websockets)}")

def send_synthesize_request_async(text, segment_index, loop):
    def send_request():
        response = requests.post(SERVER_URL, json={"text": text}, params={"segment_index": segment_index})
        time.sleep(1)
        if response.status_code == 200:
            filename = json.loads(response.text)["filename"][4:]
            print(filename)
            for websocket in active_websockets:
                print('sending ', filename)
                asyncio.run_coroutine_threadsafe(websocket.send_text(f"{filename}"), loop)
    thread = threading.Thread(target=send_request)
    thread.start()

def openai_event_generator(request_body):
    openai.api_base = "https://api.openai.com/v1"
    openai.api_key = os.getenv("OPENAI_API_KEY")
    response = openai.ChatCompletion.create(**request_body)
    for message in response:
        #print(f"data: {json.dumps(message)}\n\n")
        yield f"data: {json.dumps(message)}\n\n"
    yield "data: [DONE]\n\n"

@app.post("/synthesize/")
async def create_synthesize(request: SynthesizeRequest, segment_index: int):
    if not request.text:
        raise HTTPException(status_code=400, detail="No text provided")
    print('sending this to google api: ', request)
    wav_filename = await synthesize_speech(request.text, segment_index)

    # Trigger the conversion process (can also be run as a background task)
    #convert_to_ts_and_update_playlist()

    return {"message": "Audio synthesized", "filename": wav_filename}

@app.post("/reset/")
async def reset_hls_folder():
    try:
        # Loop through all files in the HLS directory and delete them
        for filename in os.listdir(HLS_DIRECTORY):
            file_path = os.path.join(HLS_DIRECTORY, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
        return {"message": "HLS folder reset successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

app.mount("/hls", StaticFiles(directory="hls"), name="hls")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
# uvicorn server:app --host 0.0.0.0 --port 8000 --reload 