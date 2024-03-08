from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from starlette.websockets import WebSocketDisconnect
import json
import base64
import asyncio
import audioop
import wave
import torchaudio
import torchaudio.transforms as T
import numpy as np
from xtts_engine_twilio import XttsEngine
from openai_interface import OpenAIInterface
from anthropic_interface import AnthropicInterface
from voice_engine import VoiceRecognitionEngine
from typing import Set
from asyncio import Task
import queue
app = FastAPI()
voice = VoiceRecognitionEngine()
tts_reader = XttsEngine()

running_tasks: Set[Task] = set()

with open('instructions.txt', encoding='utf-8') as f:
    system_instructions = f.read()

chatbot = OpenAIInterface(default_system=system_instructions)


async def send_audio(websocket: WebSocket, streamSid):
        try:
            chunk = await asyncio.to_thread(tts_reader.audio_buffer.get_nowait)
            if chunk is None:
                #print('nada')
                return
            else:
                #dong = chunk.slice(5)
                #print(f'sending audio chunk')
                await websocket.send_json(
                    {"event": "media",
                    "streamSid": streamSid,
                    "media": {"payload": chunk}
                    })
                await websocket.send_json(
                    {"event": "mark",
                    "streamSid": streamSid,
                    "mark": {"name": "playback completed"}
                    })
        except queue.Empty:
            return

async def process_sentences():
    async for donk in chatbot.sentence_generator():
        await asyncio.to_thread(tts_reader.add_text_for_synthesis, donk)

async def get_transcription(audio_task):
    try:
        text = voice.text_queue.get_nowait()
        if text is None:
            return
        else:
            print(text)
            tts_reader.halt()
            tts_reader.reset()
            chatbot.add_user_message(f"Detected user speech: {text}")
             # Cancel previous tasks
            for task in running_tasks:
                task.cancel()
            running_tasks.clear()
            audio_task.cancel()
            # Create a new task and add it to the running tasks set
            task = asyncio.create_task(process_sentences())
            running_tasks.add(task)
            
            # Remove the task from the running tasks set when it completes
            task.add_done_callback(lambda t: running_tasks.discard(t))

    except queue.Empty:
        return

@app.websocket("/socket")
async def websocket_endpoint(websocket: WebSocket):
    await asyncio.to_thread(tts_reader.reset)
    await asyncio.to_thread(voice.reset)
    await websocket.accept()

    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            streamSid = message.get("streamSid")

            if message.get("event") == "connected":
                print(f"Connected to protocol: {message.get('protocol')}")
            elif message.get("event") == "start": 
                print(f"Starting stream {message.get('start').get('streamSid')} {message.get('sequenceNumber')}")
            elif message.get("event") == "media":
                if message.get("sequenceNumber") == "2":
                    print("First media message received.")
                    chatbot.add_user_message("user has connected")
                    async for donk in chatbot.sentence_generator():
                        await asyncio.to_thread(tts_reader.add_text_for_synthesis, donk)

                if message.get("media").get("track") == "inbound":
                    payload = message.get("media").get("payload")
                    await asyncio.to_thread(voice.process_speech_chunk, payload)
                if message.get("media").get("track") == "outbound":
                    print("Outbound message received")
               
            elif message.get("event") == "stop":
                print("Stream stopped.")
                await asyncio.to_thread(voice.stop)
                await asyncio.to_thread(tts_reader.stop)
                for task in running_tasks:
                    task.cancel()
            elif message.get("event") == "mark":
                pass
            elif message.get("event") == "dtmf":
                chatbot.add_user_message(f"user has pressed {message.get('dtmf').get('digit')}")
                async for donk in chatbot.sentence_generator():
                    await asyncio.to_thread(tts_reader.add_text_for_synthesis, donk)
            audio_task = asyncio.create_task(send_audio(websocket, streamSid))
            running_tasks.add(audio_task)
            await get_transcription(audio_task)
    except KeyboardInterrupt:
        print("exiting...")
    except WebSocketDisconnect:
        print("WebSocket connection closed")
    except Exception as e:
        print(f"An error occurred: {e}") 
    finally:
        print("connection handler exiting")
        await asyncio.to_thread(voice.stop)
        await asyncio.to_thread(tts_reader.stop)