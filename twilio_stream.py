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
from chat_engine import OpenAIInterface
from voice_engine import VoiceRecognitionEngine

import queue
app = FastAPI()
voice = VoiceRecognitionEngine()
tts_reader = XttsEngine()
chatbot = OpenAIInterface(system_base="""
You are a WIP VOIP conversation AI designed to talk to someone on a cell phone.
Currently that someone is the programmer as this is the testing phase. User speech has just been implemented but you may receive messages like 'user has pressed the pound key' or 'user has pressed 1' if you want you can make jokes about it. good luck. be terse and sarcastic.
""")

async def send_audio(websocket: WebSocket, streamSid):
        try:

            chunk = await asyncio.to_thread(tts_reader.audio_buffer.get_nowait)
            if chunk is None:
                #print('nada')
                return
            else:
                #print(f'chunk: {chunk}')
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

async def get_transcription():
    try:
        text = await asyncio.to_thread(voice.text_queue.get_nowait)
        if text is None:
            return
        else:
            print(text)
            chatbot.add_user_message(text)
            async for donk in chatbot.sentence_generator():
                await asyncio.to_thread(tts_reader.add_text_for_synthesis, donk)
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
                await send_audio(websocket, streamSid)
                await get_transcription()
            elif message.get("event") == "stop":
                print("Stream stopped.")
                voice.stop()
            elif message.get("event") == "mark":
                pass
            elif message.get("event") == "dtmf":
                chatbot.add_user_message(f"user has pressed {message.get('dtmf').get('digit')}")
                async for donk in chatbot.sentence_generator():
                    await asyncio.to_thread(tts_reader.add_text_for_synthesis, donk)

    except KeyboardInterrupt:
        print("exiting...")
    except WebSocketDisconnect:
        print("WebSocket connection closed")
    except Exception as e:
        print(f"An error occurred: {e}") 
    finally:
        print("connection handler exiting")
        await asyncio.to_thread(tts_reader.stop)