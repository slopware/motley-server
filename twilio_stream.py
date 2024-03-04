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
from speech_engine import process_speech_chunk
import queue
app = FastAPI()

tts_reader = XttsEngine()
chatbot = OpenAIInterface(system_base="""
You are a WIP VOIP conversation AI designed to talk to someone on a cell phone.
Currently that someone is the programmer as this is the testing phase. User speech has not yet been implemented but you may receive messages like 'user has pressed the pound key' or 'user has pressed 1' if you want you can make jokes about it. good luck. be terse and sarcastic.
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

@app.websocket("/socket")
async def websocket_endpoint(websocket: WebSocket):
    await asyncio.to_thread(tts_reader.reset)
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
                    await process_speech_chunk(payload)
                if message.get("media").get("track") == "outbound":
                    print("Outbound message received")
                await send_audio(websocket, streamSid)
            elif message.get("event") == "stop":
                print("Stream stopped.")
                # tts_reader.stop()
            elif message.get("event") == "mark":
                print("mark message received.")
            elif message.get("event") == "dtmf":
                # tts_reader.add_text_for_synthesis("you have connected. this is a successful test.")
                #A DTMF message will be sent when someone presses a touch-tone number key (such as the "1" key in the example message below) in the inbound stream, typically in response to a prompt in the outbound stream.
                print(f"DTMF message received. Digit pressed: {message.get('dtmf').get('digit')}")
                # await asyncio.to_thread(tts_reader.add_text_for_synthesis, f"you have pressed {message.get('dtmf').get('digit')}")
                chatbot.add_user_message(f"user has pressed {message.get('dtmf').get('digit')}")
                #print("AI: ", end="", flush=True)
                async for donk in chatbot.sentence_generator():
                    await asyncio.to_thread(tts_reader.add_text_for_synthesis, donk)
                    #print(donk, end=" ", flush=True)
                #print("")
                #await send_audio(websocket, streamSid)
    except KeyboardInterrupt:
        print("exiting...")
    except WebSocketDisconnect:
        print("WebSocket connection closed")
    except Exception as e:
        print(f"An error occurred: {e}") 
    finally:
        print("connection handler exiting")
        await asyncio.to_thread(tts_reader.stop)