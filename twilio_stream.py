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

app = FastAPI()

tts_reader = XttsEngine()

async def send_audio(websocket: WebSocket, streamSid):
        chunk = tts_reader.audio_buffer.get()
        tts_reader.audio_buffer.task_done()
        if chunk is None:
            print('nada')
            return
        else:
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

@app.websocket("/socket")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    try:
        while True:
            
            data = await websocket.receive_text()
            message = json.loads(data)
            streamSid = message.get("streamSid")

            if message.get("event") == "connected":
                #The first message sent once a WebSocket connection is established is the Connected event. This message describes the protocol to expect in the following messages.
                print(f"Connected to protocol: {message.get('protocol')}")
            elif message.get("event") == "start":
                #This message contains important metadata about the stream and is sent immediately after the Connected message. It is only sent once at the start of the Stream.
                print(f"Starting stream {message.get('start').get('streamSid')} {message.get('sequenceNumber')}")
                tts_reader.add_text_for_synthesis("you have connected. this is a successful test.")
            elif message.get("event") == "media":
                if message.get("sequenceNumber") == "2":
                    print("First media message received.")
                if message.get("media").get("track") == "inbound":
                    None
                if message.get("media").get("track") == "outbound":
                    print("Outbound message received")
            elif message.get("event") == "stop":
                #A stop message will be sent when the Stream is either <Stop>ped or the Call has ended.
                print("Stream stopped.")
                #tts_reader.stop()
            elif message.get("event") == "mark":
                print("mark message received.")
            elif message.get("event") == "dtmf":
                tts_reader.add_text_for_synthesis("you have connected. this is a successful test.")
                #A DTMF message will be sent when someone presses a touch-tone number key (such as the "1" key in the example message below) in the inbound stream, typically in response to a prompt in the outbound stream.
                print(f"DTMF message received. Digit pressed: {message.get('dtmf').get('digit')}")
                #await send_audio(websocket, streamSid)
            await send_audio()
    except WebSocketDisconnect:
        print("WebSocket connection closed")
    except:
        print("unknown error")
    finally:
        print("connection handler exiting")

#twilio automatically buffers and synchronously plays audio sent over the socket
#format:
# {
#   "event": "media",
#   "streamSid": "MZ18ad3ab5a668481ce02b83e7395059f0",
#   "media": {
#     "payload": "a3242sadfasfa423242... (a base64 encoded string of 8000/mulaw)"
#   }
# }
# await websocket.send_text(
#     json.dumps(
#         {"event": "media",
#          "streamSid":streamSid,
#          "media": {"payload": chunk}
#          }))