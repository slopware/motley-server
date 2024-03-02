from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from starlette.websockets import WebSocketDisconnect
import json
import base64
import asyncio

from xtts_engine_twilio import XttsEngine

app = FastAPI()

tts_reader = XttsEngine()

@app.websocket("/socket")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            streamSid = message.get("streamSid")
            # chunk = tts_reader.get_chunk()
            # if chunk is None:  # End signal received
            #     break
            # await websocket.send_text(json.dumps({"event": "media", "streamSid":streamSid, "media": {"payload": chunk}}))
            if message.get("event") == "connected":
                print("connected")
                tts_reader.add_text_for_synthesis("you have connected")
            elif message.get("event") == "media":
                try:
                    chunk = tts_reader.get_chunk()
                    if chunk is None:  # End signal received
                        break
                    await websocket.send_text(json.dumps({"event": "media", "streamSid":streamSid, "media": {"payload": chunk}}))
                except:
                    print("error")
            elif message.get("event") in ["start", "stop"]:
                print(f"Received {message.get('event')} event")
    except WebSocketDisconnect:
        print("WebSocket connection closed")
    finally:
        print("connection handler exiting")