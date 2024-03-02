from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from starlette.websockets import WebSocketDisconnect
import json
import base64

app = FastAPI()

@app.websocket("/socket")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            print(message)
            #process incoming message from Twilio
            if message.get("event") == "media":
                payload = message["media"]["payload"]
                #echo the audio back to twilio
                await websocket.send_text(json.dumps({"media": {"payload": payload}}))
            elif message.get("event") in ["connect", "start", "stop"]:
                print(f"Received {message.get('event')} event")
    except WebSocketDisconnect:
        print("WebSocket connection closed")
    finally:
        print("connection handler exiting")