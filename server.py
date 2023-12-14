from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import asyncio
from google.cloud import texttospeech_v1
from pydantic import BaseModel

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

class SynthesizeRequest(BaseModel):
    text: str


async def synthesize_speech(text):
    client = texttospeech_v1.TextToSpeechAsyncClient()

    input = texttospeech_v1.SynthesisInput(text=text)

    voice = texttospeech_v1.VoiceSelectionParams()
    voice.language_code = "en-au"
    voice.name = "en-AU-Neural2-A"

    audio_config = texttospeech_v1.AudioConfig()
    audio_config.audio_encoding = "MP3"

    request = texttospeech_v1.SynthesizeSpeechRequest(
        input=input,
        voice=voice,
        audio_config=audio_config,
    )

    response = await client.synthesize_speech(request=request)
    return response.audio_content

@app.post("/synthesize/")
async def create_synthesize(request: SynthesizeRequest):
    if not request.text:
        raise HTTPException(status_code=400, detail="No text provided")

    audio_content = await synthesize_speech(request.text)
    with open("output.mp3", "wb") as out:
        out.write(audio_content)

    return FileResponse("output.mp3", media_type='audio/mp3')

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, ssl_certfile="cert.pem", ssl_keyfile="nopass_key.pem")
