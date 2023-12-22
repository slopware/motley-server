from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import asyncio
from google.cloud import texttospeech_v1
from pydantic import BaseModel
from io import BytesIO
import ffmpeg
import os
import subprocess
import json
import math
import time
app = FastAPI()

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

    # Save the audio to a temporary file in the HLS directory
    temp_filename = os.path.join(HLS_DIRECTORY, f"temp_audio_{segment_index}.wav")
    with open(temp_filename, "wb") as audio_file:
        audio_file.write(response.audio_content)

    # Convert the audio to TS format using FFmpeg command line
    segment_filename = os.path.join(HLS_DIRECTORY, f"segment_{segment_index}.ts")
    command = [
        'ffmpeg', '-y', '-i', temp_filename, 
        '-c:a', 'aac', '-b:a', '128k', 
        '-f', 'mpegts', segment_filename
    ]
    subprocess.run(command, check=True)

    # Determine the duration of the segment
    duration_command = [
        'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1', temp_filename
    ]
    result = subprocess.run(duration_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    segment_duration = float(result.stdout)

    # Remove the temporary file
    os.remove(temp_filename)

    return {"segment_filename": segment_filename, "segment_duration": segment_duration}

playlist_filename = os.path.join(HLS_DIRECTORY, "stream.m3u8")

max_segment_duration = 0

def update_playlist(segment_filename, segment_duration):
    global max_segment_duration
    segment_basename = os.path.basename(segment_filename)
    max_segment_duration = max(max_segment_duration, segment_duration)
    if not os.path.exists(playlist_filename):
        with open(playlist_filename, 'w') as f:
            f.write("#EXTM3U\n")
            f.write("#EXT-X-VERSION:3\n")
            f.write("#EXT-X-TARGETDURATION:{}\n".format(math.ceil(max_segment_duration)))
            f.write("#EXT-X-MEDIA-SEQUENCE:0\n")

    with open(playlist_filename, 'a') as f:
        f.write("#EXTINF:{},\n".format(segment_duration))
        f.write(segment_basename + "\n")  # Write only the file name

@app.post("/synthesize/")
async def create_synthesize(request: SynthesizeRequest, segment_index: int):
    if not request.text:
        raise HTTPException(status_code=400, detail="No text provided")

    synthesis_result = await synthesize_speech(request.text, segment_index)
    segment_filename = synthesis_result["segment_filename"]
    segment_duration = synthesis_result["segment_duration"]

    update_playlist(segment_filename, segment_duration)
    return {"message": "Audio synthesized and added to playlist"}

# Assuming your segment files and playlist are in a directory named 'hls'
app.mount("/hls", StaticFiles(directory="hls"), name="hls")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
