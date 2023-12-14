import asyncio
from google.cloud import texttospeech_v1


async def sample_synthesize_speech():
    # Create a client
    client = texttospeech_v1.TextToSpeechAsyncClient()

    # Initialize request argument(s)
    input = texttospeech_v1.SynthesisInput()
    input.text = "Hello! I am so glad you upgraded my voice!"

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

    # Make the request
    response = await client.synthesize_speech(request=request)

    # Handle the response
    #print(response)
    with open("output.mp3", "wb") as out:
        # Write the response to the output file.
        out.write(response.audio_content)
        print('Audio content written to file "output.mp3"')

asyncio.run(sample_synthesize_speech())