import asyncio
from google.cloud import texttospeech_v1


async def sample_synthesize_speech():
    # Create a client
    client = texttospeech_v1.TextToSpeechAsyncClient()

    # Initialize request argument(s)
    input = texttospeech_v1.SynthesisInput()
    input.text = "Google Cloud Text-to-Speech enables developers to synthesize natural-sounding speech with 100+ voices, available in multiple languages and variants. It applies DeepMind’s groundbreaking research in WaveNet and Google’s powerful neural networks to deliver the highest fidelity possible."

    voice = texttospeech_v1.VoiceSelectionParams()
    voice.language_code = "en-GB"
    voice.name = "en-GB-Studio-C"

    audio_config = texttospeech_v1.AudioConfig()
    audio_config.audio_encoding = "LINEAR16"

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