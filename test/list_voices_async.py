import asyncio
from google.cloud import texttospeech_v1

async def sample_list_voices():
    # Create a client
    client = texttospeech_v1.TextToSpeechAsyncClient()

    # Initialize request argument(s)
    request = texttospeech_v1.ListVoicesRequest(
    )

    # Make the request
    response = await client.list_voices(request=request)

    for voice in response.voices:
        if any(language_code.startswith("en") for language_code in voice.language_codes):
            print(voice)


    # Handle the response
    #print(response)

asyncio.run(sample_list_voices())