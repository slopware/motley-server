from google.cloud import texttospeech_v1

def sample_list_voices():
    # Create a client
    client = texttospeech_v1.TextToSpeechClient()

    # Initialize request argument(s)
    request = texttospeech_v1.ListVoicesRequest(
    )

    # Make the request
    response = client.list_voices(request=request)
    print("test")
    # Handle the response
    #print(response)
    for voice in response.voices:
        if any(language_code.startswith("en") for language_code in voice.language_codes):
            print(voice)

sample_list_voices()