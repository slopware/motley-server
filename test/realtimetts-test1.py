from RealtimeTTS import TextToAudioStream, OpenAIEngine, CoquiEngine
import os
from openai import OpenAI
client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)

def stream_generator():
    stream = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "Tell a story in 10 sentences."}],
            stream=True,
    )
    for chunk in stream:
        yield chunk.choices[0].delta.content or ""

engine = OpenAIEngine(model="tts-1", voice="onyx")
stream = TextToAudioStream(engine)
stream.feed(stream_generator())
print ("Synthesizing...")
stream.play()