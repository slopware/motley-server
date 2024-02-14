import logging
logger = logging.getLogger()
logger.disabled = True
import asyncio
from openai import AsyncOpenAI
from nltk.tokenize import sent_tokenize
import json
import requests
import httpx
from xtts_module import XttsEngine

class OpenAIEngine:
    
    def __init__(self, system_base="You are a terse and sarcastic assistant.", model="gpt-4-1106-preview"):
        self.client = AsyncOpenAI()
        self.model = model
        self.messages = [{"role": "system", "content": system_base}]
        self.current_response = ""
    def add_user_message(self, content):
        self.messages.append({"role": "user", "content": content})

    async def stream_generator(self):
        self.current_response = ""
        stream = await self.client.chat.completions.create(
            model = self.model,
            messages = self.messages,
            stream = True
        )
        async for chunk in stream:
            self.current_response += chunk.choices[0].delta.content or ""
            yield chunk.choices[0].delta.content or ""
        self.messages.append({"role": "assistant", "content": self.current_response})

    async def sentence_generator(self):
        printed_sentences = []
        async for chunk in self.stream_generator():
            sentences = sent_tokenize(self.current_response)
            if len(sentences) > 1 and sentences[-2] not in printed_sentences:
                printed_sentences.append(sentences[-2])
                yield sentences[-2]
            if chunk == "" and len(sentences) > 1:
                yield sentences[-1]
            if chunk == "" and len(sentences) == 1:
                yield sentences[1]

async def main():
    chatbot = OpenAIEngine(model="gpt-4")
    tts_reader = XttsEngine()
    try:
        while True:
            user_input = input("User: ")
            chatbot.add_user_message(user_input)
            print("AI: ", end="", flush=True)
            async for donk in chatbot.sentence_generator():
                tts_reader.add_text_for_synthesis(donk)
                print(donk, end=" ", flush=True)
            print("")
    except KeyboardInterrupt:
        tts_reader.stop()
        print("exiting...")
    

if __name__ == '__main__':
    asyncio.run(main())