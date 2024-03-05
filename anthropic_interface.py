import asyncio
from anthropic import AsyncAnthropic
from nltk.tokenize import sent_tokenize
import json

class AnthropicInterface():
    def __init__(self, model="claude-3-sonnet-20240229", default_system="You are a terse and sarcastic assistant."):
        self.client = AsyncAnthropic()
        self.model = model
        self.system = default_system
        self.messages = []
        self.current_response = ""

    def add_user_message(self, content):
        self.messages.append({"role": "user", "content": content})
    
    async def stream_generator(self):
        self.current_response = ""
        async with self.client.messages.stream(
            max_tokens=1024,
            messages=self.messages,
            model=self.model
        ) as stream:
            async for text in stream.text_stream:
                self.current_response += text
                yield text
                #print(text, end="", flush=True)
            #print()
            self.messages.append({"role": "assistant", "content": self.current_response})
        
        #message = await stream.get_final_message()
        #print(message.model_dump_json(indent=2))

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
                yield sentences[-1]

if __name__ == "__main__":
    async def main():
        chatbot = AnthropicInterface()
        chatbot.add_user_message("Hello!")
        #await chatbot.stream_generator()
        async for sentence in chatbot.sentence_generator():
            print(sentence, end=" ", flush=True)
    
    asyncio.run(main())