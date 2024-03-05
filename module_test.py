import asyncio

from xtts_engine import XttsEngine
from chat_engine import OpenAIInterface
from anthropic_interface import AnthropicInterface
#

async def main():
    chatbot = AnthropicInterface()
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