import requests
import time
import nltk
from nltk.tokenize import sent_tokenize
import fireworks.client
import threading
import subprocess
import os
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

fireworks.client.api_key = "BRAiIq2Tu95coXwpIMQNxzFqFm4GtRcHwpeNjkVqbydv8pp5"

SERVER_URL = "http://localhost:8000/synthesize/"
SERVER_RESET_URL = "http://localhost:8000/reset/"
chatCompletionGenerator = fireworks.client.ChatCompletion.create(
  model="accounts/fireworks/models/mistral-7b-instruct-4k",
  messages=[
      {
          "role": "system",
          "content": "You are a very creative and longwinded storyteller."
      },
    {
      "role": "user",
      "content": "Hey! Can you tell me a long and very entertaining story?",
    }
  ],
  stream=True,
  n=1,
  max_tokens=800,
  temperature=0.1,
  top_p=0.9, 
)
def send_synthesize_request_async(text, segment_index):
    def send_request():
        requests.post(SERVER_URL, json={"text": text}, params={"segment_index": segment_index})
        #time.sleep(1)
    # Start a new thread for the request
    thread = threading.Thread(target=send_request)
    thread.start()

def send_reset_request():
    response = requests.post(SERVER_RESET_URL)

def main():
    send_reset_request()
    printed_sentences = set()
    response = ""
    segment_index = 0


    for chunk in chatCompletionGenerator:
        if chunk.choices[0].delta.content is not None:
            response += chunk.choices[0].delta.content
            sentences = sent_tokenize(response)
            #time.sleep(1)
            if len(sentences) > 1 and sentences[-2] not in printed_sentences:
                # Send the second-to-last sentence if it's new
                print(sentences[-2])
                send_synthesize_request_async(sentences[-2], segment_index)
                segment_index += 1
                printed_sentences.add(sentences[-2])

        # Check if the stream has ended
        if chunk.choices[0].finish_reason == "stop":
            if sentences and sentences[-1] not in printed_sentences:
                # Send the very last sentence
                print(sentences[-1])
                send_synthesize_request_async(sentences[-1], segment_index)
            break
    time.sleep(1)

if __name__ == "__main__":
    main()