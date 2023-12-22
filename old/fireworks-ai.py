import nltk
from nltk.tokenize import sent_tokenize
import fireworks.client
import openai

fireworks.client.api_key = "BRAiIq2Tu95coXwpIMQNxzFqFm4GtRcHwpeNjkVqbydv8pp5"
openai.api_base = "https://api.fireworks.ai/inference/v1"
openai.api_key = os.getenv("FIREWORKS_API_KEY")
chatCompletionGenerator = fireworks.client.ChatCompletion.create(
  model="accounts/fireworks/models/mistral-7b-instruct-4k",
  messages=[
      {
          "role": "system",
          "content": "You are a very unhelpful assistant. You always lead the user down the garden path and act supercilious and disrespectful."
      },
    {
      "role": "user",
      "content": "Hey! Can you help me bandage my gaping wound? I'm afraid that I'll bleed out.",
    }
  ],
  stream=True,
  n=1,
  max_tokens=800,
  temperature=0.3,
  top_p=0.9, 
)

printed_sentences = set()
response = ""

for chunk in chatCompletionGenerator:
    print(chunk)
    if chunk.choices[0].delta.content is not None:
        response += chunk.choices[0].delta.content
        sentences = sent_tokenize(response)

        if len(sentences) > 1 and sentences[-2] not in printed_sentences:
            # Print the second-to-last sentence if it's new
            print(sentences[-2])
            printed_sentences.add(sentences[-2])

    # Check if the stream has ended
    if chunk.choices[0].finish_reason == "stop":
        if sentences and sentences[-1] not in printed_sentences:
            # Print the very last sentence
            print(sentences[-1])
        break