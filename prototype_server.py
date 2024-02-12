from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

import asyncio
from openai import AsyncOpenAI
from nltk.tokenize import sent_tokenize
import json

client = AsyncOpenAI()


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def generate(request_body):
    stream = await client.chat.completions.create(**request_body)
    async for chunk in stream:
        print(chunk.choices[0])
        #yield f"data: {json.dumps(chunk.choices[0])}\n\n"
    yield "data: [DONE]\n\n"

@app.post("/chat")
async def chat_completions(request_body: dict):
    return StreamingResponse(generate(request_body))
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)