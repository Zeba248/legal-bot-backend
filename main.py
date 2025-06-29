from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import requests
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

@app.post("/ask")
async def ask_legal(request: Request):
    body = await request.json()
    user_input = body.get("question")

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "llama3-70b-8192",
        "messages": [
            {"role": "system", "content": """
You are a professional Indian legal assistant...
(⚠️ Yaha pe tu apna final prompt daalna — jo tu use kar rahi hai Hinglish detection etc)
"""},
            {"role": "user", "content": user_input}
        ],
        "temperature": 0.2
    }

    response = requests.post(url, headers=headers, json=data)
    result = response.json()
    return {"response": result["choices"][0]["message"]["content"]}

