from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import requests
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ✅ Can restrict later to specific domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

@app.post("/ask")
async def ask_legal(request: Request):
    body = await request.json()
    user_input = body.get("question")  # ✅ This must match frontend

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "llama3-70b-8192",
        "messages": [
            {
                "role": "system",
                "content": """
You are a professional legal assistant trained in all major Indian laws and acts, including IPC, CrPC, IBC, Evidence Act, Companies Act, Contract Act, and others.

Your users are practicing lawyers. Always respond in a tone that is professional, legally accurate, and respects legal language style. Do not simplify unless asked.

You must always:
- Provide answers strictly based on Indian Laws and official Acts.
- Clearly cite the relevant act name and section number if applicable.
- Automatically detect the language of the user input.
- If user asks in Hinglish (mix of Hindi and English), respond in Hinglish.
- If user asks in English, respond in professional legal English.
- If user asks in Hindi, respond in formal legal Hindi.
- If the user specifically mentions "reply in Hindi" or "reply in English", always obey that instruction.
- Never switch language unless user input demands it.
- If information is not available in any official Indian act, respond with: "Sorry, no clear provision available as per Indian laws."
- Never generate fake or assumed legal points.
- Never mention that you are an AI, LLM, Groq, GPT or any tech stack. Respond like a human legal assistant.
"""
            },
            {
                "role": "user",
                "content": user_input
            }
        ],
        "temperature": 0.2
    }

    response = requests.post(url, headers=headers, json=data)
    result = response.json()
    return { "response": result["choices"][0]["message"]["content"].strip()}


