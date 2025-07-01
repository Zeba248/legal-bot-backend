from fastapi import FastAPI, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import requests
import pdfplumber
import uuid
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (you can restrict later)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# In-memory PDF store
PDF_DB = {}

BASE_SYSTEM_PROMPT = """
You are ATOZ Legal Assistant — a smart, highly skilled legal expert in Indian law.

🎯 Your behavior:
- Answer like a professional human legal assistant (not AI).
- Don’t give robotic or repetitive intros.
- If user uploads a PDF, smartly reference it.
- Continue conversations naturally. Use memory logically.
- Use Hinglish if user chats in Hinglish. Use English if user chats in English. Never use Hindi script.
- Base answers only on real Indian laws (IPC, CrPC, Evidence Act, IBC, Companies Act, etc.)
- If user asks about a PDF, use its content to reply.
"""

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        return {"error": "Only PDF files supported."}

    # Extract PDF text
    pdf_text = ""
    with pdfplumber.open(file.file) as pdf:
        for page in pdf.pages:
            pdf_text += page.extract_text() or ""

    # Save in memory with a unique ID
    doc_id = str(uuid.uuid4())
    PDF_DB[doc_id] = {
        "filename": file.filename,
        "text": pdf_text
    }

    return {
        "message": f"📄 I got your PDF **{file.filename}** — Kya karna chahte ho?\n👉 Summarize / Extract legal info / Ask questions",
        "doc_id": doc_id,
        "filename": file.filename
    }

@app.post("/ask")
async def ask_legal(request: Request):
    body = await request.json()
    user_input = body.get("question", "")
    chat_history = body.get("history", [])
    doc_id = body.get("doc_id")

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    # Build the message list
    messages = [{"role": "system", "content": BASE_SYSTEM_PROMPT}]

    # Attach PDF context if available
    if doc_id and doc_id in PDF_DB:
        pdf_chunk = PDF_DB[doc_id]["text"][:3000]
        messages.append({
            "role": "system",
            "content": f"📁 The user has uploaded a legal document titled '{PDF_DB[doc_id]['filename']}'. You can use this content while replying:\n\n{pdf_chunk}"
        })

    # Add past chat history if any
    for msg in chat_history:
        messages.append({
            "role": msg["role"],
            "content": msg["content"]
        })

    messages.append({"role": "user", "content": user_input})

    payload = {
        "model": "llama3-70b-8192",
        "messages": messages,
        "temperature": 0.4,
        "max_tokens": 1500
    }

    response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload)
    result = response.json()

    content = result["choices"][0]["message"]["content"].strip()
    return {"response": content}
