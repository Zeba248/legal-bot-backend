from fastapi import FastAPI, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import requests
import pdfplumber
import uuid
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

# ✅ Temporary PDF storage (in-memory dict)
PDF_DB = {}

BASE_SYSTEM_PROMPT = """
You are ATOZ Legal Assistant, a highly skilled legal expert in Indian law.

🌟 Your goals:
- Give helpful, smart legal replies to lawyers and clients
- Always sound natural and human — avoid robotic tone or AI disclaimers
- Answer based only on official Indian Acts like IPC, CrPC, IBC, Evidence Act, Contract Act, Companies Act, etc.
- If user chats in Hinglish, reply in Hinglish. If English, reply in English. Never use Hindi script.
- Continue the conversation smoothly based on previous chats
- Ask helpful follow-up questions naturally
- When PDF is uploaded, say: "📄 I got your PDF: *filename* — kya karna chahte ho?"
- Never mention you're an AI or model. Behave like a real legal assistant.
"""

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

    messages = [{"role": "system", "content": BASE_SYSTEM_PROMPT}]

    for msg in chat_history:
        messages.append({"role": msg["role"], "content": msg["content"]})

    # 📎 Include PDF content if available
    if doc_id and doc_id in PDF_DB:
        pdf_content = PDF_DB[doc_id][:3000]

        # Smart append PDF text for relevant insights
        if "summarize" in user_input.lower() or "insight" in user_input.lower():
            user_input += f"\n\nYe uploaded case file hai:\n{pdf_content[:1000]}"

        messages.append({
            "role": "system",
            "content": f"📁 FYI, user ne ek PDF diya tha. Summary:\n{pdf_content[:1000]}"
        })

    messages.append({"role": "user", "content": user_input})

    payload = {
        "model": "llama3-70b-8192",
        "messages": messages,
        "temperature": 0.4,
        "max_tokens": 1024
    }

    response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload)
    result = response.json()
    content = result["choices"][0]["message"]["content"].strip()
    return {"response": content}

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        return {"error": "Only PDF files supported."}

    pdf_text = ""
    with pdfplumber.open(file.file) as pdf:
        for page in pdf.pages:
            pdf_text += page.extract_text() or ""

    doc_id = str(uuid.uuid4())
    PDF_DB[doc_id] = pdf_text

    return {
        "message": f"📄 I got your PDF: {file.filename} — kya karna chahte ho?",
        "doc_id": doc_id,
        "filename": file.filename
    }
