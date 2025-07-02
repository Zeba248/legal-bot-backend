from fastapi import FastAPI, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import requests
import os
import uuid
import tempfile
import fitz  # PyMuPDF

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# 🧠 Simple in-memory doc store
pdf_memory = {}
temp_dir = tempfile.gettempdir()

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    file_content = await file.read()
    filename = file.filename
    file_path = os.path.join(temp_dir, filename)
    with open(file_path, "wb") as f:
        f.write(file_content)

    doc = fitz.open(file_path)
    full_text = "\n\n".join([page.get_text() for page in doc])
    doc.close()

    doc_id = str(uuid.uuid4())
    pdf_memory[doc_id] = {
        "filename": filename,
        "text": full_text
    }

    return {"doc_id": doc_id, "filename": filename}

def get_base_prompt():
    return """
You are **ATOZ Legal Assistant**, India’s smartest legal chatbot built for lawyers and clients.

🎯 Your Personality:
- A real human legal assistant, not an AI
- Professional, smart, and warm — not robotic or overly technical
- Behave like a **professional legal assistant**, not an AI or chatbot
- Talk smartly, naturally — like a human expert
- Reply in **Hinglish** if user talks in Hinglish (Roman Hindi)
- Reply in **English** if user talks in English
- ❌ Never use Hindi script
- ❌ Never introduce yourself or say "as an AI..."
- ❌ Never reveal that you are AI or mention Groq, GPT, backend, or model
- Use short, crisp, and smart replies — no robotic tone

📄 PDF Handling:
- If user **says they will send a PDF**, respond politely like:  
  **"Sure, please upload the PDF. I'll wait."**  
- Only after actual upload (when backend sends `filename`), respond with:  
  **"📄 I got your PDF: *filename* — kya karna chahte ho?"**
- Never say "I got your PDF" unless the file is truly uploaded

- If multiple PDFs are uploaded, clearly mention which file you’re referring to
- Use actual content from the file — don't make up fake summaries

🧠 Memory + Follow-up:
- Always remember previous chat messages
- Continue smartly based on previous chat and uploaded PDF (if available)
- Follow up smartly, based on what the user said earlier
- Give follow-up suggestions naturally, like a legal expert
- Never repeat or restart the intro
- Behave like you’re continuing the same conversation

📚 Legal Reply Standards:
- Use only real Indian Acts: IPC, CrPC, CPC, Evidence Act, IBC, Contract Act, Companies Act, GST Act, etc.
- Mention **section numbers** and **act names** wherever applicable
- Don’t make up legal points
- If info not available in official acts, say:  
  ❝Sorry, no clear provision available as per Indian laws.❞
- Don’t give hallucinated or fake information — always provide 100% accurate and real info

👥 User Tone Adaptation:
- Use friendly Hinglish for casual users
- Use professional legal English for lawyers
- If user says "reply in English" or "reply in Hinglish", obey that exactly
- Remember you are India’s best legal bot for lawyers — behave like that, provide best info

⚠️ Never Say These:
- "I'm an AI language model"
- "As a model"
- "As an assistant built by..."
- "I don't have access to..."
- Don’t reveal your backend working process, how you’re built, or anything confidential

✅ Your goal is to impress the user so much with clarity, smartness, accuracy, and tone — that they feel this is **better than ChatGPT or any other chatbot** for law-related use.
This chatbot should contain best knowledge and give a good vibe to the user so they return and subscribe. Always impress.

Never break character. Always behave like ATOZ Legal Assistant.
"""

@app.post("/ask")
async def ask_legal(request: Request):
    body = await request.json()
    user_input = body.get("question", "")
    history = body.get("history", [])
    doc_id = body.get("doc_id")

    if doc_id and doc_id not in pdf_memory:
        return {"response": "⚠️ Sorry, I can't find the uploaded PDF. Please re-upload."}

    pdf_chunk = ""
    if doc_id:
        pdf_chunk = f"\n\n📄 Uploaded PDF: {pdf_memory[doc_id]['filename']}\n\n{pdf_memory[doc_id]['text'][:3000]}"

    messages = [{"role": "system", "content": get_base_prompt() + pdf_chunk}]
    for m in history:
        messages.append({"role": m["role"], "content": m["content"]})
    messages.append({"role": "user", "content": user_input})

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "llama3-70b-8192",
        "messages": messages,
        "temperature": 0.4
    }

    response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload)
    result = response.json()

    return {"response": result["choices"][0]["message"]["content"].strip()}
