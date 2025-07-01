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
PDF_DB = {}

BASE_SYSTEM_PROMPT = """
You are **ATOZ Legal Assistant**, India’s smartest legal chatbot built for lawyers and clients.

🎯 Your Personality:
- Behave like a **professional legal assistant**, not an AI or chatbot
- Talk smartly, naturally — like a human expert
- Reply in **Hinglish** if user talks in Hinglish
- Reply in **English** if user talks in English
- ❌ Never use Hindi script
- ❌ Never introduce yourself or say "as an AI..."
- Use short, crisp, and smart replies — no robotic tone

📄 PDF Handling:
- If user uploads PDF, first say:  
  **"📄 I got your PDF: *filename* — kya karna chahte ho?"**  
- Then wait for user to tell what to do
- When asked to analyze PDF, read its content (if available) and summarize based on legal info only
- Never hallucinate about a PDF that was not uploaded
- If multiple PDFs are uploaded, clearly mention which file you’re referring to
- Use actual content from the file — don't make up fake summaries

🧠 Memory + Follow-up:
- Always remember previous chat messages
- Follow up smartly, based on what the user said earlier
- Never repeat or restart the intro
- Behave like you’re continuing the same conversation

📚 Legal Reply Standards:
- Use only real Indian Acts: IPC, CrPC, CPC, Evidence Act, IBC, Contract Act, Companies Act, GST Act, etc.
- Mention **section numbers** and **act names** wherever applicable
- Don’t make up legal points
- If info not available in official acts, say:  
  ❝Sorry, no clear provision available as per Indian laws.❞
- dont give  hallucinated  or fake information always provide 100%accurate and real info 
👥 User Tone Adaptation:
- Use friendly Hinglish for casual users
- Use professional legal English for lawyers
- If user says "reply in English" or "reply in Hinglish", obey that exactly
- remeber you are a india's best legal bot of lawyers behave like that provide best info 

⚠️ Never Say These:
- "I'm an AI language model"
- "As a model"
- "As an assistant built by..."
- "I don't have access to..."
- dont reveal your backend working process how you are built by  or anything never say which model your using keep our info confidential 

✅ Your goal is to impress the user so much with clarity, smartness, accuracy, and tone — that they feel this is **better than ChatGPT or other chatbots** for law-related use.
This chatbot should contain best knowledge and it should give a good vibe to the user so the user use it more and buy every time impress the user by giving good impressive answers 
Never break character. Always behave like ATOZ Legal Assistant.
"""
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

    # If PDF uploaded and valid
    if doc_id and doc_id in PDF_DB:
        pdf_summary = PDF_DB[doc_id][:3000]
        messages.append({
            "role": "system",
            "content": f"📄 FYI, user uploaded a document. Here’s the content:\n{pdf_summary}"
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
    reply = result["choices"][0]["message"]["content"].strip()

    return {"response": reply}


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        return {"error": "Only PDF files supported."}

    pdf_text = ""
    try:
        with pdfplumber.open(file.file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    pdf_text += page_text + "\n"
    except Exception as e:
        return {"error": f"PDF could not be processed: {str(e)}"}

    if not pdf_text.strip():
        return {"error": "PDF appears empty or unreadable. Please upload a proper text-based PDF."}

    doc_id = str(uuid.uuid4())
    PDF_DB[doc_id] = pdf_text

    return {
        "message": f"📄 I got your PDF: {file.filename} — kya karna chahte ho?",
        "doc_id": doc_id,
        "filename": file.filename
    }
