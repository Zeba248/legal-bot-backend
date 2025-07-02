from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uuid
import os

from utils import extract_pdf_text, get_groq_response

app = FastAPI()

# Allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or restrict to your Netlify domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory chat memory (doc_id -> [messages])
memory_store = {}

class AskRequest(BaseModel):
    question: str
    doc_id: str = None

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    contents = await file.read()
    file_text = extract_pdf_text(contents)
    doc_id = str(uuid.uuid4())

    # Save PDF text in memory
    memory_store[doc_id] = [{
        "role": "system",
        "content": f"You are ATOZ Legal Assistant. Only respond in Hinglish (casual + clear). The user will upload Indian legal PDFs and ask follow-up questions. NEVER hallucinate if PDF not found. If PDF uploaded, always confirm filename and ask what to do next politely like a real assistant."
    }]
    memory_store[doc_id].append({
        "role": "user",
        "content": f"Please read and understand this PDF:\n\n{file_text[:3000]}"
    })
    memory_store[doc_id].append({
        "role": "assistant",
        "content": f"I got your PDF: {file.filename} — kya karna chahte ho?"
    })

    return {
        "message": f"📄 Uploaded: {file.filename}\nThanks! Processing your PDF…",
        "doc_id": doc_id
    }

@app.post("/ask")
async def ask_question(data: AskRequest):
    doc_id = data.doc_id
    question = data.question.strip()

    if not doc_id or doc_id not in memory_store:
        return { "response": "⚠️ Please upload a legal PDF first so I can help properly." }

    history = memory_store[doc_id]
    history.append({ "role": "user", "content": question })

    reply = get_groq_response(history)

    history.append({ "role": "assistant", "content": reply })
    return { "response": reply }
