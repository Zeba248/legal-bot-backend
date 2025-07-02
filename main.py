from fastapi import FastAPI, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from utils import extract_pdf_text, get_groq_response
import hashlib

app = FastAPI()

# Allow frontend to connect (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Production me specific origin lagao
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store all users' data
user_sessions = {}  # session_id -> {"pdf": "...", "history": [...]}

# Generate unique session_id using client IP
def get_session_id(request: Request):
    client_ip = request.client.host
    return hashlib.sha256(client_ip.encode()).hexdigest()

# PDF Upload Endpoint
@app.post("/upload")
async def upload_pdf(request: Request, file: UploadFile = File(...)):
    session_id = get_session_id(request)
    contents = await file.read()
    text = extract_pdf_text(contents)

    user_sessions[session_id] = {
        "pdf": text,
        "history": [
            {
                "role": "system",
                "content": (
                    "You are ATOZ Legal Chatbot, a smart Indian legal assistant that deeply understands the uploaded legal PDF."
                    " Speak clearly in Hinglish or English depending on user tone. NEVER say you're a text-based AI."
                    " If the user has uploaded a PDF (like FIR, legal notice etc.), use that fully for context. Be realistic, smart, and behave like a real assistant."
                )
            },
            {
                "role": "user",
                "content": f"Maine ek PDF upload kiya hai: {file.filename}. Jab tak mai na kahu, uska analysis mat karo."
            },
            {
                "role": "assistant",
                "content": f"Theek hai, maine {file.filename} receive kar liya. Bataye ab kya karna hai?"
            }
        ]
    }

    return {"message": f"I got your PDF: {file.filename}. Kya karna chahte ho?"}

# Question Asking Endpoint
@app.post("/ask")
async def ask_question(request: Request):
    session_id = get_session_id(request)
    data = await request.json()
    question = data.get("question", "")

    # Load user's session
    session = user_sessions.get(session_id)
    if not session:
        return JSONResponse({"response": "⚠️ Please upload a PDF before asking a question."})

    history = session.get("history", [])
    pdf_text = session.get("pdf", "")

    if pdf_text:
        question += f"\n\n(Use this PDF content if needed):\n{pdf_text[:3000]}"

    history.append({"role": "user", "content": question})

    reply = get_groq_response(history)

    history.append({"role": "assistant", "content": reply})
    user_sessions[session_id]["history"] = history

    return JSONResponse({"response": reply})

# Health check
@app.get("/")
def root():
    return {"message": "ATOZ Legal Chatbot is running."}
