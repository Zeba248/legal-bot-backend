from fastapi import FastAPI, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from utils import extract_pdf_text, get_groq_response

app = FastAPI()

# CORS setup for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change to specific domain in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Memory store (just for current session)
chat_memory = {}
pdf_store = {}

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    contents = await file.read()
    text = extract_pdf_text(contents)

    # Store PDF content for future questions
    pdf_store["filename"] = file.filename
    pdf_store["text"] = text

    # Reset chat history for fresh start
    chat_memory["history"] = [
        {"role": "system", "content": f"You are ATOZ Legal Chatbot, a smart Indian legal assistant. Be helpful, realistic, and respond in Hinglish or English as per user tone. Use the PDF if user uploaded anything."},
        {"role": "user", "content": f"Maine ek FIR upload kiya hai — {file.filename}. Jab tak mai na puchu, bas itna yaad rakho."},
        {"role": "assistant", "content": f"Theek hai, maine {file.filename} receive kar liya. Bataye kya karna hai ab?"}
    ]

    return {"message": f"I got your PDF: {file.filename}. Kya karna chahte ho?"}

@app.post("/ask")
async def ask_question(request: Request):
    data = await request.json()
    question = data.get("question", "")

    history = chat_memory.get("history", [])
    pdf_text = pdf_store.get("text")

    # PDF prompt integration if PDF is present
    if pdf_text:
        question = f"{question}\n\nPDF Content (if needed):\n{pdf_text[:3000]}"

    # Add user message
    history.append({"role": "user", "content": question})

    # Get assistant reply
    reply = get_groq_response(history)

    # Add assistant reply to memory
    history.append({"role": "assistant", "content": reply})
    chat_memory["history"] = history

    return JSONResponse({"response": reply})

@app.get("/")
def root():
    return {"message": "ATOZ Legal Chatbot is running."}
