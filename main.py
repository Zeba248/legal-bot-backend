from fastapi import FastAPI, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from utils import extract_pdf_text, get_groq_response

app = FastAPI()

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict this in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

chat_memory = {}
pdf_store = {}

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    contents = await file.read()
    text = extract_pdf_text(contents)

    # Store PDF
    pdf_store["filename"] = file.filename
    pdf_store["text"] = text

    # Setup memory with strict role instructions
    chat_memory["history"] = [
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

    return {"message": f"I got your PDF: {file.filename}. Kya karna chahte ho?"}

@app.post("/ask")
async def ask_question(request: Request):
    data = await request.json()
    question = data.get("question", "")
    history = chat_memory.get("history", [])
    pdf_text = pdf_store.get("text")

    # Add PDF content if available
    if pdf_text:
        question += f"\n\n(Use this PDF content if needed):\n{pdf_text[:3000]}"

    # Add question to memory
    history.append({"role": "user", "content": question})

    # Get response
    reply = get_groq_response(history)
    
    if reply.startswith("Groq Error") or reply.startswith("Internal Error"):
    return JSONResponse({"response": "⚠️ Legal server is currently busy. Please try again in 1 minute."})

    
    # Save reply to memory
    history.append({"role": "assistant", "content": reply})
    chat_memory["history"] = history

    return JSONResponse({"response": reply})

@app.get("/")
def root():
    return {"message": "ATOZ Legal Chatbot is running."}
