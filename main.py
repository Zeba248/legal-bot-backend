from fastapi import FastAPI, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from utils import extract_pdf_text, get_groq_response

app = FastAPI()

# CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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

    # Set up memory
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
    try:
        data = await request.json()
        question = data.get("question", "")

        history = chat_memory.get("history", [])
        pdf_text = pdf_store.get("text")

        if pdf_text:
            question = f"{question}\n\nPDF Content (if needed):\n{pdf_text[:3000]}"

        history.append({"role": "user", "content": question})

        reply = get_groq_response(history)

        if "Groq Error" in reply or "Internal Error" in reply:
            return JSONResponse({"response": "⚠️ Legal server is currently busy. Please try again in 1 minute."})

        history.append({"role": "assistant", "content": reply})
        chat_memory["history"] = history

        return JSONResponse({"response": reply})

    except Exception as e:
        return JSONResponse({"response": f"⚠️ Server error: {str(e)}"})

@app.get("/")
def root():
    return {"message": "ATOZ Legal Chatbot is running."}
