from fastapi import FastAPI, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from utils import extract_pdf_text, get_groq_response
import re

app = FastAPI()

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
    pdf_store["filename"] = file.filename
    pdf_store["text"] = text

    chat_memory["history"] = [
        {
            "role": "system",
            "content": (
                "You are ATOZ Legal Chatbot, a smart Indian legal assistant. "
                "Speak clearly in Hinglish or English as per user tone. Behave like a real assistant. "
                "If PDF is uploaded, refer to it only if user permits."
            )
        },
        {
            "role": "user",
            "content": f"Maine ek PDF upload kiya hai: {file.filename}. Jab tak mai na kahu, uska analysis mat karo."
        },
        {
            "role": "assistant",
            "content": f"Theek hai, maine {file.filename} receive kar liya. Ab batao kya karna hai?"
        }
    ]
    return {"message": f"✅ I got your PDF: {file.filename}. Kya karna chahte ho?"}

@app.post("/ask")
async def ask_question(request: Request):
    try:
        data = await request.json()
        question = data.get("question", "")
        history = chat_memory.get("history", [])
        pdf_text = pdf_store.get("text", "")

        # Smart handling if user refers to PDF but hasn't uploaded
        if not pdf_text and re.search(r"\b(pdf|upload|file|document)\b", question, re.IGNORECASE):
            response = "📎 Abhi tak mujhe koi PDF file receive nahi hua. Aap upload karein, main wait karunga."
            history.append({"role": "user", "content": question})
            history.append({"role": "assistant", "content": response})
            chat_memory["history"] = history
            return JSONResponse({"response": response})

        if pdf_text:
            question += f"\n\n(Reference PDF if needed):\n{pdf_text[:3000]}"

        history.append({"role": "user", "content": question})
        reply = get_groq_response(history)
        history.append({"role": "assistant", "content": reply})
        chat_memory["history"] = history

        return JSONResponse({"response": reply})

    except Exception as e:
        return JSONResponse({"response": f"⚠️ Internal Error: {str(e)}"})

@app.post("/reset")
async def reset_memory():
    chat_memory.clear()
    pdf_store.clear()
    return {"status": "✅ Memory cleared."}
