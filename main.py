from fastapi import FastAPI, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from utils import extract_pdf_text, get_groq_response

app = FastAPI()

# CORS setup for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 👈 restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global memory (improved)
chat_memory = {}
pdf_store = {}

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    contents = await file.read()
    text = extract_pdf_text(contents)

    # Save PDF data
    pdf_store["filename"] = file.filename
    pdf_store["text"] = text

    # Setup initial memory
    chat_memory["history"] = [
        {
            "role": "system",
            "content": (
                "You are ATOZ Legal Chatbot, a smart Indian legal assistant."
                " Speak clearly in Hinglish or English based on user tone. NEVER say you're an AI."
                " Always behave realistically. If PDF uploaded (like FIR, notice), use it smartly if asked."
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

        # Include PDF content if exists
        if pdf_text:
            question += f"\n\n(PDF se jarurat ho to use yeh content):\n{pdf_text[:3000]}"

        history.append({"role": "user", "content": question})
        reply = get_groq_response(history)
        history.append({"role": "assistant", "content": reply})
        chat_memory["history"] = history

        return JSONResponse({"response": reply})

    except Exception as e:
        return JSONResponse({"response": f"⚠️ Internal Error: {str(e)}"})

@app.get("/")
def root():
    return {"message": "✅ ATOZ Legal Chatbot backend is running!"}
