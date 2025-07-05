# ✅ Updated main.py with per-chat memory + PDF logic (structure unchanged)
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from uuid import uuid4
from langchain.prompts.chat import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
from PyPDF2 import PdfReader
import re

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

chat_sessions = {}   # doc_id -> memory
pdf_store = {}        # doc_id -> pdf text


def get_prompt():
    return ChatPromptTemplate.from_messages([
        ("system",
         "You're ATOZ Legal Assistant — a professional, helpful legal expert trained in Indian law. "
         "Reply in the same tone (Hinglish or English) as the user. Don't say 'I'm a language model' or act robotic. "
         "If a PDF has been uploaded, consider its content (context is passed). "
         "Avoid giving false statements, and always reply naturally and intelligently like a legal expert."),
        ("human", "{input}")
    ])


@app.post("/ask")
async def ask_question(payload: dict):
    question = payload.get("question", "")
    session_id = payload.get("doc_id", str(uuid4()))

    # Load memory
    if session_id not in chat_sessions:
        memory = ConversationBufferMemory(return_messages=True)
        chat_sessions[session_id] = memory
    else:
        memory = chat_sessions[session_id]

    # Load PDF for that session
    pdf_text = pdf_store.get(session_id)

    # Smart PDF detection
    if not pdf_text and re.search(r"\b(pdf|upload|file|document)\b", question, re.IGNORECASE):
        response = "Sure, please go ahead and upload the PDF. I'm ready!"
        memory.chat_memory.add_user_message(question)
        memory.chat_memory.add_ai_message(response)
        return JSONResponse({"response": response})

    full_input = f"{pdf_text}\n\n{question}" if pdf_text else question

    prompt = get_prompt()
    llm = ChatGroq(model_name="llama3-70b-8192", temperature=0.3)
    chain = prompt | llm

    chat_history = memory.load_memory_variables({}).get("history", [])
    response = chain.invoke({"input": full_input, "chat_history": chat_history})
    memory.save_context({"input": full_input}, {"output": response.content})
    return JSONResponse({"response": response.content})


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        reader = PdfReader(file.file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        session_id = str(uuid4())
        pdf_store[session_id] = text.strip()
        return {
            "doc_id": session_id,
            "message": f"✅ I got your PDF: {file.filename}. Kya karna chahte ho?"
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/")
def root():
    return {"status": "Legal Bot is running."}
