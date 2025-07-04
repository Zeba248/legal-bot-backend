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

# Memory and PDF storage
chat_sessions = {}
pdf_store = {}

# ✅ Smart Prompt with Hinglish/English handling & context awareness
def get_prompt():
    return ChatPromptTemplate.from_messages([
        ("system", 
         "You're ATOZ Legal Assistant — a professional, helpful legal expert trained in Indian law. "
         "Reply in the same tone (Hinglish or English) as the user. Don't say 'I'm a language model'. "
         "If a PDF has been uploaded, consider its content (context is passed). "
         "Avoid giving false info. Always behave like a human legal expert, not robotic. "
         "NEVER ask about personal things like hobbies or feelings. "
         "If user says casual things like 'hmm', 'ok', 'acha', respond casually but always redirect to legal topic. "
         "Every answer must help them navigate Indian law smartly and smoothly."),
        ("human", "{input}")
    ])

@app.post("/ask")
async def ask_question(payload: dict):
    question = payload.get("question", "")
    session_id = payload.get("doc_id", str(uuid4()))

    # Memory init
    if session_id not in chat_sessions:
        memory = ConversationBufferMemory(return_messages=True)
        chat_sessions[session_id] = memory
    else:
        memory = chat_sessions[session_id]

    pdf_text = pdf_store.get(session_id)

    # ✅ Smart PDF intent detection and soft reply
    if not pdf_text and re.search(r"\b(pdf|upload|file|document)\b", question, re.IGNORECASE):
        response = "Sure, please go ahead and upload the PDF. I'm ready!"
        memory.chat_memory.add_user_message(question)
        memory.chat_memory.add_ai_message(response)
        return JSONResponse({"response": response})

    # ✅ Context merge
    full_input = f"{pdf_text}\n\n{question}" if pdf_text else question

    # Prompt + model
    prompt = get_prompt()
    llm = ChatGroq(model_name="llama3-70b-8192", temperature=0.3)
    chain = prompt | llm

    # Load full history & inject
    chat_history = memory.load_memory_variables({}).get("history", [])
    try:
        response = chain.invoke({"input": full_input, "chat_history": chat_history})
        memory.save_context({"input": full_input}, {"output": response.content})
        return JSONResponse({"response": response.content})
    except Exception as e:
        return JSONResponse({"response": f"⚠️ Error: {str(e)}"})

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
