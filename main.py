from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from uuid import uuid4
from langchain_core.messages import AIMessage, HumanMessage
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from PyPDF2 import PdfReader
import re

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Session memory and PDFs
chat_sessions = {}
pdf_store = {}

# Smart Prompt
def get_prompt():
    return ChatPromptTemplate.from_messages([
        ("system", 
         "You're ATOZ Legal Assistant — a professional, helpful legal expert trained in Indian law. "
         "Reply in the same tone (Hinglish or English) as the user. Don't say 'I'm a language model' or act robotic. "
         "If a PDF has been uploaded, consider its content (context is passed). "
         "Avoid giving false statements, and always reply naturally and intelligently like a legal expert."),
        ("messages", "{chat_history}"),
        ("human", "{input}")
    ])

@app.post("/ask")
async def ask_question(payload: dict):
    question = payload.get("question", "")
    session_id = payload.get("doc_id", str(uuid4()))

    # Create new memory for new session
    if session_id not in chat_sessions:
        memory = ConversationBufferMemory(return_messages=True)
        chat_sessions[session_id] = memory
    else:
        memory = chat_sessions[session_id]

    history = memory.load_memory_variables({}).get("history", [])

    # Smart PDF detection and reply
    pdf_text = pdf_store.get(session_id)
    if not pdf_text and re.search(r"\b(pdf|upload|file|document)\b", question, re.IGNORECASE):
        response = "📎 Abhi tak mujhe koi PDF file receive nahi hua. Aap upload karein, main wait karunga."
        memory.chat_memory.add_user_message(question)
        memory.chat_memory.add_ai_message(response)
        return JSONResponse({"response": response})

    # Build final input
    full_context = pdf_text + "\n\n" + question if pdf_text else question

    prompt = get_prompt()
    model = ChatGroq(model_name="llama3-70b-8192", temperature=0.3)
    chain = prompt | model | memory

    response = chain.invoke({"input": full_context})
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
