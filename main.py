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

# Session storage for chat + PDFs
chat_sessions = {}
pdf_store = {}

# Law list for real legal context
INDIAN_LAWS = ['IPC', 'CrPC', 'Companies Act', 'Income Tax Act', 'Evidence Act', 'CPC']

def get_prompt():
    return ChatPromptTemplate.from_messages([
        ("system",
         "You are ATOZ Legal Assistant — a professional Indian legal expert. "
         "Always reply ONLY using real Indian laws (IPC, CrPC, Companies Act, etc). "
         "If the user asks a legal question, give a direct, accurate answer from these laws. "
         "If user talks casual, reply in the same tone but always gently bring back to legal context (e.g. 'Kya legal query puchna hai?'). "
         "Never discuss hobbies, opinions, or unrelated topics. "
         "Reply in Hinglish or English matching user tone. "
         "If PDF is uploaded, consider its content. Never hallucinate about documents. "
         "Never say 'I'm an AI language model', never act robotic. Be natural and lawyer-like."),
        ("human", "{input}")
    ])

@app.post("/ask")
async def ask_question(payload: dict):
    question = payload.get("question", "")
    session_id = payload.get("doc_id", str(uuid4()))

    # Memory for session
    if session_id not in chat_sessions:
        memory = ConversationBufferMemory(return_messages=True)
        chat_sessions[session_id] = memory
    else:
        memory = chat_sessions[session_id]

    pdf_text = pdf_store.get(session_id, "")

    # PDF upload flow
    if not pdf_text and re.search(r"\b(pdf|upload|file|document)\b", question, re.IGNORECASE):
        response = "Sure, please upload your PDF. I'm ready!"
        memory.chat_memory.add_user_message(question)
        memory.chat_memory.add_ai_message(response)
        return JSONResponse({"response": response})

    # Context + input
    full_input = f"{pdf_text}\n\n{question}" if pdf_text else question

    # LLM pipeline
    prompt = get_prompt()
    llm = ChatGroq(model_name="llama3-70b-8192", temperature=0.3)
    chain = prompt | llm

    # Memory-aware response
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
