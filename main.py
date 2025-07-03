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

# Smart Prompt with Hinglish match
def get_prompt():
    return ChatPromptTemplate.from_messages([
        ("system", 
         "You are ATOZ Legal Assistant — a legal expert in Indian laws. "
         "Always match the user’s language: Hinglish if mixed Hindi-English, or fluent English if pure English. "
         "Never say you are an AI. Be helpful, natural, and smart. "
         "If a PDF is uploaded, use its content. Respond professionally."),
        ("messages", "{chat_history}"),
        ("human", "{input}")
    ])

@app.post("/ask")
async def ask_question(payload: dict):
    question = payload.get("question", "")
    session_id = payload.get("doc_id", str(uuid4()))

    # Create memory for session
    if session_id not in chat_sessions:
        memory = ConversationBufferMemory(return_messages=True)
        chat_sessions[session_id] = memory
    memory = chat_sessions[session_id]

    history = memory.load_memory_variables({}).get("history", [])
    pdf_text = pdf_store.get(session_id)

    # Smart PDF response fix
    if not pdf_text and re.search(r"\b(pdf|upload|file|document)\b", question, re.IGNORECASE):
        response = "Sure, please upload your PDF file whenever you're ready. 📎"
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
        text = "".join([page.extract_text() or "" for page in reader.pages])
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
