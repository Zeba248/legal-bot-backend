import os
import fitz  # PyMuPDF
from groq.client import Groq 

client = Groq(api_key=os.getenv("GROQ_API_KEY"))  # Secure load

def extract_pdf_text(contents: bytes) -> str:
    with open("temp.pdf", "wb") as f:
        f.write(contents)

    text = ""
    with fitz.open("temp.pdf") as doc:
        for page in doc:
            text += page.get_text()

    os.remove("temp.pdf")
    return text

def get_groq_response(history):
    chat = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=history,
        temperature=0.3
    )
    return chat.choices[0].message.content
