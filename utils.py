import os
import fitz  # PyMuPDF
from groq import Groq
import time

client = Groq(api_key=os.getenv("GROQ_API_KEY"))  # Secure load

# ✅ Extract PDF Text
def extract_pdf_text(contents: bytes) -> str:
    with open("temp.pdf", "wb") as f:
        f.write(contents)

    text = ""
    with fitz.open("temp.pdf") as doc:
        for page in doc:
            text += page.get_text()

    os.remove("temp.pdf")
    return text.strip()

# ✅ Hinglish/English Aware Response Generator
def get_groq_response(history):
    for attempt in range(2):
        try:
            chat = client.chat.completions.create(
                model="llama3-70b-8192",
                messages=history,
                temperature=0.3,
            )
            return chat.choices[0].message.content.strip()

        except Exception as e:
            print(f"[Groq Error] Attempt {attempt + 1} failed: {e}")
            time.sleep(1)

    return "⚠️ Backend is temporarily busy. Please try again shortly."
