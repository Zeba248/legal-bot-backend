import os
import requests
import pdfplumber
import io

def extract_pdf_text(file_bytes):
    try:
        text = ""
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
        return text
    except Exception as e:
        return f"PDF Extraction Error: {str(e)}"

def get_groq_response(history):
    try:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            return "API key missing. Please check GROQ_API_KEY in Render settings."

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        data = {
            "model": "mixtral-8x7b-32768",
            "messages": history,
            "temperature": 0.4
        }

        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=data
        )

        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            return f"Groq Error: {response.status_code} — {response.text}"

    except Exception as e:
        return f"Internal Error: {str(e)}"
