# 🩺 MediScan RAG Assistant

AI-powered medical document Q&A with grounded page citations.

## Features
- Upload medical PDF
- Ask questions in plain English
- Retrieval-Augmented Generation (RAG)
- Citation-style answers: Sources: p.X, p.Y
- Safe fallback: "I don't know based on the provided document."

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deployment
Deploy on Streamlit Community Cloud with:
- app file: app.py
- secret: GROQ_API_KEY
