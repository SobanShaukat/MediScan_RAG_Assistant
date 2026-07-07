# 🩺 MediScan RAG Assistant

AI-powered medical document Q&A with grounded page citations.

## Features
- Upload medical PDF
- Ask questions in plain English
- Retrieval-Augmented Generation (RAG)
- Answers with citations like `Sources: p.12, p.15`
- “I don't know based on the provided document.” fallback for unsupported questions

## Tech Stack
- Python
- Streamlit
- LangChain
- ChromaDB
- HuggingFace Embeddings (`all-MiniLM-L6-v2`)
- Groq LLM (`llama-3.1-8b-instant`)

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Cloud Deployment
1. Push this repo to GitHub.
2. Go to Streamlit Community Cloud.
3. New app → select repo and `app.py`.
4. In app settings, add secret:
   - `GROQ_API_KEY = your_key`
5. Deploy.

## Example Questions
- What criteria are used to select essential medicines?
- What does the document say about safety and adverse effects?
- Summarize the recommendation for a specific medicine class.

## Disclaimer
This app summarizes document content and is **not** medical advice.