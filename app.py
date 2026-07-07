import tempfile
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate


# -------------------------
# App Config
# -------------------------
st.set_page_config(page_title="MediScan RAG Assistant", page_icon="🩺", layout="wide")
st.title("🩺 MediScan RAG Assistant")
st.caption("Upload a medical PDF and ask grounded questions with page citations.")


# -------------------------
# Prompt
# -------------------------
RAG_PROMPT = ChatPromptTemplate.from_template("""
You are MediScan, a medical document assistant.

Rules:
1) Answer ONLY from the provided context.
2) If answer is not in context, say exactly: "I don't know based on the provided document."
3) Be concise, factual, and avoid speculation.
4) Do not provide medical advice; only summarize document content.
5) At the end, provide citations as: Sources: p.X, p.Y

Question:
{question}

Context:
{context}
""")


# -------------------------
# Helpers
# -------------------------
def format_context(docs):
    parts = []
    for d in docs:
        page = d.metadata.get("page_number", "?")
        txt = d.page_content.strip().replace("\n", " ")
        parts.append(f"[Page {page}] {txt}")
    return "\n\n".join(parts)


def format_citations(docs):
    pages = sorted(set(d.metadata.get("page_number", "?") for d in docs))
    return ", ".join([f"p.{p}" for p in pages])


@st.cache_resource(show_spinner=False)
def get_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_vectorstore(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    docs = PyPDFLoader(tmp_path).load()

    for i, d in enumerate(docs):
        page_idx = d.metadata.get("page", i)
        d.metadata["page_number"] = int(page_idx) + 1
        d.metadata["source_file"] = uploaded_file.name

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=180,
        separators=["\n\n", "\n", ". ", "; ", ", ", " "],
        length_function=len,
    )
    chunks = splitter.split_documents(docs)

    for idx, d in enumerate(chunks):
        d.metadata["chunk_id"] = idx
        d.metadata["char_count"] = len(d.page_content)

    embeddings = get_embedding_model()

    persist_dir = tempfile.mkdtemp(prefix="mediscan_chroma_")
    vs = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="mediscan_collection",
        persist_directory=persist_dir,
    )

    return vs, len(docs), len(chunks)


def answer_question(question, retriever, llm):
    retrieved_docs = retriever.invoke(question)
    context = format_context(retrieved_docs)
    citations = format_citations(retrieved_docs)

    messages = RAG_PROMPT.format_messages(question=question, context=context)
    answer = llm.invoke(messages).content.strip()

    if "Sources:" not in answer:
        answer += f"\n\nSources: {citations}"

    return answer, retrieved_docs


# -------------------------
# Sidebar
# -------------------------
st.sidebar.header("⚙️ Settings")
groq_api_key = st.sidebar.text_input("GROQ_API_KEY", type="password")
model_name = st.sidebar.selectbox(
    "Model",
    ["llama-3.1-8b-instant", "llama-3.1-70b-versatile"],
    index=0
)
top_k = st.sidebar.slider("Top-k chunks", 2, 8, 4)


# -------------------------
# Session State
# -------------------------
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "ready" not in st.session_state:
    st.session_state.ready = False
if "history" not in st.session_state:
    st.session_state.history = []


# -------------------------
# Upload + Process
# -------------------------
uploaded_pdf = st.file_uploader("📄 Upload medical PDF", type=["pdf"])

if uploaded_pdf is not None:
    if st.button("Process PDF"):
        with st.spinner("Processing PDF and building vector index..."):
            vectorstore, page_count, chunk_count = build_vectorstore(uploaded_pdf)
            st.session_state.retriever = vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={"k": top_k, "fetch_k": 12, "lambda_mult": 0.5}
            )
            st.session_state.ready = True
            st.session_state.history = []

        st.success(f"✅ Ready! Pages: {page_count} | Chunks: {chunk_count}")


# -------------------------
# Ask
# -------------------------
if st.session_state.ready:
    if not groq_api_key:
        st.warning("Please enter your GROQ_API_KEY in the sidebar.")
    else:
        llm = ChatGroq(
            model=model_name,
            api_key=groq_api_key,
            temperature=0
        )

        question = st.text_input("Ask a question about the uploaded document")
        if st.button("Ask") and question.strip():
            with st.spinner("Generating grounded answer..."):
                answer, docs = answer_question(question, st.session_state.retriever, llm)

            st.session_state.history.append({
                "question": question,
                "answer": answer,
                "docs": docs
            })

        for i, item in enumerate(reversed(st.session_state.history), start=1):
            st.markdown(f"### ❓ Q{i}: {item['question']}")
            st.markdown(item["answer"])

            with st.expander("Retrieved evidence"):
                for j, d in enumerate(item["docs"], start=1):
                    st.markdown(
                        f"**{j}. Page {d.metadata.get('page_number')} | Chunk {d.metadata.get('chunk_id')}**"
                    )
                    snippet = d.page_content[:700]
                    if len(d.page_content) > 700:
                        snippet += "..."
                    st.write(snippet)

            st.markdown("---")
else:
    st.info("Upload a PDF and click **Process PDF** to start.")