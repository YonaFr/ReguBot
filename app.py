import os
import json
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

# =========================
# 🔵 GOOGLE GEMINI API
# =========================
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


# =========================
# 📁 FILE STATE MANAGEMENT
# =========================
STATE_FILE = "app_state.json"

def save_state(new_file_names):
    state = load_state()
    existing = set(state.get("processed_files", []))
    updated = list(existing.union(new_file_names))

    with open(STATE_FILE, "w") as f:
        json.dump({"processed_files": updated}, f)

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return {"processed_files": []}


# =========================
# 📄 PDF PROCESSING
# =========================
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            extracted = page.extract_text() or ""
            text += extracted
    return text


def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    return splitter.split_text(text)


# =========================
# 🧠 VECTOR STORE UPDATE (Gemini Embeddings)
# =========================
def create_or_update_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

    from langchain.vectorstores import FAISS

    try:
        if os.path.exists("faiss_index"):
            db = FAISS.load_local(
                "faiss_index",
                embeddings,
                allow_dangerous_deserialization=True
            )
            db.add_texts(chunks)
        else:
            db = FAISS.from_texts(chunks, embedding=embeddings)

        db.save_local("faiss_index")
        return db

    except Exception as e:
        st.error(f"❌ Gagal memperbarui FAISS index: {e}")

        db = FAISS.from_texts(chunks, embedding=embeddings)
        db.save_local("faiss_index")

        st.warning("⚠️ FAISS index rusak dan telah dibuat ulang.")
        return db


# =========================
# 🧠 VECTOR STORE LOAD (Gemini)
# =========================
@st.cache_resource(show_spinner=False)
def load_vector_store():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    from langchain.vectorstores import FAISS

    try:
        return FAISS.load_local(
            "faiss_index",
            embeddings,
            allow_dangerous_deserialization=True
        )
    except:
        return FAISS.from_texts([""], embedding=embeddings)


# =========================
# 🤖 PROMPT BUILDER
# =========================
def build_gemini_prompt(question, context):
    template = """
Jawablah pertanyaan berikut HANYA berdasarkan pada informasi yang termuat dalam regulasi di bawah ini.
JANGAN gunakan pengetahuan umum atau sumber informasi eksternal lainnya.
Jika jawaban tidak ditemukan dalam daftar regulasi ini, jawab:
"Tidak tersedia atau tidak diatur dalam daftar regulasi yang ditentukan."

Daftar Regulasi (cuplikan hasil pencarian):
{context}

Pertanyaan:
{question}

Jawaban:
"""
    return template.format(question=question, context=context)


# =========================
# 🤖 QA via Gemini (RAG)
# =========================
def user_input(user_question):
    db = load_vector_store()

    # 1) Ambil dokumen relevan
    docs = db.similarity_search(user_question, k=4)
    context = "\n\n".join([d.page_content for d in docs])

    # 2) Buat prompt
    prompt = build_gemini_prompt(user_question, context)

    # 3) Kirim ke Gemini
    model = genai.GenerativeModel("gemini-2.5-flash")

    try:
        response = model.generate_content(prompt)
        return {"output_text": response.text}
    except Exception as e:
        return {"output_text": f"❌ Error from Gemini API: {e}"}


# =========================
# 💬 CHAT HANDLING
# =========================
def clear_chat_history():
    st.session_state.messages = [
        {"role": "assistant", "content": "Tanyakan apapun terkait regulasi pengadaan barang/jasa."}
    ]


# =========================
# 🚀 MAIN STREAMLIT APP
# =========================
def main():
    st.set_page_config(page_title="ReguBot | Regulasi ChatBot", page_icon="🤖")
    st.title("Selamat datang di ReguBot!")

    state = load_state()

    # Sidebar (TIDAK DIUBAH)
    with st.sidebar:
        st.header("📂 Unggah & Proses Dokumen")
        pdf_docs = st.file_uploader("Unggah File PDF", accept_multiple_files=True, type=["pdf"])

        if st.button("Submit & Process"):
            if pdf_docs:
                uploaded_names = [pdf.name for pdf in pdf_docs]
                with st.spinner("🔄 Membaca dan memproses dokumen..."):
                    raw_text = get_pdf_text(pdf_docs)
                    chunks = get_text_chunks(raw_text)

                    create_or_update_vector_store(chunks)
                    save_state(uploaded_names)

                    st.success("✅ Dokumen berhasil diproses.")
            else:
                st.warning("⚠️ Tolong unggah minimal satu dokumen.")

        st.button("🧹 Bersihkan Jejak Digital", on_click=clear_chat_history)

        if state["processed_files"]:
            st.markdown("### 📚 Data Dokumen:")
            for f in state["processed_files"]:
                st.write(f"• " + f)

    # Init chat history
    if "messages" not in st.session_state:
        clear_chat_history()

    # Render chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User typing area
    if prompt := st.chat_input("Ketik di sini..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                try:
                    response = user_input(prompt)
                    full_response = response.get("output_text", "")
                    st.markdown(full_response)

                    st.session_state.messages.append(
                        {"role": "assistant", "content": full_response}
                    )
                except Exception as e:
                    st.error(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
