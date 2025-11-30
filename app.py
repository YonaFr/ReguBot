import os
import json
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
import requests

load_dotenv()



# ====================================
# 🔵 GOOGLE GEMINI (REST API CONFIG)
# ====================================
GEMINI_ENDPOINT = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-2.5-flash:generateContent"
)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")



# ====================================
# 📁 FILE STATE MANAGEMENT
# ====================================
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



# ====================================
# 📄 PDF PROCESSING
# ====================================
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



# ====================================
# 🧠 VECTOR STORE UPDATE (FAISS)
# ====================================
def create_or_update_vector_store(chunks):
    embeddings = OllamaEmbeddings(model="llama3.2")

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



# ====================================
# 🧠 LOAD FAISS (CACHED)
# ====================================
@st.cache_resource(show_spinner=False)
def load_vector_store():
    embeddings = OllamaEmbeddings(model="llama3.2")

    try:
        return FAISS.load_local(
            "faiss_index",
            embeddings,
            allow_dangerous_deserialization=True
        )
    except:
        # recovery
        return FAISS.from_texts([""], embedding=embeddings)



# ====================================
# 🤖 GEMINI PROMPT (RAG)
# ====================================
def build_gemini_prompt(question, context):
    template = f"""
Jawablah pertanyaan berikut HANYA berdasarkan konteks regulasi di bawah ini.
JANGAN gunakan pengetahuan umum atau sumber eksternal lainnya.
Jika jawaban tidak ditemukan secara eksplisit dalam konteks,
jawab dengan:
"Tidak tersedia atau tidak diatur dalam daftar regulasi yang ditentukan."

Konteks (hasil pencarian regulasi terkait):
{context}

Pertanyaan:
{question}

Jawaban:
"""
    return template



# ====================================
# 🤖 RAG + GEMINI ANSWER
# ====================================
def user_input(user_question):
    # 1. Load FAISS
    db = load_vector_store()

    # 2. Ambil chunk paling relevan
    docs = db.similarity_search(user_question, k=5)
    context = "\n\n".join([d.page_content for d in docs])

    # 3. Build prompt
    prompt = build_gemini_prompt(user_question, context)

    # 4. Call Gemini REST API
    url = f"{GEMINI_ENDPOINT}?key={GEMINI_API_KEY}"
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ]
    }

    try:
        response = requests.post(url, json=payload)
        data = response.json()

        text = data["candidates"][0]["content"]["parts"][0]["text"]
        return {"output_text": text}

    except Exception as e:
        return {"output_text": f"❌ Error from Gemini API: {e}"}



# ====================================
# 💬 CHAT HANDLING
# ====================================
def clear_chat_history():
    st.session_state.messages = [
        {"role": "assistant", "content": "Tanyakan apapun terkait regulasi pengadaan barang/jasa."}
    ]



# ====================================
# 🚀 MAIN STREAMLIT APP
# ====================================
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

                    st.success("✅ Dokumen berhasil diproses dan ditambahkan.")
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

    # User question
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
