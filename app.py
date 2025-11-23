import os
import json
import base64
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

# Gemini
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

# GitHub
from github import Github

# Konfigurasi API Key Gemini
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# =========================
# 📁 CONFIG & STATE
# =========================
STATE_FILE = "app_state.json"
FAISS_FILE = "faiss_index"

GITHUB_REPO = "username/repo-name"  # Ganti dengan repo GitHub kamu
GITHUB_BRANCH = "main"

# =========================
# 🔹 GitHub Helper Functions
# =========================
def commit_file_to_github(local_path, commit_message):
    token = st.secrets["GITHUB_TOKEN"]
    g = Github(token)
    repo = g.get_repo(GITHUB_REPO)

    # Baca file
    with open(local_path, "rb") as f:
        content = f.read()

    # Encode jika binary
    try:
        content_str = content.decode("utf-8")  # json/text
    except:
        content_str = base64.b64encode(content).decode("utf-8")  # binary FAISS

    try:
        git_file = repo.get_contents(local_path, ref=GITHUB_BRANCH)
        repo.update_file(path=git_file.path, message=commit_message,
                         content=content_str, sha=git_file.sha, branch=GITHUB_BRANCH)
    except:
        repo.create_file(path=local_path, message=commit_message,
                         content=content_str, branch=GITHUB_BRANCH)


def load_file_from_github(local_path):
    token = st.secrets["GITHUB_TOKEN"]
    g = Github(token)
    repo = g.get_repo(GITHUB_REPO)
    git_file = repo.get_contents(local_path, ref=GITHUB_BRANCH)
    content_str = git_file.decoded_content

    # Simpan ke lokal
    try:
        content_bytes = content_str
        with open(local_path, "wb") as f:
            f.write(content_bytes)
    except:
        # Jika binary di base64
        content_bytes = base64.b64decode(content_str)
        with open(local_path, "wb") as f:
            f.write(content_bytes)

# =========================
# 🔹 State Management
# =========================
def save_state(file_names):
    with open(STATE_FILE, "w") as f:
        json.dump({"processed_files": file_names}, f)
    commit_file_to_github(STATE_FILE, "Update app state")
    commit_file_to_github(FAISS_FILE, "Update FAISS index")


def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    # Jika tidak ada, coba load dari GitHub
    try:
        load_file_from_github(STATE_FILE)
        return load_state()
    except:
        return {"processed_files": []}

# =========================
# 🔹 PDF Processing
# =========================
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text

def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
    return splitter.split_text(text)

# =========================
# 🔹 Vector Store
# =========================
@st.cache_resource(show_spinner=False)
def create_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="text-embedding-004",
        api_key=st.secrets["GEMINI_API_KEY"]
    )
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local(FAISS_FILE)
    return vector_store

@st.cache_resource(show_spinner=False)
def load_vector_store():
    if not os.path.exists(FAISS_FILE):
        try:
            load_file_from_github(FAISS_FILE)
        except:
            st.warning("FAISS index belum tersedia. Silakan upload PDF dulu.")
            return None

    embeddings = GoogleGenerativeAIEmbeddings(
        model="text-embedding-004",
        api_key=st.secrets["GEMINI_API_KEY"]
    )
    return FAISS.load_local(FAISS_FILE, embeddings, allow_dangerous_deserialization=True)

# =========================
# 🔹 QA Chain
# =========================
@st.cache_resource(show_spinner=False)
def get_conversational_chain():
    prompt_template = """
Answer the question using the provided context as accurately as possible.
If the answer is not found in the context, simply reply:
"Answer is not available in the context."

Context:
{context}

Question:
{question}

Answer:
"""
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)

# =========================
# 🔹 Chat Handling
# =========================
def clear_chat_history():
    st.session_state.messages = [
        {"role": "assistant", "content": "Tanyakan apapun terkait regulasi pengadaan barang/jasa."}
    ]

def user_input(user_question):
    db = load_vector_store()
    if not db:
        return {"output_text": "FAISS index belum tersedia. Silakan upload PDF terlebih dahulu."}
    docs = db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response

# =========================
# 🔹 Streamlit App
# =========================
def main():
    st.set_page_config(page_title="ReguBot | Regulasi ChatBot", page_icon="🤖")
    st.title("Selamat datang di ReguBot!")

    state = load_state()

    with st.sidebar:
        st.header("📂 Unggah & Proses Dokumen")
        pdf_docs = st.file_uploader("Unggah File PDF", accept_multiple_files=True, type=["pdf"])

        if st.button("Submit & Process"):
            if pdf_docs:
                uploaded_names = [pdf.name for pdf in pdf_docs]
                with st.spinner("🔄 Membaca dan memproses dokumen..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    create_vector_store(text_chunks)
                    save_state(uploaded_names)
                    st.success("✅ Dokumen berhasil diproses dan disimpan ke GitHub.")
            else:
                st.warning("⚠️ Tolong unggah minimal satu dokumen.")

        st.button("🧹 Bersihkan Jejak Digital", on_click=clear_chat_history)

        if state["processed_files"]:
            st.markdown("### 📚 Data Dokumen:")
            for f in state["processed_files"]:
                st.write(f"• {f}")

    if "messages" not in st.session_state:
        clear_chat_history()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

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
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                except Exception as e:
                    st.error(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
