import os
import json
import base64               # ← Added
import requests             # ← Added
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate


# ======================================================
# 🔵 FUNGSI UPLOAD KE GITHUB (MINIMAL, WAJIB UTK SOLUSI C)
# ======================================================
def upload_to_github(local_path, github_path):
    """Upload a local file to GitHub repo."""
    with open(local_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    url = f"https://api.github.com/repos/{st.secrets['github']['repo']}/contents/{github_path}"
    headers = {"Authorization": f"token {st.secrets['github']['token']}"}

    # Check if exists → needed for updating file (SHA)
    get_res = requests.get(url, headers=headers)
    sha = get_res.json().get("sha") if get_res.status_code == 200 else None

    payload = {
        "message": f"update {github_path}",
        "content": encoded,
        "branch": st.secrets["github"]["branch"]
    }
    if sha:
        payload["sha"] = sha

    requests.put(url, json=payload, headers=headers)



# =========================
# 📁 FILE STATE MANAGEMENT
# =========================
STATE_FILE = "app_state.json"

def save_state(file_names):
    """Save state locally + upload to GitHub."""
    with open(STATE_FILE, "w") as f:
        json.dump({"processed_files": file_names}, f)

    # ← MODIFIKASI SOLUSI C
    upload_to_github(STATE_FILE, "app_state.json")


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
            text += page.extract_text() or ""
    return text


def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    return splitter.split_text(text)


# =========================
# 🧠 VECTOR STORE CREATION
# =========================
@st.cache_resource(show_spinner=False)
def create_vector_store(chunks):
    embeddings = OllamaEmbeddings(model="llama3.2")
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)

    vector_store.save_local("faiss_index")  # tetap ada

    # ← MODIFIKASI SOLUSI C
    upload_to_github("faiss_index/index.faiss", "faiss_index/index.faiss")
    upload_to_github("faiss_index/index.pkl", "faiss_index/index.pkl")

    return vector_store


@st.cache_resource(show_spinner=False)
def load_vector_store():
    embeddings = OllamaEmbeddings(model="llama3.2")
    return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)


# =========================
# 🤖 QA CHAIN (LLM + PROMPT)
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

    model = ChatOllama(model="llama3.2", temperature=0.3)
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    return load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)


# =========================
# 💬 CHAT HANDLING
# =========================
def clear_chat_history():
    st.session_state.messages = [
        {"role": "assistant", "content": "Tanyakan apapun terkait regulasi pengadaan barang/jasa."}
    ]


def user_input(user_question):
    db = load_vector_store()
    docs = db.similarity_search(user_question)

    chain = get_conversational_chain()
    return chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)


# =========================
# 🚀 MAIN STREAMLIT APP
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
                    st.success("✅ Dokumen berhasil diproses dan disimpan.")
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
                    st.session_state.messages.append(
                        {"role": "assistant", "content": full_response}
                    )
                except Exception as e:
                    st.error(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
