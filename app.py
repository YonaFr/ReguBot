import os
import json
import requests
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()



# =========================
# 🔵 GOOGLE GEMINI (REST API)
# =========================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_ENDPOINT = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-2.5-flash:generateContent"
)



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
# 🤖 PROMPT BUILDER
# =========================
def build_gemini_prompt(question):
    template = """
Anda adalah asisten yang sangat teliti dalam menjawab pertanyaan terkait regulasi pengadaan barang/jasa.
Ikuti aturan berikut secara ketat:

1. Gunakan informasi dari regulasi terlebih dahulu.
   Jika jawaban untuk pertanyaan dapat ditemukan secara jelas: implisit atau eksplisit dalam regulasi, jawablah hanya berdasarkan regulasi tersebut.

2. Jika regulasi tidak cukup, Anda boleh menggunakan sumber informasi eksternal untuk menjawab pertanyaan, tetapi WAJIB mencantumkan bahwa sumber utama adalah regulasi, dan sertakan referensi/cuplikan regulasi terkait.

3. Jika pertanyaan tidak ada kaitannya dengan regulasi, jawab dengan informasi yang relevan dari sumber eksternal, tetap sertakan referensi regulasi pada bagian yang mendukung.

4. Jangan membuat asumsi atau menjawab di luar konteks regulasi, kecuali memang diminta oleh poin 2 dan 3.

---

Daftar Regulasi:
1. UU No 3 Tahun 2024
2. UU No 6 Tahun 2014
3. PP No 11 Tahun 2019
4. PP Nomor 8 Tahun 2016
5. Peraturan Presiden Nomor 46 Tahun 2025
6. Peraturan Presiden Nomor 12 Tahun 2021
7. Peraturan Presiden Nomor 16 Tahun 2018
8. Permendagri No 111 Tahun 2014
9. Permendagri No 112 Tahun 2014
10. Permendagri No 20 Tahun 2018
11. Permendagri No 114 Tahun 2014
12. Permendesa No. 2 Tahun 2024
13. Peraturan Lembaga Nomor 2 Tahun 2025
14. Peraturan Lembaga Nomor 12 Tahun 2019
15. Keputusan Deputi I Nomor 1 Tahun 2025
16. Keputusan Deputi I Nomor 2 Tahun 2024
17. Surat Edaran Kepala LKPP Nomor 1 Tahun 2025
18. Perbup No 44 Tahun 2020

Pertanyaan:
{question}

Jawaban:
- Berikan jawaban secara jelas dan lengkap.
- Sertakan bagian “Sumber Regulasi” di akhir jawaban.

Contoh format jawaban:

Jawaban:
[Isi jawaban yang relevan]

Sumber Regulasi:
[Cuplikan atau referensi regulasi yang relevan]
"""
    return template.format(question=question)



# =========================
# 🤖 GEMINI REST CLIENT
# =========================
def call_gemini_rest(prompt):
    """
    Memanggil Gemini via REST API.
    Sama seperti AppScript Anda.
    """
    url = f"{GEMINI_ENDPOINT}?key={GEMINI_API_KEY}"

    payload = {
        "contents": [
            {
                "parts": [{"text": prompt}]
            }
        ]
    }

    try:
        response = requests.post(url, json=payload)
        data = response.json()

        return (
            data["candidates"][0]["content"]["parts"][0]["text"]
        )

    except Exception as e:
        return f"❌ Error from Gemini API: {e}"



# =========================
# 💬 CHAT HANDLING
# =========================
def user_input(user_question):
    """
    Final QA flow:
    - Build prompt
    - Kirim ke Gemini REST
    - Return text
    """
    prompt = build_gemini_prompt(user_question)
    result = call_gemini_rest(prompt)
    return {"output_text": result}


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

                    # Tidak mengubah fitur upload — tetap sama
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
