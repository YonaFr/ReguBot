import os
import json
import requests
import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import re

load_dotenv()

# =========================
# ⚠️ HARD-CODE EXCLUDED REGULATIONS
# =========================
EXCLUDED_REGULATIONS = []

# =========================
# 📘 STRUKTUR PASAL
# =========================
REGULATION_STRUCTURE = {
    "UU No 3 Tahun 2024": list(range(1, 133)),
    "UU No 6 Tahun 2014": list(range(1, 133)),
    "PP No 11 Tahun 2019": list(range(1, 152)),
    "PP Nomor 8 Tahun 2016": list(range(1, 29)),
    "Permendagri No 111 Tahun 2014": list(range(1, 34)),
    "Permendagri No 112 Tahun 2014": list(range(1, 51)),
    "Permendagri No 20 Tahun 2018": list(range(1, 107)),
    "Permendagri No 114 Tahun 2014": list(range(1, 45)),
    "Permendesa No. 2 Tahun 2024": list(range(1, 26)),
    "Peraturan Presiden Nomor 12 Tahun 2021": list(range(1, 138)),
    "Peraturan Presiden Nomor 16 Tahun 2018": list(range(1, 133)),
    "Peraturan Presiden Nomor 46 Tahun 2025": list(range(1, 50)),
    "Peraturan Lembaga Nomor 2 Tahun 2025": list(range(1, 25)),
    "Peraturan Lembaga Nomor 12 Tahun 2019": list(range(1, 20)),
    "Keputusan Deputi I Nomor 1 Tahun 2025": list(range(1, 10)),
    "Keputusan Deputi I Nomor 2 Tahun 2024": list(range(1, 10)),
    "Perbup No 44 Tahun 2020": list(range(1, 40)),
    "Surat Edaran Kepala LKPP Nomor 1 Tahun 2025": []
}

# =========================
# 🧠 DETEKSI PERTANYAAN REGULASI
# =========================
def is_regulation_question(question: str) -> bool:
    q = question.lower()
    keywords = [
        "pasal", "ayat", "ketentuan", "diatur", "berdasarkan",
        "menurut", "regulasi", "peraturan", "undang-undang",
        "pengadaan", "pbj", "ppk", "pokja", "ulp",
        "lpse", "spse", "tender", "lelang",
        "kontrak", "e-purchasing",
        "desa", "kepala desa", "apbdes", "dana desa",
        "sanksi", "wewenang", "tugas", "larangan"
    ]
    return any(k in q for k in keywords)

# =========================
# 🔵 GOOGLE GEMINI
# =========================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_ENDPOINT = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-2.5-flash:generateContent"
)

def call_gemini(prompt):
    url = f"{GEMINI_ENDPOINT}?key={GEMINI_API_KEY}"
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    response = requests.post(url, json=payload)
    return response.json()["candidates"][0]["content"]["parts"][0]["text"]

# =========================
# 🤖 PROMPT BUILDER (ASLI DIPERTAHANKAN)
# =========================
def build_gemini_prompt(question):
    return f"""
Anda adalah asisten yang sangat teliti dalam menjawab pertanyaan regulasi PBJ.

FORMAT WAJIB:

Jawaban:
...

Sumber Regulasi:
...

Pertanyaan:
{question}
"""

# =========================
# 💬 CHAT HANDLING
# =========================
def user_input(user_question):
    prompt = build_gemini_prompt(user_question)
    return call_gemini(prompt)

def clear_chat_history():
    st.session_state.messages = [
        {"role": "assistant", "content": "Tanyakan apapun terkait regulasi pengadaan barang/jasa."}
    ]

# =========================
# 🚀 MAIN APP (UI TIDAK DIUBAH)
# =========================
def main():
    st.set_page_config(
        page_title="ReguBot | Regulasi ChatBot",
        page_icon="https://raw.githubusercontent.com/YonaFr/YonaFr.GitHub.IO/main/PBJ.ico"
    )

    st.title("Selamat datang di ReguBot!")

    # ---------- SESSION STATE ----------
    if "messages" not in st.session_state:
        clear_chat_history()

    if "pending_question" not in st.session_state:
        st.session_state.pending_question = None

    if "awaiting_confirmation" not in st.session_state:
        st.session_state.awaiting_confirmation = False

    # ---------- SIDEBAR (ASLI) ----------
    with st.sidebar:
        st.markdown(
            """
            <div style="text-align:center;">
                <img src="https://raw.githubusercontent.com/YonaFr/ReguBot/main/PBJ.png" width="140">
            </div>
            """,
            unsafe_allow_html=True
        )

        st.header("📂 Unggah & Proses Dokumen")
        st.file_uploader("Unggah File PDF", accept_multiple_files=True, type=["pdf"])
        st.button("🧹 Bersihkan Jejak Digital", on_click=clear_chat_history)

    # ---------- CHAT HISTORY ----------
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # ---------- CHAT INPUT ----------
    if prompt := st.chat_input("Ketik di sini..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        if is_regulation_question(prompt):
            with st.chat_message("assistant"):
                with st.spinner("🤔 Thinking..."):
                    answer = user_input(prompt)
                    st.markdown(answer)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer}
                    )
        else:
            st.session_state.pending_question = prompt
            st.session_state.awaiting_confirmation = True

    # ---------- KONFIRMASI ----------
    if st.session_state.awaiting_confirmation:
        st.warning(
            "⚠️ Regulasi tidak ditemukan dalam daftar. "
            "Apakah Anda ingin melanjutkan dengan sumber eksternal?"
        )

        col1, col2 = st.columns(2)

        with col1:
            if st.button("✅ Ya, lanjutkan"):
                q = st.session_state.pending_question
                with st.chat_message("assistant"):
                    with st.spinner("🤔 Thinking..."):
                        answer = user_input(q)
                        st.markdown(answer)
                        st.session_state.messages.append(
                            {"role": "assistant", "content": answer}
                        )
                st.session_state.awaiting_confirmation = False
                st.session_state.pending_question = None

        with col2:
            if st.button("❌ Tidak"):
                st.info("Pertanyaan dibatalkan.")
                st.session_state.awaiting_confirmation = False
                st.session_state.pending_question = None

if __name__ == "__main__":
    main()
