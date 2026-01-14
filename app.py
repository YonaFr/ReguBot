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
# 🔍 DETEKSI REGULASI DI PERTANYAAN
# =========================
def detect_related_regulation(question: str):
    q = question.lower()
    return [r for r in REGULATION_STRUCTURE if r.lower() in q]

# =========================
# 🤖 GEMINI SETUP
# =========================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_ENDPOINT = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-3-flash-preview:generateContent"
)

def build_gemini_prompt(question):
    return f"""
Anda adalah asisten ahli regulasi pengadaan barang/jasa.

Aturan:
- Jawab berdasarkan regulasi jika tersedia.
- Jangan mengarang pasal.
- Jika pasal tidak pasti, tulis tanpa pasal.

Format WAJIB:

Jawaban:
...

Sumber Regulasi:
...

Pertanyaan:
{question}
"""

def call_gemini(prompt):
    url = f"{GEMINI_ENDPOINT}?key={GEMINI_API_KEY}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}]
    }
    r = requests.post(url, json=payload)
    return r.json()["candidates"][0]["content"]["parts"][0]["text"]

# =========================
# 💬 STREAMLIT APP
# =========================
def main():
    st.set_page_config(page_title="ReguBot")
    st.title("📘 ReguBot")

    # --- STATE ---
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Silakan ajukan pertanyaan regulasi."}
        ]

    if "pending_question" not in st.session_state:
        st.session_state.pending_question = None

    if "awaiting_confirmation" not in st.session_state:
        st.session_state.awaiting_confirmation = False

    # --- CHAT HISTORY ---
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # --- INPUT ---
    if prompt := st.chat_input("Ketik pertanyaan..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        detected = detect_related_regulation(prompt)

        if detected:
            with st.chat_message("assistant"):
                with st.spinner("🤔 Thinking..."):
                    answer = call_gemini(build_gemini_prompt(prompt))
                    st.markdown(answer)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer}
                    )
        else:
            st.session_state.pending_question = prompt
            st.session_state.awaiting_confirmation = True

    # --- CONFIRMATION ---
    if st.session_state.awaiting_confirmation:
        st.warning(
            "⚠️ Regulasi tidak ditemukan dalam daftar. "
            "Apakah Anda ingin melanjutkan dengan sumber eksternal?"
        )

        col1, col2 = st.columns(2)

        with col1:
            if st.button("✅ Ya"):
                q = st.session_state.pending_question
                with st.chat_message("assistant"):
                    with st.spinner("🤔 Thinking..."):
                        answer = call_gemini(build_gemini_prompt(q))
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
