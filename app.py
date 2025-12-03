import os
import json
import requests
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import re   # >>> ADDED

load_dotenv()


# =========================
# ⚠️ HARD-CODE EXCLUDED REGULATIONS
# =========================
EXCLUDED_REGULATIONS = [
    # Contoh:
    # "UU No 6 Tahun 2014",
    # "PP No 11 Tahun 2019",
]


# =========================
# 📘 STRUKTUR PASAL (UNTUK VALIDASI) 
# =========================
REGULATION_STRUCTURE = {    
    "UU No 3 Tahun 2024": list(range(1, 133)),   # UU Perubahan, mencakup 132 Pasal total
    "UU No 6 Tahun 2014": list(range(1, 133)),
    "PP No 11 Tahun 2019": list(range(1, 152)),  # Mengacu pada PP 43/2014 (regulasi utama yang diubah)
    "PP Nomor 8 Tahun 2016": list(range(1, 29)), # Mengacu pada PP 60/2014 (regulasi utama yang diubah)
    "Permendagri No 111 Tahun 2014": list(range(1, 34)),
    "Permendagri No 112 Tahun 2014": list(range(1, 51)),
    "Permendagri No 20 Tahun 2018": list(range(1, 107)),
    "Permendagri No 114 Tahun 2014": list(range(1, 45)),
    "Permendesa No. 2 Tahun 2024": list(range(1, 26)), # Batang Tubuh
    
    # Regulasi Pengadaan Barang/Jasa (PBJ)
    "Peraturan Presiden Nomor 12 Tahun 2021": list(range(1, 138)), # Total Pasal Perpres 16/2018 setelah diubah
    "Peraturan Presiden Nomor 16 Tahun 2018": list(range(1, 133)),
    
    # Regulasi yang sangat spesifik atau belum tersedia publik
    "Peraturan Presiden Nomor 46 Tahun 2025": list(range(1, 50)),
    "Peraturan Lembaga Nomor 2 Tahun 2025": list(range(1, 25)),
    "Peraturan Lembaga Nomor 12 Tahun 2019": list(range(1, 20)),
    "Keputusan Deputi I Nomor 1 Tahun 2025": list(range(1, 10)),
    "Keputusan Deputi I Nomor 2 Tahun 2024": list(range(1, 10)),
    "Perbup No 44 Tahun 2020": list(range(1, 40)),
    
    # Regulasi tanpa pasal bernomor
    "Surat Edaran Kepala LKPP Nomor 1 Tahun 2025": list(range(1, 1))
}


# =========================
# 🔍 VALIDASI CITATION 
# =========================
def validate_citation(response_text):   
    pattern = r"(?P<reg>[A-Za-z0-9 ./-]+?),?\s*Pasal\s*(?P<pasal>\d+)"

    def replacer(match):
        reg = match.group("reg").strip()
        pasal = int(match.group("pasal"))

        # Jika regulasi dikecualikan → tandai
        if reg in EXCLUDED_REGULATIONS:
            return f"{reg} (dikecualikan dari penggunaan)"

        # Jika tidak ada struktur pasal → tidak bisa diverifikasi
        if reg not in REGULATION_STRUCTURE:
            return f"{reg} (pasal tidak dapat diverifikasi)"

        valid_list = REGULATION_STRUCTURE[reg]

        # Untuk SE tanpa pasal
        if valid_list == [None]:
            return f"{reg}"

        # Jika pasal tidak valid → tandai
        if pasal not in valid_list:
            return f"{reg} (pasal tidak dapat diverifikasi)"

        return f"{reg}, Pasal {pasal}"

    cleaned = re.sub(pattern, replacer, response_text)

    # Hapus penyebutan regulasi tanpa pasal jika dikecualikan
    for ex in EXCLUDED_REGULATIONS:
        cleaned = cleaned.replace(ex, f"{ex} (dikecualikan dari penggunaan)")

    return cleaned

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
# 🤖 PROMPT BUILDER (DIREFAKTOR)
# =========================
def build_gemini_prompt(question):
    """
    Build prompt dengan regulasi yang dikecualikan secara hardcode.
    Gemini boleh improvisasi, tetapi tidak boleh menyertakan regulasi yang dikecualikan.
    """

    all_regulations = [
        "UU No 3 Tahun 2024",
        "UU No 6 Tahun 2014",
        "PP No 11 Tahun 2019",
        "PP Nomor 8 Tahun 2016",
        "Peraturan Presiden Nomor 46 Tahun 2025",
        "Peraturan Presiden Nomor 12 Tahun 2021",
        "Peraturan Presiden Nomor 16 Tahun 2018",
        "Permendagri No 111 Tahun 2014",
        "Permendagri No 112 Tahun 2014",
        "Permendagri No 20 Tahun 2018",
        "Permendagri No 114 Tahun 2014",
        "Permendesa No. 2 Tahun 2024",
        "Peraturan Lembaga Nomor 2 Tahun 2025",
        "Peraturan Lembaga Nomor 12 Tahun 2019",
        "Keputusan Deputi I Nomor 1 Tahun 2025",
        "Keputusan Deputi I Nomor 2 Tahun 2024",
        "Surat Edaran Kepala LKPP Nomor 1 Tahun 2025",
        "Perbup No 44 Tahun 2020"
    ]

    included_regulations = [r for r in all_regulations if r not in EXCLUDED_REGULATIONS]
    regulations_text = "\n".join(f"{i+1}. {r}" for i, r in enumerate(included_regulations))
    excluded_text = ", ".join(EXCLUDED_REGULATIONS) if EXCLUDED_REGULATIONS else "Tidak ada"

    template = f"""
Anda adalah asisten yang sangat teliti dalam menjawab pertanyaan terkait regulasi pengadaan barang/jasa.
Ikuti aturan berikut secara ketat:

1. Gunakan informasi dari regulasi terlebih dahulu.
   Jika jawaban untuk pertanyaan dapat ditemukan secara jelas, implisit atau eksplisit dalam regulasi, jawablah hanya berdasarkan regulasi tersebut.

2. Dilarang menggunakan atau menyebut regulasi yang dikecualikan: {excluded_text}

3. Jika regulasi tidak cukup, Anda boleh menggunakan sumber informasi eksternal untuk menjawab pertanyaan.
   Namun tetap WAJIB menjelaskan bahwa jawaban utama berasal dari regulasi yang tersedia.
   Tetap tidak boleh menyertakan regulasi yang dikecualikan.

4. Jika pertanyaan tidak ada kaitannya dengan regulasi, jawab dengan informasi yang relevan dari sumber eksternal.
   Namun jika ada bagian yang masih dapat dikaitkan dengan regulasi, sebutkan regulasi tersebut.

5. Jangan membuat asumsi atau mengarang ketentuan regulasi.
   Jika Anda tidak yakin dengan nomor pasal atau ayat, JANGAN mengarang.
   Anda boleh tetap memberikan jawaban yang benar berdasarkan substansi regulasi tanpa menyebut nomor pasal.


---

Daftar Regulasi:
{regulations_text}

Pertanyaan:
{question}

Instruksi Jawaban:
- Jawab dengan jelas dan lengkap.
- Jika Anda mengetahui pasal atau ayat secara pasti, sebutkan.
- Jika Anda tidak mengetahui pasal secara pasti, tulis tanpa pasal.
- Tetap wajib mencantumkan nama regulasi yang digunakan.
- Dilarang menuliskan pasal yang tidak valid.
- Dilarang menyebut regulasi yang dikecualikan.

Format jawaban:

Jawaban:
[Isi jawaban yang relevan]

Sumber Regulasi:
[Regulasi yang digunakan, opsional beserta pasal jika diketahui dengan pasti]

Contoh jawaban:

Jawaban: Jabatan Kepala Desa adalah selama 8 (delapan) tahun terhitung sejak tanggal pelantikan. Kepala Desa dapat menjabat paling banyak 2 (dua) kali masa jabatan, baik secara berturut-turut maupun tidak secara berturut-turut.

Sumber Regulasi: Undang-Undang Republik Indonesia Nomor 3 Tahun 2024
"""

    return template


# =========================
# 🤖 GEMINI REST CLIENT
# =========================
def call_gemini_rest(prompt):
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
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        return f"❌ Error from Gemini API: {e}"



# =========================
# 💬 CHAT HANDLING
# =========================
def user_input(user_question):
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

    # Sidebar
    with st.sidebar:

        # =========================
        # 🖼️ LOGO DI ATAS HEADER
        # =========================
        st.markdown(
            """
            <div style="text-align:center;">
                <img src="https://raw.githubusercontent.com/YonaFr/ReguBot/main/PBJ.png" width="140">
            </div>
            """,
            unsafe_allow_html=True
        )

        st.header("📂 Unggah & Proses Dokumen")
        pdf_docs = st.file_uploader("Unggah File PDF", accept_multiple_files=True, type=["pdf"])

        if st.button("Submit & Process"):
            if pdf_docs:
                uploaded_names = [pdf.name for pdf in pdf_docs]
                with st.spinner("🔄 Membaca dan memproses dokumen..."):
                    raw_text = get_pdf_text(pdf_docs)
                    chunks = get_text_chunks(raw_text)
                    save_state(uploaded_names)
                    st.success("✅ Dokumen berhasil diproses.")
            else:
                st.warning("⚠️ Tolong unggah minimal satu dokumen.")

        st.button("🧹 Bersihkan Jejak Digital", on_click=clear_chat_history)

        if state["processed_files"]:
            st.markdown("### 📚 Data Dokumen:")
            for f in state["processed_files"]:
                st.write(f"• " + f)

        st.markdown("---")

        st.markdown(
            """
            <div style="text-align:center; font-size:12px; color:#777;">
                 <img src="https://mirrors.creativecommons.org/presskit/icons/cc.svg" alt="" style="max-width: 1em;max-height:1em;margin-left: .2em;"><img src="https://mirrors.creativecommons.org/presskit/icons/by.svg" alt="" style="max-width: 1em;max-height:1em;margin-left: .2em;"><img src="https://mirrors.creativecommons.org/presskit/icons/nc.svg" alt="" style="max-width: 1em;max-height:1em;margin-left: .2em;"><img src="https://mirrors.creativecommons.org/presskit/icons/sa.svg" alt="" style="max-width: 1em;max-height:1em;margin-left: .2em;"><br>
                2025. Yona Friantina.<br>
                Some rights reserved.<br>
                <span style="font-size:11px;">Build with Streamlit</span>
            </div>
            """,
            unsafe_allow_html=True
        )


    # Init chat history
    if "messages" not in st.session_state:
        clear_chat_history()

    # Render chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
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
