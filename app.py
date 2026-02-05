import os
import json
import requests
import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import re
from enum import Enum
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

load_dotenv()

# =========================
# 📋 CONFIGURATION & CONSTANTS
# =========================

class QueryContext(Enum):
    """Enum untuk menentukan konteks pertanyaan"""
    UPLOADED_REGULATIONS = "uploaded"      # Ada regulasi yang di-upload
    PBJ_NO_UPLOAD = "pbj_no_upload"       # Konteks PBJ tapi tidak ada upload
    NON_PBJ = "non_pbj"                    # Bukan konteks PBJ

@dataclass
class ResponseData:
    """Data class untuk response yang terstruktur"""
    output_text: str
    note: str = ""
    warning: str = ""
    source_type: str = "regulation"  # regulation, external, mixed, none

class Config:
    """Konfigurasi aplikasi yang terpusat"""
    STATE_FILE = "app_state.json"
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
    
    # Excluded regulations (dapat dikustomisasi)
    EXCLUDED_REGULATIONS = [
        # Contoh:
        # "UU No 6 Tahun 2014",
        # "PP No 11 Tahun 2019",
    ]
    
    # Struktur pasal untuk validasi
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
    
    # Keywords untuk deteksi konteks PBJ
    PBJ_KEYWORDS = [
        "pengadaan", "barang", "jasa", "lelang", "tender", 
        "kontrak", "penyedia", "pokja", "ukpbj", "pejabat pengadaan",
        "hps", "aanwijzing", "sanggah", "pascakualifikasi", "prakualifikasi"
    ]
    
    # Fluent Design Colors
    FLUENT_COLORS = {
        "primary": "#0078D4",
        "primary_hover": "#106EBE",
        "secondary": "#005A9E",
        "accent": "#2B88D8",
        "success": "#107C10",
        "warning": "#FF8C00",
        "error": "#E81123",
        "background": "#F3F2F1",
        "surface": "#FFFFFF",
        "text_primary": "#323130",
        "text_secondary": "#605E5C",
        "border": "#EDEBE9"
    }

# =========================
# 🔧 UTILITY MODULES
# =========================

class StateManager:
    """Mengelola state aplikasi (file uploads, etc.)"""
    
    @staticmethod
    def save_state(new_file_names: List[str]) -> None:
        """Simpan state baru ke file"""
        state = StateManager.load_state()
        existing = set(state.get("processed_files", []))
        updated = list(existing.union(new_file_names))
        with open(Config.STATE_FILE, "w", encoding="utf-8") as f:
            json.dump({"processed_files": updated}, f, ensure_ascii=False, indent=2)
    
    @staticmethod
    def load_state() -> Dict:
        """Load state dari file"""
        if os.path.exists(Config.STATE_FILE):
            with open(Config.STATE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"processed_files": []}
    
    @staticmethod
    def clear_state() -> None:
        """Hapus semua state"""
        if os.path.exists(Config.STATE_FILE):
            os.remove(Config.STATE_FILE)

class PDFProcessor:
    """Mengelola pemrosesan PDF"""
    
    @staticmethod
    def extract_text(pdf_docs: List) -> str:
        """Ekstrak teks dari PDF"""
        text = ""
        for pdf in pdf_docs:
            try:
                pdf_reader = PdfReader(pdf)
                for page in pdf_reader.pages:
                    extracted = page.extract_text() or ""
                    text += extracted
            except Exception as e:
                st.error(f"❌ Error membaca {pdf.name}: {e}")
        return text
    
    @staticmethod
    def create_chunks(text: str, chunk_size: int = 2000, chunk_overlap: int = 200) -> List[str]:
        """Buat chunks dari teks"""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap
        )
        return splitter.split_text(text)

class CitationValidator:
    """Validasi sitasi regulasi"""
    
    @staticmethod
    def validate(response_text: str) -> str:
        """Validasi dan bersihkan sitasi dalam response"""
        pattern = r"(?P<reg>(?:UU|PP|Permendagri|Permendesa|Peraturan Presiden|Peraturan Lembaga|Keputusan Deputi I|Perbup|Surat Edaran Kepala LKPP)[^,]*),\s*Pasal\s*(?P<pasal>\d+)"
        
        def replacer(match):
            reg = match.group("reg").strip()
            pasal = int(match.group("pasal"))
            
            if reg in Config.EXCLUDED_REGULATIONS:
                return f"{reg} (dikecualikan dari penggunaan)"
            if reg not in Config.REGULATION_STRUCTURE:
                return f"{reg} (pasal tidak dapat diverifikasi)"
            
            valid_list = Config.REGULATION_STRUCTURE[reg]
            if not valid_list:
                return f"{reg}"
            if pasal not in valid_list:
                return f"{reg} (pasal tidak dapat diverifikasi)"
            
            return f"{reg}, Pasal {pasal}"
        
        cleaned = re.sub(pattern, replacer, response_text)
        
        # Tandai regulasi yang dikecualikan
        for ex in Config.EXCLUDED_REGULATIONS:
            cleaned = cleaned.replace(ex, f"{ex} (dikecualikan dari penggunaan)")
        
        return cleaned

# =========================
# 🤖 AI SERVICE MODULE
# =========================

class GeminiService:
    """Service untuk berkomunikasi dengan Gemini API"""
    
    @staticmethod
    def build_prompt(question: str, has_uploaded: bool = True) -> str:
        """Build prompt untuk Gemini berdasarkan kondisi"""
        all_regulations = list(Config.REGULATION_STRUCTURE.keys())
        included_regulations = [r for r in all_regulations if r not in Config.EXCLUDED_REGULATIONS]
        regulations_text = "\n".join(f"{i+1}. {r}" for i, r in enumerate(included_regulations))
        excluded_text = ", ".join(Config.EXCLUDED_REGULATIONS) if Config.EXCLUDED_REGULATIONS else "Tidak ada"
        
        # Template berbeda berdasarkan status upload
        if has_uploaded:
            validation_status = "REGULASI SUDAH DIUPLOAD - Validitas jawaban terjamin"
            reliability_note = "Jawaban berikut berdasarkan regulasi yang telah diupload dan diverifikasi."
        else:
            validation_status = "⚠️ REGULASI BELUM DIUPLOAD - Validitas jawaban TIDAK terjamin"
            reliability_note = "PERINGATAN: Regulasi belum diupload. Jawaban berikut mungkin tidak akurat atau tidak dapat diverifikasi."
        
        template = f"""
Anda adalah asisten yang sangat teliti dalam menjawab pertanyaan terkait regulasi pengadaan barang/jasa.

STATUS: {validation_status}

Ikuti aturan berikut secara ketat:

1. Gunakan informasi dari regulasi terlebih dahulu.
   Jika jawaban untuk pertanyaan dapat ditemukan secara jelas, implisit atau eksplisit dalam regulasi, jawablah hanya berdasarkan regulasi tersebut.

2. Dilarang menggunakan atau menyebut regulasi yang dikecualikan: {excluded_text}

3. Jika regulasi tidak cukup, Anda boleh menggunakan sumber informasi eksternal untuk menjawab pertanyaan.
   Namun tetap WAJIB menjelaskan bahwa jawaban utama berasal dari regulasi yang tersedia.
   Tetap tidak boleh menyertakan regulasi yang dikecualikan.

4. Jangan membuat asumsi atau mengarang ketentuan regulasi.
   Jika Anda tidak yakin dengan nomor pasal atau ayat, JANGAN mengarang.
   Anda boleh tetap memberikan jawaban yang benar berdasarkan substansi regulasi tanpa menyebut nomor pasal.

---

Daftar Regulasi:
{regulations_text}

Pertanyaan:
{question}

Instruksi Jawaban:
- Awali dengan: "{reliability_note}"
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
"""
        return template
    
    @staticmethod
    def call_api(prompt: str) -> str:
        """Panggil Gemini API"""
        url = f"{Config.GEMINI_ENDPOINT}?key={Config.GEMINI_API_KEY}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}]
        }
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except requests.exceptions.RequestException as e:
            return f"❌ Error dari Gemini API: {e}"
        except (KeyError, IndexError) as e:
            return f"❌ Format response tidak valid: {e}"

# =========================
# 🎯 BUSINESS LOGIC MODULE
# =========================

class QueryAnalyzer:
    """Menganalisis pertanyaan user"""
    
    @staticmethod
    def detect_pbj_context(question: str) -> bool:
        """Deteksi apakah pertanyaan terkait PBJ"""
        question_lower = question.lower()
        return any(keyword in question_lower for keyword in Config.PBJ_KEYWORDS)
    
    @staticmethod
    def get_uploaded_regulations(state: Dict) -> List[str]:
        """Dapatkan regulasi yang sudah diupload"""
        uploaded_files = state.get("processed_files", [])
        return [r for r in Config.REGULATION_STRUCTURE if r in uploaded_files]
    
    @staticmethod
    def determine_context(question: str, state: Dict) -> QueryContext:
        """Tentukan konteks pertanyaan"""
        uploaded_regulations = QueryAnalyzer.get_uploaded_regulations(state)
        is_pbj = QueryAnalyzer.detect_pbj_context(question)
        
        if uploaded_regulations:
            return QueryContext.UPLOADED_REGULATIONS
        elif is_pbj:
            return QueryContext.PBJ_NO_UPLOAD
        else:
            return QueryContext.NON_PBJ

class ResponseHandler:
    """Menangani pembuatan response berdasarkan konteks"""
    
    @staticmethod
    def handle_uploaded_regulations(question: str, uploaded_regs: List[str]) -> ResponseData:
        """Handle ketika ada regulasi yang diupload"""
        prompt = GeminiService.build_prompt(question, has_uploaded=True)
        response = GeminiService.call_api(prompt)
        validated = CitationValidator.validate(response)
        
        return ResponseData(
            output_text=validated,
            note="",
            source_type="regulation"
        )
    
    @staticmethod
    def handle_pbj_no_upload(question: str) -> ResponseData:
        """Handle ketika konteks PBJ tapi belum ada upload"""
        prompt = GeminiService.build_prompt(question, has_uploaded=False)
        response = GeminiService.call_api(prompt)
        validated = CitationValidator.validate(response)
        
        return ResponseData(
            output_text=validated,
            warning="⚠️ Regulasi belum diupload. Validitas jawaban TIDAK terjamin.",
            note="Jawaban ini menggunakan pengetahuan umum dan mungkin tidak akurat. Silakan upload regulasi untuk jawaban yang terverifikasi.",
            source_type="external"
        )
    
    @staticmethod
    def handle_non_pbj(question: str) -> ResponseData:
        """Handle ketika bukan konteks PBJ"""
        return ResponseData(
            output_text="Maaf, pertanyaan Anda tidak sesuai dengan konteks pengadaan barang/jasa. ReguBot dirancang khusus untuk menjawab pertanyaan terkait regulasi pengadaan barang/jasa di Indonesia.",
            note="Silakan ajukan pertanyaan terkait: pengadaan barang/jasa, lelang, tender, kontrak, atau regulasi terkait.",
            source_type="none"
        )
    
    @staticmethod
    def process_query(question: str, state: Dict) -> ResponseData:
        """Proses query user dan return response yang sesuai"""
        context = QueryAnalyzer.determine_context(question, state)
        uploaded_regs = QueryAnalyzer.get_uploaded_regulations(state)
        
        if context == QueryContext.UPLOADED_REGULATIONS:
            return ResponseHandler.handle_uploaded_regulations(question, uploaded_regs)
        elif context == QueryContext.PBJ_NO_UPLOAD:
            return ResponseHandler.handle_pbj_no_upload(question)
        else:
            return ResponseHandler.handle_non_pbj(question)

# =========================
# 🎨 UI/UX MODULE - ENTERPRISE RESPONSIVE FLUENT DESIGN
# =========================

class FluentUI:

    @staticmethod
    def inject_custom_css():
        st.markdown(f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Segoe+UI:wght@300;400;600;700&display=swap');

        * {{
            font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif;
            box-sizing: border-box;
        }}

        /* App Background */
        .stApp {{
            background: linear-gradient(180deg, {Config.FLUENT_COLORS['background']} 0%, #ECECEC 100%);
        }}

        /* Responsive Layout */
        @media (max-width: 1024px) {{
            .block-container {{
                padding: 1rem !important;
            }}
        }}

        @media (max-width: 640px) {{
            h1 {{ font-size: 1.4rem; }}
            h2 {{ font-size: 1.2rem; }}
            .stButton > button {{ width: 100%; }}
        }}

        /* Sidebar */
        [data-testid="stSidebar"] {{
            background: white;
            border-right: 1px solid {Config.FLUENT_COLORS['border']};
            min-width: 280px;
        }}

        /* Header */
        .header-container {{
            text-align: center;
            padding: 1rem 0 0.5rem 0;
        }}

        .header-title {{
            font-size: 2rem;
            font-weight: 700;
            color: {Config.FLUENT_COLORS['primary']};
        }}

        .header-subtitle {{
            color: {Config.FLUENT_COLORS['text_secondary']};
            font-size: 1rem;
        }}

        /* Chat Messages */
        .stChatMessage {{
            background: white;
            border-radius: 10px;
            padding: 0.85rem 1rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.06);
            border-left: 4px solid {Config.FLUENT_COLORS['primary']};
        }}

        /* Input */
        .stChatInput {{
            border-radius: 28px;
            border: 2px solid {Config.FLUENT_COLORS['border']};
            padding: 0.6rem;
        }}

        .stChatInput:focus-within {{
            border-color: {Config.FLUENT_COLORS['primary']};
            box-shadow: 0 0 0 4px rgba(0,120,212,0.12);
        }}

        /* Buttons */
        .stButton > button {{
            background: {Config.FLUENT_COLORS['primary']};
            border-radius: 6px;
            font-weight: 600;
            padding: 0.6rem 1.2rem;
        }}

        /* Cards */
        .enterprise-card {{
            background: white;
            border-radius: 10px;
            padding: 1rem;
            margin-bottom: 0.75rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
            border-left: 4px solid {Config.FLUENT_COLORS['accent']};
        }}

        /* Document Items */
        .doc-item {{
            padding: 0.65rem 0.9rem;
            border-radius: 6px;
            background: #FAFAFA;
            border-left: 3px solid {Config.FLUENT_COLORS['accent']};
            font-size: 0.95rem;
        }}

        /* Warning Badge */
        .warning-badge {{
            background: {Config.FLUENT_COLORS['warning']};
            color: white;
            border-radius: 6px;
            padding: 0.5rem 0.8rem;
            font-weight: 600;
            margin-bottom: 0.75rem;
        }}

        /* Footer */
        .footer {{
            text-align: center;
            padding: 1.2rem 0;
            font-size: 12px;
            color: #777;
        }}
        </style>
        """, unsafe_allow_html=True)

    @staticmethod
    def render_header():
        st.markdown("""
        <div class='header-container'>
            <div class='header-title'>ReguBot</div>
            <div class='header-subtitle'>Regulasi ChatBot</div>
        </div>
        """, unsafe_allow_html=True)

    @staticmethod
    def render_sidebar_logo():
        st.markdown(
            "<div style='text-align:center; padding: 0.75rem 0;'>"
            "<img src='https://raw.githubusercontent.com/YonaFr/ReguBot/main/PBJ.png' style='width: 110px; border-radius: 10px;'>"
            "</div>", unsafe_allow_html=True
        )

    @staticmethod
    def render_document_item(filename: str):
        st.markdown(f"<div class='doc-item'>📄 {filename}</div>", unsafe_allow_html=True)

    @staticmethod
    def render_response_with_badges(response: ResponseData):
        if response.warning:
            st.markdown(f"<div class='warning-badge'>{response.warning}</div>", unsafe_allow_html=True)

        st.markdown(f"<div class='enterprise-card'>{response.output_text}</div>", unsafe_allow_html=True)

        if response.note:
            st.info(response.note)

    @staticmethod
    def render_footer():
        """Render footer dengan copyright (TEKS TIDAK DIUBAH)"""
        st.markdown(
            """
            <div style='text-align:center; padding: 2rem 0 1rem 0; font-size:12px; color:#777;'>
                <div style='margin-bottom: 0.5rem;'>
                    <img src='https://mirrors.creativecommons.org/presskit/icons/cc.svg' style='max-width:1em;max-height:1em;margin:0 0.1em;'>
                    <img src='https://mirrors.creativecommons.org/presskit/icons/by.svg' style='max-width:1em;max-height:1em;margin:0 0.1em;'>
                    <img src='https://mirrors.creativecommons.org/presskit/icons/nc.svg' style='max-width:1em;max-height:1em;margin:0 0.1em;'>
                    <img src='https://mirrors.creativecommons.org/presskit/icons/sa.svg' style='max-width:1em;max-height:1em;margin:0 0.1em;'>
                </div>
                <div style='margin: 0.5rem 0;'>
                    © 2025 Yona Friantina. Some rights reserved.
                </div>
                <div style='font-size:11px; color:#999;'>
                    Built with Streamlit • Powered by Gemini AI
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown(
            "<div class='footer'>© 2025 ReguBot • Fluent Government UI • Streamlit + Gemini</div>",
            unsafe_allow_html=True
        )


# =========================
# 🚀 MAIN APPLICATION
# =========================

class ReguBotApp:
    """Main application class"""
    
    def __init__(self):
        self.state_manager = StateManager()
        self.pdf_processor = PDFProcessor()
        
    def setup_page(self):
        """Setup page configuration"""
        st.set_page_config(
            page_title="ReguBot | Asisten Regulasi PBJ",
            page_icon="https://raw.githubusercontent.com/YonaFr/YonaFr.GitHub.IO/main/PBJ.ico",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        FluentUI.inject_custom_css()
    
    def initialize_session_state(self):
        """Initialize session state"""
        if "messages" not in st.session_state:
            st.session_state.messages = [
                {
                    "role": "assistant", 
                    "content": "👋 Selamat datang! Saya ReguBot, asisten Anda untuk pertanyaan seputar regulasi pengadaan barang/jasa. Silakan ajukan pertanyaan Anda."
                }
            ]
    
    def render_sidebar(self):
        """Render sidebar dengan semua komponennya"""
        with st.sidebar:
            FluentUI.render_sidebar_logo()
            
            st.markdown("### 📂 Upload Dokumen Regulasi")
            st.markdown(
                "<p style='font-size: 0.9rem; color: #605E5C; margin-bottom: 1rem;'>"
                "Upload file PDF regulasi untuk jawaban yang terverifikasi"
                "</p>",
                unsafe_allow_html=True
            )
            
            pdf_docs = st.file_uploader(
                "Pilih file PDF",
                accept_multiple_files=True,
                type=["pdf"],
                help="Upload satu atau lebih file PDF regulasi"
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🚀 Proses", use_container_width=True):
                    if pdf_docs:
                        uploaded_names = [pdf.name for pdf in pdf_docs]
                        with st.spinner("🔄 Memproses dokumen..."):
                            try:
                                raw_text = self.pdf_processor.extract_text(pdf_docs)
                                chunks = self.pdf_processor.create_chunks(raw_text)
                                self.state_manager.save_state(uploaded_names)
                                st.success("✅ Berhasil diproses!")
                                st.rerun()
                            except Exception as e:
                                st.error(f"❌ Error: {e}")
                    else:
                        st.warning("⚠️ Pilih minimal 1 file")
            
            with col2:
                if st.button("🧹 Reset", use_container_width=True):
                    st.session_state.messages = [
                        {
                            "role": "assistant",
                            "content": "👋 Chat telah direset. Silakan mulai percakapan baru."
                        }
                    ]
                    st.rerun()
            
            st.markdown("---")
            
            # Tampilkan dokumen yang sudah diupload
            state = self.state_manager.load_state()
            if state["processed_files"]:
                st.markdown("### 📚 Dokumen Terupload")
                for filename in state["processed_files"]:
                    FluentUI.render_document_item(filename)
                
                st.markdown("---")
                
                # Status info
                st.markdown(
                    "<div class='info-card'>"
                    "<strong>✅ Status:</strong> Regulasi Terverifikasi<br>"
                    "<small>Jawaban berdasarkan dokumen yang telah diupload</small>"
                    "</div>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    "<div class='info-card' style='border-left-color: #FF8C00;'>"
                    "<strong>⚠️ Status:</strong> Belum Ada Upload<br>"
                    "<small>Upload regulasi untuk jawaban terverifikasi</small>"
                    "</div>",
                    unsafe_allow_html=True
                )
            
            st.markdown("---")
            FluentUI.render_footer()
    
    def render_chat_interface(self):
        """Render chat interface"""
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("💬 Ketik pertanyaan Anda di sini..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate assistant response
            with st.chat_message("assistant"):
                with st.spinner("🤔 Memproses jawaban..."):
                    try:
                        state = self.state_manager.load_state()
                        response = ResponseHandler.process_query(prompt, state)
                        
                        # Render response with proper formatting
                        FluentUI.render_response_with_badges(response)
                        
                        # Prepare full response for history
                        full_response = response.output_text
                        if response.warning:
                            full_response = f"{response.warning}\n\n{full_response}"
                        if response.note:
                            full_response += f"\n\n💡 {response.note}"
                        
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": full_response
                        })
                        
                    except Exception as e:
                        error_msg = f"❌ Terjadi kesalahan: {str(e)}"
                        st.error(error_msg)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": error_msg
                        })
    
    def run(self):
        """Run the application"""
        self.setup_page()
        self.initialize_session_state()
        
        FluentUI.render_header()
        self.render_sidebar()
        self.render_chat_interface()

# =========================
# 🎬 APPLICATION ENTRY POINT
# =========================

def main():
    """Main entry point"""
    app = ReguBotApp()
    app.run()

if __name__ == "__main__":
    main()
