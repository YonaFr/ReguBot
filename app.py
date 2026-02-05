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
    GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.0-flash-preview:generateContent"
    
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
    
    # Microsoft Fluent Design Colors (Enhanced)
    FLUENT_COLORS = {
        "primary": "#0078D4",
        "primary_hover": "#106EBE",
        "primary_dark": "#005A9E",
        "secondary": "#8764B8",
        "accent": "#0099BC",
        "success": "#107C10",
        "warning": "#FF8C00",
        "error": "#E81123",
        "info": "#0078D4",
        "background": "#F3F2F1",
        "background_gradient": "linear-gradient(135deg, #F3F2F1 0%, #E1DFDD 100%)",
        "surface": "#FFFFFF",
        "surface_alt": "#FAF9F8",
        "text_primary": "#323130",
        "text_secondary": "#605E5C",
        "text_tertiary": "#8A8886",
        "text_disabled": "#A19F9D",
        "border": "#EDEBE9",
        "border_strong": "#D2D0CE",
        "divider": "#E1DFDD",
        "overlay": "rgba(0, 0, 0, 0.4)",
        "shadow_light": "rgba(0, 0, 0, 0.1)",
        "shadow_medium": "rgba(0, 0, 0, 0.15)",
        "shadow_strong": "rgba(0, 0, 0, 0.25)"
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
# 🎨 UI/UX MODULE - FLUENT DESIGN
# =========================

class FluentUI:
    """Komponen UI dengan Microsoft Fluent Design System (Enhanced & Compact)"""
    
    @staticmethod
    def inject_custom_css():
        """Inject custom CSS untuk Microsoft Fluent Design"""
        st.markdown(f"""
        <style>
        /* Import Segoe UI Font (Fluent Design) */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        /* Global Styles - Fluent Design */
        * {{
            font-family: 'Segoe UI', 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            -webkit-font-smoothing: antialiased;
            -moz-osx-font-smoothing: grayscale;
        }}
        
        /* Main Container with Fluent Acrylic Background */
        .stApp {{
            background: {Config.FLUENT_COLORS['background_gradient']};
        }}
        
        /* Sidebar Styling - Fluent Acrylic */
        [data-testid="stSidebar"] {{
            background: rgba(255, 255, 255, 0.85);
            backdrop-filter: blur(30px);
            -webkit-backdrop-filter: blur(30px);
            border-right: 1px solid {Config.FLUENT_COLORS['border']};
            box-shadow: 2px 0 12px {Config.FLUENT_COLORS['shadow_light']};
        }}
        
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {{
            padding: 0.25rem;
        }}
        
        /* Typography - Fluent Design */
        h1, h2, h3, h4, h5, h6 {{
            color: {Config.FLUENT_COLORS['text_primary']};
            font-weight: 600;
            letter-spacing: -0.01em;
        }}
        
        h1 {{
            font-size: 1.75rem;
            font-weight: 600;
            line-height: 1.2;
            margin-bottom: 0.5rem;
        }}
        
        h2 {{
            font-size: 1.5rem;
            font-weight: 600;
            line-height: 1.3;
        }}
        
        h3 {{
            font-size: 1.125rem;
            font-weight: 600;
            line-height: 1.4;
            margin-bottom: 0.5rem;
        }}
        
        p {{
            color: {Config.FLUENT_COLORS['text_primary']};
            line-height: 1.5;
            font-size: 0.9375rem;
        }}
        
        /* Buttons - Fluent Design (Compact) */
        .stButton > button {{
            background: {Config.FLUENT_COLORS['primary']};
            color: white;
            border: none;
            border-radius: 4px;
            padding: 0.4rem 1rem;
            font-weight: 600;
            font-size: 0.875rem;
            transition: all 0.15s cubic-bezier(0.33, 0, 0.67, 1);
            box-shadow: 0 1.6px 3.6px 0 {Config.FLUENT_COLORS['shadow_light']}, 
                        0 0.3px 0.9px 0 {Config.FLUENT_COLORS['shadow_light']};
            height: 32px;
            min-width: 80px;
        }}
        
        .stButton > button:hover {{
            background: {Config.FLUENT_COLORS['primary_hover']};
            box-shadow: 0 3.2px 7.2px 0 {Config.FLUENT_COLORS['shadow_medium']}, 
                        0 0.6px 1.8px 0 {Config.FLUENT_COLORS['shadow_medium']};
            transform: translateY(-1px);
        }}
        
        .stButton > button:active {{
            transform: translateY(0);
            box-shadow: 0 0.8px 1.8px 0 {Config.FLUENT_COLORS['shadow_light']};
        }}
        
        .stButton > button:focus {{
            outline: 2px solid {Config.FLUENT_COLORS['primary']};
            outline-offset: 2px;
        }}
        
        /* File Uploader - Fluent Acrylic */
        [data-testid="stFileUploader"] {{
            background: rgba(255, 255, 255, 0.6);
            backdrop-filter: blur(20px);
            border: 2px dashed {Config.FLUENT_COLORS['border']};
            border-radius: 8px;
            padding: 1rem;
            transition: all 0.2s cubic-bezier(0.33, 0, 0.67, 1);
        }}
        
        [data-testid="stFileUploader"]:hover {{
            border-color: {Config.FLUENT_COLORS['primary']};
            background: rgba(255, 255, 255, 0.8);
            box-shadow: 0 4px 12px {Config.FLUENT_COLORS['shadow_light']};
        }}
        
        /* Chat Messages - Fluent Cards */
        .stChatMessage {{
            background: rgba(255, 255, 255, 0.7);
            backdrop-filter: blur(20px);
            border-radius: 8px;
            padding: 0.75rem;
            margin: 0.4rem 0;
            box-shadow: 0 2px 6px {Config.FLUENT_COLORS['shadow_light']};
            border-left: 3px solid {Config.FLUENT_COLORS['primary']};
            transition: all 0.2s cubic-bezier(0.33, 0, 0.67, 1);
        }}
        
        .stChatMessage:hover {{
            box-shadow: 0 4px 12px {Config.FLUENT_COLORS['shadow_medium']};
            transform: translateX(2px);
        }}
        
        /* Chat Input - Fluent Style */
        .stChatInput {{
            border-radius: 20px;
            border: 1.5px solid {Config.FLUENT_COLORS['border']};
            background: rgba(255, 255, 255, 0.9);
            backdrop-filter: blur(20px);
            transition: all 0.2s cubic-bezier(0.33, 0, 0.67, 1);
            box-shadow: 0 2px 6px {Config.FLUENT_COLORS['shadow_light']};
        }}
        
        .stChatInput:focus-within {{
            border-color: {Config.FLUENT_COLORS['primary']};
            box-shadow: 0 0 0 3px rgba(0, 120, 212, 0.1),
                        0 4px 12px {Config.FLUENT_COLORS['shadow_medium']};
        }}
        
        /* Input Fields - Fluent */
        input, textarea {{
            background: rgba(255, 255, 255, 0.8);
            border: 1px solid {Config.FLUENT_COLORS['border']};
            border-radius: 4px;
            padding: 0.5rem 0.75rem;
            font-size: 0.9375rem;
            color: {Config.FLUENT_COLORS['text_primary']};
            transition: all 0.15s cubic-bezier(0.33, 0, 0.67, 1);
        }}
        
        input:hover, textarea:hover {{
            border-color: {Config.FLUENT_COLORS['border_strong']};
            background: rgba(255, 255, 255, 0.95);
        }}
        
        input:focus, textarea:focus {{
            outline: none;
            border-color: {Config.FLUENT_COLORS['primary']};
            box-shadow: 0 0 0 3px rgba(0, 120, 212, 0.1);
        }}
        
        /* Alerts - Fluent Notifications */
        .stAlert {{
            border-radius: 6px;
            border: none;
            padding: 0.65rem 1rem;
            box-shadow: 0 3px 8px {Config.FLUENT_COLORS['shadow_light']};
            font-size: 0.875rem;
        }}
        
        /* Success Alert */
        [data-baseweb="notification"] {{
            background: {Config.FLUENT_COLORS['success']};
            color: white;
            border-radius: 6px;
        }}
        
        /* Warning Alert */
        .stWarning {{
            background: #FFF4CE;
            border-left: 3px solid {Config.FLUENT_COLORS['warning']};
            color: #333;
        }}
        
        /* Error Alert */
        .stError {{
            background: #FDE7E9;
            border-left: 3px solid {Config.FLUENT_COLORS['error']};
            color: #333;
        }}
        
        /* Info Alert */
        .stInfo {{
            background: #E7F3FF;
            border-left: 3px solid {Config.FLUENT_COLORS['info']};
            color: #333;
        }}
        
        /* Fluent Cards */
        .fluent-card {{
            background: rgba(255, 255, 255, 0.8);
            backdrop-filter: blur(20px);
            border-radius: 8px;
            padding: 0.75rem;
            margin: 0.4rem 0;
            box-shadow: 0 2px 6px {Config.FLUENT_COLORS['shadow_light']};
            border: 1px solid {Config.FLUENT_COLORS['border']};
            transition: all 0.2s cubic-bezier(0.33, 0, 0.67, 1);
        }}
        
        .fluent-card:hover {{
            box-shadow: 0 4px 12px {Config.FLUENT_COLORS['shadow_medium']};
            transform: translateY(-2px);
            border-color: {Config.FLUENT_COLORS['border_strong']};
        }}
        
        /* Elevated Card */
        .fluent-card-elevated {{
            background: rgba(255, 255, 255, 0.9);
            backdrop-filter: blur(30px);
            border-radius: 12px;
            padding: 1rem;
            margin: 0.75rem 0;
            box-shadow: 0 4px 16px {Config.FLUENT_COLORS['shadow_medium']};
            border: 1px solid {Config.FLUENT_COLORS['border']};
        }}
        
        /* Spinner - Fluent */
        .stSpinner > div {{
            border-color: {Config.FLUENT_COLORS['primary']} transparent transparent transparent;
        }}
        
        /* Scrollbar - Fluent */
        ::-webkit-scrollbar {{
            width: 10px;
            height: 10px;
        }}
        
        ::-webkit-scrollbar-track {{
            background: {Config.FLUENT_COLORS['surface_alt']};
            border-radius: 5px;
        }}
        
        ::-webkit-scrollbar-thumb {{
            background: {Config.FLUENT_COLORS['border_strong']};
            border-radius: 5px;
            border: 2px solid {Config.FLUENT_COLORS['surface_alt']};
        }}
        
        ::-webkit-scrollbar-thumb:hover {{
            background: {Config.FLUENT_COLORS['text_tertiary']};
        }}
        
        /* Document List - Fluent */
        .doc-item {{
            background: rgba(255, 255, 255, 0.7);
            backdrop-filter: blur(20px);
            padding: 0.5rem 0.75rem;
            margin: 0.3rem 0;
            border-radius: 6px;
            border-left: 3px solid {Config.FLUENT_COLORS['accent']};
            box-shadow: 0 1px 3px {Config.FLUENT_COLORS['shadow_light']};
            transition: all 0.15s cubic-bezier(0.33, 0, 0.67, 1);
            font-size: 0.875rem;
        }}
        
        .doc-item:hover {{
            box-shadow: 0 2px 8px {Config.FLUENT_COLORS['shadow_medium']};
            transform: translateX(4px);
            border-left-width: 4px;
        }}
        
        /* Fluent Pills/Badges */
        .fluent-badge {{
            display: inline-flex;
            align-items: center;
            background: {Config.FLUENT_COLORS['surface']};
            color: {Config.FLUENT_COLORS['text_primary']};
            padding: 0.25rem 0.65rem;
            height: 24px;
            font-size: 0.75rem;
            font-weight: 600;
            border-radius: 12px;
            margin: 0.2rem;
            border: 1px solid {Config.FLUENT_COLORS['border']};
            box-shadow: 0 1px 2px {Config.FLUENT_COLORS['shadow_light']};
        }}
        
        .fluent-badge.primary {{
            background: {Config.FLUENT_COLORS['primary']};
            color: white;
            border-color: {Config.FLUENT_COLORS['primary']};
        }}
        
        .fluent-badge.success {{
            background: {Config.FLUENT_COLORS['success']};
            color: white;
            border-color: {Config.FLUENT_COLORS['success']};
        }}
        
        .fluent-badge.warning {{
            background: {Config.FLUENT_COLORS['warning']};
            color: white;
            border-color: {Config.FLUENT_COLORS['warning']};
        }}
        
        .fluent-badge.error {{
            background: {Config.FLUENT_COLORS['error']};
            color: white;
            border-color: {Config.FLUENT_COLORS['error']};
        }}
        
        /* Response Container - Fluent */
        .response-container {{
            background: rgba(255, 255, 255, 0.8);
            backdrop-filter: blur(20px);
            border-radius: 8px;
            padding: 1rem;
            margin: 0.75rem 0;
            box-shadow: 0 3px 10px {Config.FLUENT_COLORS['shadow_light']};
            border-left: 4px solid {Config.FLUENT_COLORS['primary']};
        }}
        
        /* Fluent Banner */
        .fluent-banner {{
            background: rgba(255, 140, 0, 0.1);
            border-left: 3px solid {Config.FLUENT_COLORS['warning']};
            color: {Config.FLUENT_COLORS['text_primary']};
            padding: 0.65rem 1rem;
            margin: 0.4rem 0;
            border-radius: 6px;
            font-size: 0.875rem;
            display: flex;
            align-items: center;
            box-shadow: 0 2px 6px {Config.FLUENT_COLORS['shadow_light']};
            backdrop-filter: blur(10px);
        }}
        
        .fluent-banner.error {{
            background: rgba(232, 17, 35, 0.1);
            border-left-color: {Config.FLUENT_COLORS['error']};
        }}
        
        .fluent-banner.success {{
            background: rgba(16, 124, 16, 0.1);
            border-left-color: {Config.FLUENT_COLORS['success']};
        }}
        
        .fluent-banner.info {{
            background: rgba(0, 120, 212, 0.1);
            border-left-color: {Config.FLUENT_COLORS['info']};
        }}
        
        /* Divider - Fluent */
        hr {{
            border: none;
            border-top: 1px solid {Config.FLUENT_COLORS['divider']};
            margin: 0.75rem 0;
        }}
        
        /* Link Styling - Fluent */
        a {{
            color: {Config.FLUENT_COLORS['primary']};
            text-decoration: none;
            transition: all 0.15s cubic-bezier(0.33, 0, 0.67, 1);
            border-bottom: 1px solid transparent;
        }}
        
        a:hover {{
            color: {Config.FLUENT_COLORS['primary_hover']};
            border-bottom: 1px solid {Config.FLUENT_COLORS['primary_hover']};
        }}
        
        /* Code blocks - Fluent */
        code {{
            font-family: 'Consolas', 'Courier New', monospace;
            background: rgba(0, 0, 0, 0.05);
            padding: 0.125rem 0.4rem;
            font-size: 0.875rem;
            border-radius: 4px;
            border: 1px solid {Config.FLUENT_COLORS['border']};
        }}
        
        pre {{
            background: {Config.FLUENT_COLORS['text_primary']};
            color: white;
            padding: 0.75rem;
            font-family: 'Consolas', 'Courier New', monospace;
            font-size: 0.875rem;
            overflow-x: auto;
            border-radius: 6px;
            box-shadow: 0 3px 10px {Config.FLUENT_COLORS['shadow_medium']};
        }}
        
        /* Loading Animation - Fluent */
        .loading-shimmer {{
            background: linear-gradient(90deg, 
                {Config.FLUENT_COLORS['surface_alt']} 25%, 
                {Config.FLUENT_COLORS['surface']} 50%, 
                {Config.FLUENT_COLORS['surface_alt']} 75%);
            background-size: 200% 100%;
            animation: shimmer 1.5s ease-in-out infinite;
        }}
        
        @keyframes shimmer {{
            0% {{ background-position: 200% 0; }}
            100% {{ background-position: -200% 0; }}
        }}
        
        /* Reveal Animation - Fluent */
        @keyframes reveal {{
            from {{
                opacity: 0;
                transform: translateY(10px);
            }}
            to {{
                opacity: 1;
                transform: translateY(0);
            }}
        }}
        
        /* Focus visible for accessibility */
        *:focus-visible {{
            outline: 2px solid {Config.FLUENT_COLORS['primary']};
            outline-offset: 2px;
        }}
        
        /* Disabled state */
        button:disabled, input:disabled {{
            opacity: 0.4;
            cursor: not-allowed;
        }}
        
        /* Compact spacing adjustments */
        [data-testid="stMarkdownContainer"] {{
            padding: 0.25rem 0;
        }}
        
        .element-container {{
            margin: 0.25rem 0;
        }}
        </style>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def render_header():
        """Render header dengan Microsoft Fluent Design"""
        st.markdown(f"""
        <div style='padding: 1rem 0 0.75rem 0;'>
            <div style='color: {Config.FLUENT_COLORS['text_secondary']}; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.4rem; font-weight: 600;'>
                Sistem Regulasi Pengadaan
            </div>
            <h1 style='color: {Config.FLUENT_COLORS['primary']}; font-weight: 600; margin: 0; font-size: 1.75rem;'>
                ReguBot
            </h1>
            <p style='color: {Config.FLUENT_COLORS['text_secondary']}; font-size: 1rem; margin-top: 0.4rem; font-weight: 400;'>
                Asisten cerdas untuk regulasi pengadaan barang dan jasa
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def render_sidebar_logo():
        """Render logo di sidebar dengan Fluent Design"""
        st.markdown(
            f"<div style='text-align:center; padding: 1rem 0 0.75rem 0;'>"
            "<img src='https://raw.githubusercontent.com/YonaFr/ReguBot/main/PBJ.png' "
            f"style='width: 90px; border-radius: 8px; box-shadow: 0 4px 12px {Config.FLUENT_COLORS['shadow_medium']};'>"
            "</div>",
            unsafe_allow_html=True
        )
    
    @staticmethod
    def render_document_item(filename: str):
        """Render item dokumen dengan Fluent styling"""
        st.markdown(
            f"<div class='doc-item'>📄 {filename}</div>",
            unsafe_allow_html=True
        )
    
    @staticmethod
    def render_response_with_badges(response: ResponseData):
        """Render response dengan Fluent badges"""
        # Warning banner jika ada
        if response.warning:
            st.markdown(
                f"<div class='fluent-banner'>"
                f"<strong style='margin-right: 0.5rem;'>⚠️</strong>"
                f"<span>{response.warning}</span>"
                f"</div>",
                unsafe_allow_html=True
            )
        
        # Main response
        st.markdown(response.output_text)
        
        # Note jika ada
        if response.note:
            st.markdown(
                f"<div class='fluent-banner info' style='margin-top: 0.5rem;'>"
                f"<strong style='margin-right: 0.5rem;'>ℹ️</strong>"
                f"<span>{response.note}</span>"
                f"</div>",
                unsafe_allow_html=True
            )
    
    @staticmethod
    def render_footer():
        """Render footer dengan Microsoft Fluent Design"""
        st.markdown(
            f"""
            <div style='text-align:center; padding: 1.25rem 0 0.75rem 0; font-size: 0.7rem; color: {Config.FLUENT_COLORS['text_tertiary']};'>
                <div style='margin-bottom: 0.5rem;'>
                    <img src='https://mirrors.creativecommons.org/presskit/icons/cc.svg' style='max-width:0.9em;max-height:0.9em;margin:0 0.15em;opacity:0.6;'>
                    <img src='https://mirrors.creativecommons.org/presskit/icons/by.svg' style='max-width:0.9em;max-height:0.9em;margin:0 0.15em;opacity:0.6;'>
                    <img src='https://mirrors.creativecommons.org/presskit/icons/nc.svg' style='max-width:0.9em;max-height:0.9em;margin:0 0.15em;opacity:0.6;'>
                    <img src='https://mirrors.creativecommons.org/presskit/icons/sa.svg' style='max-width:0.9em;max-height:0.9em;margin:0 0.15em;opacity:0.6;'>
                </div>
                <div style='margin: 0.4rem 0; font-weight: 500;'>
                    © 2025 Yona Friantina
                </div>
                <div style='color: {Config.FLUENT_COLORS['text_disabled']}; font-size: 0.7rem;'>
                    Built with Streamlit × Powered by Gemini AI
                </div>
            </div>
            """,
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
            
            st.markdown("### 📂 Upload Dokumen")
            st.markdown(
                f"<p style='font-size: 0.8rem; color: {Config.FLUENT_COLORS['text_secondary']}; margin-bottom: 0.5rem;'>"
                "Upload file PDF regulasi untuk jawaban terverifikasi"
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
                        with st.spinner("🔄 Memproses..."):
                            try:
                                raw_text = self.pdf_processor.extract_text(pdf_docs)
                                chunks = self.pdf_processor.create_chunks(raw_text)
                                self.state_manager.save_state(uploaded_names)
                                st.success("✅ Berhasil!")
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
                # Status info - Fluent Card
                st.markdown(
                    f"<div class='fluent-card'>"
                    f"<div style='color: {Config.FLUENT_COLORS['success']}; font-weight: 600; margin-bottom: 0.25rem; display: flex; align-items: center;'>"
                    f"<span style='margin-right: 0.4rem;'>✅</span>"
                    f"<span style='font-size: 0.875rem;'>Status: Terverifikasi</span>"
                    f"</div>"
                    f"<div style='font-size: 0.75rem; color: {Config.FLUENT_COLORS['text_secondary']};'>Jawaban berdasarkan dokumen yang diupload</div>"
                    f"</div>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"<div class='fluent-card' style='border-left: 3px solid {Config.FLUENT_COLORS['warning']};'>"
                    f"<div style='color: {Config.FLUENT_COLORS['warning']}; font-weight: 600; margin-bottom: 0.25rem; display: flex; align-items: center;'>"
                    f"<span style='margin-right: 0.4rem;'>⚠️</span>"
                    f"<span style='font-size: 0.875rem;'>Status: Belum Ada Upload</span>"
                    f"</div>"
                    f"<div style='font-size: 0.75rem; color: {Config.FLUENT_COLORS['text_secondary']};'>Upload regulasi untuk jawaban terverifikasi</div>"
                    f"</div>",
                    unsafe_allow_html=True
                )
            
            st.markdown("---")
                
                st.markdown("### 📚 Dokumen Terupload")
                for filename in state["processed_files"]:
                    FluentUI.render_document_item(filename)
                
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
