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
    
    # Google Material Design Colors
    MATERIAL_COLORS = {
        "primary": "#1976D2",
        "primary_dark": "#1565C0",
        "primary_light": "#BBDEFB",
        "secondary": "#FF6F00",
        "secondary_dark": "#E65100",
        "secondary_light": "#FFE0B2",
        "accent": "#00BCD4",
        "success": "#4CAF50",
        "warning": "#FF9800",
        "error": "#F44336",
        "info": "#2196F3",
        "background": "#FAFAFA",
        "surface": "#FFFFFF",
        "surface_variant": "#F5F5F5",
        "text_primary": "#212121",
        "text_secondary": "#757575",
        "text_hint": "#9E9E9E",
        "text_disabled": "#BDBDBD",
        "divider": "#E0E0E0",
        "border": "#BDBDBD",
        "overlay": "rgba(0, 0, 0, 0.54)",
        "shadow": "rgba(0, 0, 0, 0.2)"
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
            note=f"✅ Sumber: {', '.join(uploaded_regs)}",
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

class MaterialUI:
    """Komponen UI dengan Google Material Design System"""
    
    @staticmethod
    def inject_custom_css():
        """Inject custom CSS untuk Google Material Design"""
        st.markdown(f"""
        <style>
        /* Import Roboto Font Family (Material Design) */
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&family=Roboto+Mono:wght@400;500&display=swap');
        @import url('https://fonts.googleapis.com/icon?family=Material+Icons');
        
        /* Global Styles - Material Design */
        * {{
            font-family: 'Roboto', 'Helvetica', 'Arial', sans-serif;
            -webkit-font-smoothing: antialiased;
            -moz-osx-font-smoothing: grayscale;
        }}
        
        /* Main Container */
        .stApp {{
            background: {Config.MATERIAL_COLORS['background']};
        }}
        
        /* Sidebar Styling - Material */
        [data-testid="stSidebar"] {{
            background: {Config.MATERIAL_COLORS['surface']};
            box-shadow: 2px 0 8px rgba(0, 0, 0, 0.1);
            border-right: none;
        }}
        
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {{
            padding: 0.5rem;
        }}
        
        /* Typography - Material Design */
        h1, h2, h3, h4, h5, h6 {{
            color: {Config.MATERIAL_COLORS['text_primary']};
            font-weight: 500;
            letter-spacing: 0.0125em;
            margin: 0;
        }}
        
        h1 {{
            font-size: 2.125rem;
            font-weight: 400;
            letter-spacing: 0;
            line-height: 1.235;
            margin-bottom: 0.5rem;
        }}
        
        h2 {{
            font-size: 1.5rem;
            font-weight: 500;
            letter-spacing: 0.0075em;
            line-height: 1.334;
        }}
        
        h3 {{
            font-size: 1.25rem;
            font-weight: 500;
            letter-spacing: 0.0075em;
            line-height: 1.6;
        }}
        
        p {{
            color: {Config.MATERIAL_COLORS['text_primary']};
            line-height: 1.5;
            font-size: 1rem;
            letter-spacing: 0.03125em;
        }}
        
        /* Buttons - Material Design (Raised) */
        .stButton > button {{
            background: {Config.MATERIAL_COLORS['primary']};
            color: white;
            border: none;
            border-radius: 4px;
            padding: 0.5rem 1rem;
            font-weight: 500;
            font-size: 0.875rem;
            letter-spacing: 0.0892857143em;
            text-transform: uppercase;
            transition: all 280ms cubic-bezier(0.4, 0, 0.2, 1);
            box-shadow: 0 3px 1px -2px rgba(0,0,0,0.2), 
                        0 2px 2px 0 rgba(0,0,0,0.14), 
                        0 1px 5px 0 rgba(0,0,0,0.12);
            height: 36px;
            min-width: 64px;
        }}
        
        .stButton > button:hover {{
            background: {Config.MATERIAL_COLORS['primary_dark']};
            box-shadow: 0 2px 4px -1px rgba(0,0,0,0.2), 
                        0 4px 5px 0 rgba(0,0,0,0.14), 
                        0 1px 10px 0 rgba(0,0,0,0.12);
        }}
        
        .stButton > button:active {{
            box-shadow: 0 5px 5px -3px rgba(0,0,0,0.2), 
                        0 8px 10px 1px rgba(0,0,0,0.14), 
                        0 3px 14px 2px rgba(0,0,0,0.12);
        }}
        
        .stButton > button:focus {{
            outline: none;
        }}
        
        /* File Uploader - Material */
        [data-testid="stFileUploader"] {{
            background: {Config.MATERIAL_COLORS['surface']};
            border: 2px dashed {Config.MATERIAL_COLORS['border']};
            border-radius: 4px;
            padding: 1.5rem;
            transition: all 280ms cubic-bezier(0.4, 0, 0.2, 1);
        }}
        
        [data-testid="stFileUploader"]:hover {{
            border-color: {Config.MATERIAL_COLORS['primary']};
            background: {Config.MATERIAL_COLORS['surface_variant']};
        }}
        
        /* Chat Messages - Material Cards */
        .stChatMessage {{
            background: {Config.MATERIAL_COLORS['surface']};
            border-radius: 8px;
            padding: 1rem;
            margin: 0.75rem 0;
            box-shadow: 0 2px 1px -1px rgba(0,0,0,0.2), 
                        0 1px 1px 0 rgba(0,0,0,0.14), 
                        0 1px 3px 0 rgba(0,0,0,0.12);
            transition: all 280ms cubic-bezier(0.4, 0, 0.2, 1);
        }}
        
        .stChatMessage:hover {{
            box-shadow: 0 3px 3px -2px rgba(0,0,0,0.2), 
                        0 3px 4px 0 rgba(0,0,0,0.14), 
                        0 1px 8px 0 rgba(0,0,0,0.12);
        }}
        
        /* Chat Input - Material */
        .stChatInput {{
            border-radius: 24px;
            border: 1px solid {Config.MATERIAL_COLORS['divider']};
            background: {Config.MATERIAL_COLORS['surface']};
            transition: all 280ms cubic-bezier(0.4, 0, 0.2, 1);
            box-shadow: 0 2px 1px -1px rgba(0,0,0,0.2), 
                        0 1px 1px 0 rgba(0,0,0,0.14), 
                        0 1px 3px 0 rgba(0,0,0,0.12);
        }}
        
        .stChatInput:focus-within {{
            border-color: {Config.MATERIAL_COLORS['primary']};
            box-shadow: 0 3px 3px -2px rgba(0,0,0,0.2), 
                        0 3px 4px 0 rgba(0,0,0,0.14), 
                        0 1px 8px 0 rgba(0,0,0,0.12);
        }}
        
        /* Input Fields - Material */
        input, textarea {{
            background: transparent;
            border: none;
            border-bottom: 1px solid {Config.MATERIAL_COLORS['divider']};
            border-radius: 4px 4px 0 0;
            padding: 1rem 0.75rem;
            font-size: 1rem;
            color: {Config.MATERIAL_COLORS['text_primary']};
            transition: all 280ms cubic-bezier(0.4, 0, 0.2, 1);
        }}
        
        input:hover, textarea:hover {{
            background: rgba(0, 0, 0, 0.04);
        }}
        
        input:focus, textarea:focus {{
            outline: none;
            border-bottom: 2px solid {Config.MATERIAL_COLORS['primary']};
            background: rgba(0, 0, 0, 0.04);
        }}
        
        /* Alerts - Material Snackbar/Alert */
        .stAlert {{
            border-radius: 4px;
            border: none;
            padding: 0.75rem 1rem;
            box-shadow: 0 3px 5px -1px rgba(0,0,0,0.2), 
                        0 6px 10px 0 rgba(0,0,0,0.14), 
                        0 1px 18px 0 rgba(0,0,0,0.12);
            font-size: 0.875rem;
        }}
        
        /* Success Alert */
        [data-baseweb="notification"] {{
            background: {Config.MATERIAL_COLORS['success']};
            color: white;
            border-radius: 4px;
        }}
        
        /* Warning Alert */
        .stWarning {{
            background: {Config.MATERIAL_COLORS['warning']};
            color: white;
        }}
        
        /* Error Alert */
        .stError {{
            background: {Config.MATERIAL_COLORS['error']};
            color: white;
        }}
        
        /* Info Alert */
        .stInfo {{
            background: {Config.MATERIAL_COLORS['info']};
            color: white;
        }}
        
        /* Material Cards */
        .material-card {{
            background: {Config.MATERIAL_COLORS['surface']};
            border-radius: 4px;
            padding: 1rem;
            margin: 0.75rem 0;
            box-shadow: 0 2px 1px -1px rgba(0,0,0,0.2), 
                        0 1px 1px 0 rgba(0,0,0,0.14), 
                        0 1px 3px 0 rgba(0,0,0,0.12);
            transition: all 280ms cubic-bezier(0.4, 0, 0.2, 1);
        }}
        
        .material-card:hover {{
            box-shadow: 0 3px 3px -2px rgba(0,0,0,0.2), 
                        0 3px 4px 0 rgba(0,0,0,0.14), 
                        0 1px 8px 0 rgba(0,0,0,0.12);
            transform: translateY(-2px);
        }}
        
        /* Elevated Card */
        .material-card-elevated {{
            background: {Config.MATERIAL_COLORS['surface']};
            border-radius: 8px;
            padding: 1.5rem;
            margin: 1rem 0;
            box-shadow: 0 4px 5px -2px rgba(0,0,0,0.2), 
                        0 7px 10px 1px rgba(0,0,0,0.14), 
                        0 2px 16px 1px rgba(0,0,0,0.12);
        }}
        
        /* Spinner - Material */
        .stSpinner > div {{
            border-color: {Config.MATERIAL_COLORS['primary']} transparent transparent transparent;
        }}
        
        /* Scrollbar - Material */
        ::-webkit-scrollbar {{
            width: 12px;
            height: 12px;
        }}
        
        ::-webkit-scrollbar-track {{
            background: {Config.MATERIAL_COLORS['background']};
        }}
        
        ::-webkit-scrollbar-thumb {{
            background: {Config.MATERIAL_COLORS['text_hint']};
            border-radius: 6px;
            border: 3px solid {Config.MATERIAL_COLORS['background']};
        }}
        
        ::-webkit-scrollbar-thumb:hover {{
            background: {Config.MATERIAL_COLORS['text_secondary']};
        }}
        
        /* Document List - Material */
        .doc-item {{
            background: {Config.MATERIAL_COLORS['surface']};
            padding: 0.75rem 1rem;
            margin: 0.5rem 0;
            border-radius: 4px;
            box-shadow: 0 1px 3px 0 rgba(0,0,0,0.1), 
                        0 1px 2px 0 rgba(0,0,0,0.06);
            transition: all 280ms cubic-bezier(0.4, 0, 0.2, 1);
            font-size: 0.875rem;
            border-left: 4px solid {Config.MATERIAL_COLORS['primary']};
        }}
        
        .doc-item:hover {{
            box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1), 
                        0 2px 4px -1px rgba(0,0,0,0.06);
            transform: translateX(4px);
        }}
        
        /* Material Chips */
        .material-chip {{
            display: inline-flex;
            align-items: center;
            background: {Config.MATERIAL_COLORS['surface_variant']};
            color: {Config.MATERIAL_COLORS['text_primary']};
            padding: 0 0.75rem;
            height: 32px;
            font-size: 0.8125rem;
            font-weight: 500;
            border-radius: 16px;
            margin: 0.25rem;
            transition: all 280ms cubic-bezier(0.4, 0, 0.2, 1);
        }}
        
        .material-chip:hover {{
            background: {Config.MATERIAL_COLORS['divider']};
        }}
        
        /* Material Chip - Primary */
        .material-chip.primary {{
            background: {Config.MATERIAL_COLORS['primary']};
            color: white;
        }}
        
        /* Material Chip - Success */
        .material-chip.success {{
            background: {Config.MATERIAL_COLORS['success']};
            color: white;
        }}
        
        /* Material Chip - Warning */
        .material-chip.warning {{
            background: {Config.MATERIAL_COLORS['warning']};
            color: white;
        }}
        
        /* Material Chip - Error */
        .material-chip.error {{
            background: {Config.MATERIAL_COLORS['error']};
            color: white;
        }}
        
        /* Response Container - Material */
        .response-container {{
            background: {Config.MATERIAL_COLORS['surface']};
            border-radius: 8px;
            padding: 1.5rem;
            margin: 1rem 0;
            box-shadow: 0 2px 1px -1px rgba(0,0,0,0.2), 
                        0 1px 1px 0 rgba(0,0,0,0.14), 
                        0 1px 3px 0 rgba(0,0,0,0.12);
        }}
        
        /* Material Banner */
        .material-banner {{
            background: {Config.MATERIAL_COLORS['warning']};
            color: white;
            padding: 1rem 1.5rem;
            margin: 0.75rem 0;
            border-radius: 4px;
            font-size: 0.875rem;
            display: flex;
            align-items: center;
            box-shadow: 0 2px 1px -1px rgba(0,0,0,0.2), 
                        0 1px 1px 0 rgba(0,0,0,0.14), 
                        0 1px 3px 0 rgba(0,0,0,0.12);
        }}
        
        .material-banner.error {{
            background: {Config.MATERIAL_COLORS['error']};
        }}
        
        .material-banner.success {{
            background: {Config.MATERIAL_COLORS['success']};
        }}
        
        .material-banner.info {{
            background: {Config.MATERIAL_COLORS['info']};
        }}
        
        /* Material Divider */
        hr {{
            border: none;
            border-top: 1px solid {Config.MATERIAL_COLORS['divider']};
            margin: 1.5rem 0;
        }}
        
        /* Link Styling - Material */
        a {{
            color: {Config.MATERIAL_COLORS['primary']};
            text-decoration: none;
            transition: all 280ms cubic-bezier(0.4, 0, 0.2, 1);
        }}
        
        a:hover {{
            color: {Config.MATERIAL_COLORS['primary_dark']};
            text-decoration: underline;
        }}
        
        /* Code blocks - Material */
        code {{
            font-family: 'Roboto Mono', monospace;
            background: {Config.MATERIAL_COLORS['surface_variant']};
            padding: 0.125rem 0.375rem;
            font-size: 0.875rem;
            border-radius: 4px;
        }}
        
        pre {{
            background: {Config.MATERIAL_COLORS['text_primary']};
            color: white;
            padding: 1rem;
            font-family: 'Roboto Mono', monospace;
            font-size: 0.875rem;
            overflow-x: auto;
            border-radius: 4px;
            box-shadow: 0 2px 1px -1px rgba(0,0,0,0.2), 
                        0 1px 1px 0 rgba(0,0,0,0.14), 
                        0 1px 3px 0 rgba(0,0,0,0.12);
        }}
        
        /* Loading State - Material */
        .loading-skeleton {{
            background: linear-gradient(90deg, {Config.MATERIAL_COLORS['divider']} 25%, {Config.MATERIAL_COLORS['surface_variant']} 50%, {Config.MATERIAL_COLORS['divider']} 75%);
            background-size: 200% 100%;
            animation: loading 1.5s ease-in-out infinite;
        }}
        
        @keyframes loading {{
            0% {{ background-position: 200% 0; }}
            100% {{ background-position: -200% 0; }}
        }}
        
        /* Ripple Effect */
        @keyframes ripple {{
            to {{
                transform: scale(4);
                opacity: 0;
            }}
        }}
        
        /* Focus visible for accessibility */
        *:focus-visible {{
            outline: 2px solid {Config.MATERIAL_COLORS['primary']};
            outline-offset: 2px;
        }}
        
        /* Disabled state */
        button:disabled, input:disabled {{
            opacity: 0.38;
            cursor: not-allowed;
        }}
        
        /* FAB (Floating Action Button) */
        .fab {{
            width: 56px;
            height: 56px;
            border-radius: 50%;
            background: {Config.MATERIAL_COLORS['secondary']};
            color: white;
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: 0 3px 5px -1px rgba(0,0,0,0.2), 
                        0 6px 10px 0 rgba(0,0,0,0.14), 
                        0 1px 18px 0 rgba(0,0,0,0.12);
            transition: all 280ms cubic-bezier(0.4, 0, 0.2, 1);
        }}
        
        .fab:hover {{
            box-shadow: 0 5px 5px -3px rgba(0,0,0,0.2), 
                        0 8px 10px 1px rgba(0,0,0,0.14), 
                        0 3px 14px 2px rgba(0,0,0,0.12);
            transform: scale(1.05);
        }}
        </style>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def render_header():
        """Render header dengan Google Material Design"""
        st.markdown(f"""
        <div style='padding: 1.5rem 0;'>
            <div style='color: {Config.MATERIAL_COLORS['text_secondary']}; font-size: 0.875rem; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.5rem;'>
                Sistem Regulasi Pengadaan
            </div>
            <h1 style='color: {Config.MATERIAL_COLORS['primary']}; font-weight: 400; margin: 0;'>
                ReguBot
            </h1>
            <p style='color: {Config.MATERIAL_COLORS['text_secondary']}; font-size: 1.125rem; margin-top: 0.75rem; font-weight: 300;'>
                Asisten cerdas untuk regulasi pengadaan barang dan jasa
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def render_sidebar_logo():
        """Render logo di sidebar dengan Material Design"""
        st.markdown(
            f"<div style='text-align:center; padding: 1.5rem 0;'>"
            "<img src='https://raw.githubusercontent.com/YonaFr/ReguBot/main/PBJ.png' "
            f"style='width: 100px; border-radius: 8px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1), 0 2px 4px -1px rgba(0,0,0,0.06);'>"
            "</div>",
            unsafe_allow_html=True
        )
    
    @staticmethod
    def render_document_item(filename: str):
        """Render item dokumen dengan Material styling"""
        st.markdown(
            f"<div class='doc-item'>📄 {filename}</div>",
            unsafe_allow_html=True
        )
    
    @staticmethod
    def render_response_with_badges(response: ResponseData):
        """Render response dengan Material chips"""
        # Warning banner jika ada
        if response.warning:
            st.markdown(
                f"<div class='material-banner'>"
                f"<span class='material-icons' style='margin-right: 0.75rem;'>warning</span>"
                f"<span>{response.warning}</span>"
                f"</div>",
                unsafe_allow_html=True
            )
        
        # Main response
        st.markdown(response.output_text)
        
        # Note jika ada
        if response.note:
            st.markdown(
                f"<div class='material-banner info' style='margin-top: 1rem;'>"
                f"<span class='material-icons' style='margin-right: 0.75rem;'>info</span>"
                f"<span>{response.note}</span>"
                f"</div>",
                unsafe_allow_html=True
            )
    
    @staticmethod
    def render_footer():
        """Render footer dengan Google Material Design"""
        st.markdown(
            f"""
            <div style='text-align:center; padding: 2rem 0 1rem 0; font-size: 0.75rem; color: {Config.MATERIAL_COLORS['text_secondary']};'>
                <div style='margin-bottom: 0.75rem;'>
                    <img src='https://mirrors.creativecommons.org/presskit/icons/cc.svg' style='max-width:1em;max-height:1em;margin:0 0.2em;opacity:0.6;'>
                    <img src='https://mirrors.creativecommons.org/presskit/icons/by.svg' style='max-width:1em;max-height:1em;margin:0 0.2em;opacity:0.6;'>
                    <img src='https://mirrors.creativecommons.org/presskit/icons/nc.svg' style='max-width:1em;max-height:1em;margin:0 0.2em;opacity:0.6;'>
                    <img src='https://mirrors.creativecommons.org/presskit/icons/sa.svg' style='max-width:1em;max-height:1em;margin:0 0.2em;opacity:0.6;'>
                </div>
                <div style='margin: 0.75rem 0; font-family: Roboto, sans-serif; font-weight: 400;'>
                    © 2025 Yona Friantina
                </div>
                <div style='color: {Config.MATERIAL_COLORS['text_hint']}; font-size: 0.75rem; font-weight: 300;'>
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
        MaterialUI.inject_custom_css()
    
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
            MaterialUI.render_sidebar_logo()
            
            st.markdown("### 📂 Upload Dokumen Regulasi")
            st.markdown(
                f"<p style='font-size: 0.875rem; color: {Config.MATERIAL_COLORS['text_secondary']}; margin-bottom: 1rem;'>"
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
                    MaterialUI.render_document_item(filename)
                
                st.markdown("---")
                
                # Status info - Material Card
                st.markdown(
                    f"<div class='material-card'>"
                    f"<div style='color: {Config.MATERIAL_COLORS['success']}; font-weight: 500; margin-bottom: 0.5rem; display: flex; align-items: center;'>"
                    f"<span class='material-icons' style='font-size: 1.25rem; margin-right: 0.5rem;'>check_circle</span>"
                    f"<span>Status: Terverifikasi</span>"
                    f"</div>"
                    f"<div style='font-size: 0.875rem; color: {Config.MATERIAL_COLORS['text_secondary']};'>Jawaban berdasarkan dokumen yang telah diupload</div>"
                    f"</div>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"<div class='material-card' style='border-left: 4px solid {Config.MATERIAL_COLORS['warning']};'>"
                    f"<div style='color: {Config.MATERIAL_COLORS['warning']}; font-weight: 500; margin-bottom: 0.5rem; display: flex; align-items: center;'>"
                    f"<span class='material-icons' style='font-size: 1.25rem; margin-right: 0.5rem;'>warning</span>"
                    f"<span>Status: Belum Ada Upload</span>"
                    f"</div>"
                    f"<div style='font-size: 0.875rem; color: {Config.MATERIAL_COLORS['text_secondary']};'>Upload regulasi untuk jawaban terverifikasi</div>"
                    f"</div>",
                    unsafe_allow_html=True
                )
            
            st.markdown("---")
            MaterialUI.render_footer()
    
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
                        MaterialUI.render_response_with_badges(response)
                        
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
        
        MaterialUI.render_header()
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
