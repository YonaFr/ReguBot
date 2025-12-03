# ReguBot | Regulasi ChatBot 🤖

**ReguBot** adalah aplikasi berbasis **Streamlit** yang dirancang untuk membantu pengguna menjawab pertanyaan terkait regulasi pengadaan barang/jasa di Indonesia. Aplikasi ini dapat membaca dokumen PDF, memprosesnya, dan menjawab pertanyaan menggunakan model **Google Gemini** sambil tetap mematuhi regulasi yang berlaku.

---

## Fitur

* **Unggah & Proses Dokumen PDF**: Mendukung unggah beberapa dokumen PDF sekaligus dan memproses teks dari seluruh halaman.
* **Chat Interaktif**: Menjawab pertanyaan terkait regulasi pengadaan barang/jasa secara real-time.
* **Validasi Pasal & Regulasi**: Menandai pasal yang tidak valid atau regulasi yang dikecualikan agar jawaban tetap akurat.
* **Penyimpanan Status File**: Menyimpan daftar dokumen yang sudah diproses sehingga tidak hilang saat reload aplikasi.
* **Bersihkan Chat & Jejak Digital**: Memudahkan membersihkan riwayat chat dan file yang sudah diproses.

---

## Instalasi

1. **Clone repository ini:**

```bash
git clone https://github.com/YonaFr/ReguBot.git
cd ReguBot
```

2. **Buat virtual environment (opsional tapi disarankan):**

```bash
python -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate      # Windows
```

3. **Install dependencies:**

```bash
pip install -r requirements.txt
```

4. **Tambahkan file `.env`** dengan API key Google Gemini:

```
GEMINI_API_KEY=your_api_key_here
```

---

## Cara Menggunakan

1. Jalankan Streamlit:

```bash
streamlit run app.py
```

2. Buka browser dan akses `http://localhost:8501`.

3. Gunakan sidebar untuk:

   * Mengunggah dokumen PDF.
   * Memproses dokumen.
   * Membersihkan chat history.

4. Ketik pertanyaan terkait regulasi pada kotak chat, ReguBot akan menjawab berdasarkan dokumen dan regulasi yang tersedia.

---

## Struktur Regulasi

ReguBot sudah dilengkapi daftar regulasi beserta jumlah pasal untuk validasi, antara lain:

* UU No 3 Tahun 2024
* UU No 6 Tahun 2014
* PP No 11 Tahun 2019
* PP Nomor 8 Tahun 2016
* Peraturan Presiden Nomor 12 Tahun 2021, dll.

Regulasi tertentu dapat **dikecualikan** agar tidak disertakan dalam jawaban.

---

## Catatan Penggunaan

* Jawaban yang diberikan ReguBot **harus selalu diverifikasi** terutama jika berkaitan dengan keputusan hukum penting.
* ReguBot **tidak membuat asumsi tentang pasal atau ayat** yang tidak ada dalam regulasi.
* Jika pertanyaan tidak terkait regulasi, ReguBot tetap dapat memberikan jawaban berbasis sumber eksternal.

---

## Kontribusi

Kontribusi sangat diterima!
Jika ingin menambah fitur, regulasi, atau memperbaiki bug, silakan buat **pull request**.

---

## Lisensi

* [Creative Commons BY-NC-SA](https://creativecommons.org/licenses/by-nc-sa/4.0/)
* © 2025. Yona Friantina. Some rights reserved.

---

## Teknologi

* Python 3.10+
* [Streamlit](https://streamlit.io/)
* [PyPDF2](https://pypi.org/project/PyPDF2/)
* [LangChain](https://www.langchain.com/)
* Google Gemini API
* dotenv untuk manajemen environment variables

