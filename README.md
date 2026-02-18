# 🧾 Receipt Vault & Analyzer

An AI-powered receipt digitization and analytics platform that extracts structured data from receipts using OCR + LLM, stores them in a database, and provides analytics, search, and intelligent chat interaction.

Built with Streamlit, SQLite, Tesseract OCR, and Groq LLM.

---

## 🚀 Features

- 📥 Upload JPG / PNG / PDF receipts
- 🔍 OCR + AI extraction of structured receipt data
- 🎯 Template-based parsing for improved accuracy
- 🧪 Receipt validation system
- 📊 Analytics dashboard with filters
- 🔎 Advanced search and filtering
- 💬 AI chatbot to query receipt data
- 📁 CSV / Excel export
- 🗄 SQLite persistent storage
- ⚡ Optimized processing pipeline

---

## 🧠 AI Capabilities

- Groq LLM integration for smart extraction
- Template-based vendor parsing
- Natural language chat interface
- Intelligent receipt comparison
- Error correction & validation

---

## 🛠 Tech Stack

- Python 3.10+
- Streamlit
- Tesseract OCR
- Groq API (LLM)
- SQLite
- Pandas
- PIL / OpenCV
- spaCy

---

## 📂 Project Structure

```
ReceiptVault/
│
├── app.py                  # Main Streamlit app
├── styles.py               # UI styling
├── receipts.db             # SQLite database
│
└── .streamlit/
    ├── config.toml
    └── secrets.toml
```

---

## ⚙ Installation

### 1. Clone the repository

```bash
git clone https://github.com/Shreyansh2k23/TeamC_InfosysSpringboard_ReceiptDigitizer.git
cd TeamC_InfosysSpringboard_ReceiptDigitizer
```

### 2. Create virtual environment

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install Tesseract OCR

Download from:
https://github.com/tesseract-ocr/tesseract

Then set path in `.streamlit/secrets.toml`:

```
TESSERACT_PATH = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
```

### 5. Add Groq API Key

```
GROQ_API_KEY = "your_api_key_here"
```

⚠ Never upload secrets.toml to GitHub.

---

## ▶ Running the App

```bash
streamlit run app.py
```

Open browser:

```
http://localhost:8501
```

---

## 📊 Milestone 4 Highlights

- Template-based parsing improves accuracy
- Search & filter dashboard
- Query optimization
- AI chatbot integration
- Polished UI & deployment-ready architecture

---

## 🧪 Usage Flow

1. Upload receipts
2. Preprocess & validate
3. Save to database
4. Analyze dashboard
5. Chat with your data
6. Export reports

---

## 🔒 Security Notes

- API keys stored in `.streamlit/secrets.toml`
- Database is local SQLite
- No external data leakage

---

## 📈 Future Improvements

- Cloud deployment
- Multi-user authentication
- Vendor template learning
- Mobile camera capture
- Auto categorization

---

## 👨‍💻 Author
### Infosys Springboard Project – Team C
- Shreyansh Gupta
- Prajanashree
- Prasula Nikhileshwar Reddy
- Majji Meghana
- Vanguri Madhulika


---

## 📜 License

MIT License
