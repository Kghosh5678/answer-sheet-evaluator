# 📚 Answer Sheet Evaluator (Streamlit App)

This Streamlit app compares a teacher's model answer with a student's answer sheet using OCR and semantic similarity.

## Features
- Accepts images or PDFs
- Extracts text using pytesseract and pdfplumber
- Compares meaning using sentence-transformers
- Shows similarity score and suggested marks

## 🚀 Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
