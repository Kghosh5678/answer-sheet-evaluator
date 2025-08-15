# app.py

import streamlit as st
import pytesseract
import pdfplumber
from PIL import Image
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re

# Streamlit config
st.set_page_config(page_title="📚 Answer Sheet Evaluator", layout="centered")
st.title("📚 Answer Sheet Evaluation App")

# Load sentence transformer
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# --- Utility Functions ---

def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text.strip()

def extract_text_from_image(file):
    image = Image.open(file)
    return pytesseract.image_to_string(image).strip()

def get_text_from_files(files):
    all_text = ""
    for file in files:
        if file.name.endswith('.pdf'):
            all_text += extract_text_from_pdf(file) + "\n"
        else:
            all_text += extract_text_from_image(file) + "\n"
    return all_text.strip()

def split_answers_by_question(text):
    pattern = r'(?<=\n|^)(\d+)\.\s'
    parts = re.split(pattern, text)
    qna = {}
    for i in range(1, len(parts)-1, 2):
        q_num = parts[i].strip()
        ans = parts[i+1].strip()
        if ans:
            qna[q_num] = ans
    return qna

def compare_answers(model_qna, student_qna):
    results = []
    total_similarity = 0
    count = 0

    for q_num in model_qna:
        if q_num in student_qna:
            model_ans = model_qna[q_num]
            student_ans = student_qna[q_num]

            embeddings = model.encode([model_ans, student_ans])
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            percent = round(similarity * 100, 2)

            results.append((q_num, percent))
            total_similarity += percent
            count += 1
        else:
            results.append((q_num, "❌ Not Answered"))

    average = round(total_similarity / count, 2) if count > 0 else 0
    return results, average

# --- SESSION STATE ---
if "model_qna" not in st.session_state:
    st.session_state.model_qna = None

# --- Upload Model Answer ---
st.header("Step 1: Upload Model Answer Sheet (One-Time)")
model_files = st.file_uploader("Upload one or more pages (PDF or images)", type=["pdf", "png", "jpg", "jpeg"], accept_multiple_files=True, key="model")

if model_files and st.button("📖 Process Model Answer"):
    with st.spinner("Extracting model answers..."):
        model_text = get_text_from_files(model_files)
        model_qna = split_answers_by_question(model_text)
        st.session_state.model_qna = model_qna

    st.success("Model answer saved. Ready to evaluate students.")
    st.text_area("📄 Extracted Model Answers", model_text, height=300)

elif st.session_state.model_qna:
    st.info("✅ Model answer already uploaded. You can now upload student answers.")

# --- Upload Student Answer ---
if st.session_state.model_qna:
    st.header("Step 2: Upload Student Answer Sheet")

    student_files = st.file_uploader("Upload student answer (PDF or images)", type=["pdf", "png", "jpg", "jpeg"], accept_multiple_files=True, key="student")

    if student_files and st.button("🧮 Evaluate Student"):
        with st.spinner("Extracting student answers..."):
            student_text = get_text_from_files(student_files)
            student_qna = split_answers_by_question(student_text)

        st.text_area("📝 Extracted Student Answers", student_text, height=300)

        st.subheader("🔍 Evaluation Result (Per Question)")
        results, avg = compare_answers(st.session_state.model_qna, student_qna)

        for q_num, score in results:
            if isinstance(score, (float, int)):
                st.metric(f"Question {q_num}", f"{score}%")
            else:
                st.warning(f"Question {q_num}: {score}")

        st.success(f"🎯 Final Suggested Marks: {avg / 100 * 10:.2f} / 10")

