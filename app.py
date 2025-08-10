# app.py

import streamlit as st
import pytesseract
import pdfplumber
from PIL import Image
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="📚 Answer Sheet Evaluator", layout="centered")
st.title("📚 Answer Sheet Evaluation App")

# Load the sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# --- Text extraction functions ---
def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text.strip()

def extract_text_from_image(file):
    image = Image.open(file)
    return pytesseract.image_to_string(image).strip()

def get_text(file):
    if file.name.endswith('.pdf'):
        return extract_text_from_pdf(file)
    else:
        return extract_text_from_image(file)

def compare_answers(model_answer, student_answer):
    embeddings = model.encode([model_answer, student_answer])
    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    return round(similarity * 100, 2)

# --- Upload Model Answer ---
st.header("Step 1: Upload Teacher's Model Answer")
model_file = st.file_uploader("Upload PDF or Image", type=["pdf", "png", "jpg", "jpeg"])

if not model_file:
    model_file = st.camera_input("Or capture photo of model answer")

if model_file:
    with st.spinner("Extracting text from model answer..."):
        model_answer_text = get_text(model_file)
    st.success("Model answer uploaded successfully.")
    st.text_area("📖 Extracted Model Answer:", model_answer_text, height=200)

# --- Upload Student Answer ---
st.header("Step 2: Upload Student Answer Sheet")
student_file = st.file_uploader("Upload student's answer", type=["pdf", "png", "jpg", "jpeg"], key="student")

if not student_file:
    student_file = st.camera_input("Or capture photo of student answer")

if student_file:
    with st.spinner("Extracting text from student answer..."):
        student_answer_text = get_text(student_file)
    st.success("Student answer uploaded.")
    st.text_area("📝 Extracted Student Answer:", student_answer_text, height=200)

# --- Compare and Display ---
if st.button("🧮 Evaluate"):
    if model_file and student_file:
        similarity_score = compare_answers(model_answer_text, student_answer_text)       
        st.metric(label="🔍 Similarity Score", value=f"{similarity_score}%")
        st.success(f"🎯 Suggested Marks: {(similarity_score / 100) * 10:.2f} / 10")
    else:
        st.warning("Please upload both model and student answer files.")
