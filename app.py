# app.py

import streamlit as st
import pytesseract
import pdfplumber
from PIL import Image
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
import pandas as pd

# --- CONFIG ---
st.set_page_config(page_title="📚 Answer Sheet Evaluator", layout="centered")
st.title("📚 Answer Sheet Evaluation App")

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# --- SESSION STATE INIT ---
if "model_qna" not in st.session_state:
    st.session_state["model_qna"] = None

if "results" not in st.session_state:
    st.session_state["results"] = []

if "student_evaluated" not in st.session_state:
    st.session_state["student_evaluated"] = False

if "student_form_counter" not in st.session_state:
    st.session_state["student_form_counter"] = 0

if "student_name" not in st.session_state:
    st.session_state["student_name"] = ""

# --- UTILITY FUNCTIONS ---

def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text.strip()

def extract_text_from_image(file):
    image = Image.open(file).convert("L")  # Convert to grayscale
    return pytesseract.image_to_string(image, lang='eng').strip()

def get_text_from_files(files):
    all_text = ""
    files = sorted(files, key=lambda x: x.name)
    for file in files:
        if file.name.endswith('.pdf'):
            all_text += extract_text_from_pdf(file) + "\n"
        else:
            try:
                all_text += extract_text_from_image(file) + "\n"
            except Exception as e:
                st.error(f"❌ Could not read image {file.name}: {e}")
    return all_text.strip()

def split_answers_by_question(text):
    text = text.replace('\r', '').replace('\t', '')    
    pattern = r'(?:^|\n)\s*(?:[Qq](?:uestion)?[\s\-]*)?(\d{1,4})[\)\.\:\-\s]'
    text += "\nQuestion 9999."
    matches = list(re.finditer(pattern, text))
    qna = {}
    for i in range(len(matches) - 1):
        start = matches[i].start()
        end = matches[i + 1].start()
        q_num = matches[i].group(1).lstrip('0')
        answer = text[start:end].strip()
        qna[q_num] = answer
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

# --- STEP 1: Upload Model Answer ---
st.header("Upload Model Answer Sheet")
model_files = st.file_uploader(
    "Upload model answer (PDF or images)",
    type=["pdf", "png", "jpg", "jpeg"],
    accept_multiple_files=True,
    key="model_files"
)

if model_files:
    st.subheader("🖼️ Model Answer Preview")
    model_cols = st.columns(3)
    for idx, file in enumerate(model_files):
        if file.type.startswith("image"):
            with model_cols[idx % 3]:
                st.image(file, caption=file.name, use_column_width=True)

if model_files and st.button("📖 Process Model Answer"):
    with st.spinner("Extracting model answers..."):
        model_text = get_text_from_files(model_files)
        model_qna = split_answers_by_question(model_text)
        st.session_state["model_qna"] = model_qna
    st.success("✅ Model answer saved. Ready to evaluate students.")
elif st.session_state["model_qna"]:
    st.info("✅ Model already uploaded. You may now evaluate students.")

# --- STEP 2: Evaluate Student Answers ---
if st.session_state["model_qna"]:
    st.header("Evaluate Student Answer Sheet")

    if not st.session_state["student_evaluated"]:
        st.session_state["student_name"] = st.text_input(
            "Enter student name or roll number", key="student_name_input"
        )

        student_files = st.file_uploader(
            "Upload student answer (PDF or images)",
            type=["pdf", "png", "jpg", "jpeg"],
            accept_multiple_files=True,
            key=f"student_files_{st.session_state['student_form_counter']}"
        )

        if student_files:
            st.subheader("🖼️ Student Answer Preview")
            student_cols = st.columns(3)
            for idx, file in enumerate(student_files):
                if file.type.startswith("image"):
                    with student_cols[idx % 3]:
                        st.image(file, caption=file.name, use_column_width=True)

        if st.button("🧮 Evaluate Student"):
            if not st.session_state["student_name"]:
                st.warning("Please enter the student's name or ID.")
            elif not student_files:
                st.warning("Please upload the student's answer sheet.")
            else:
                with st.spinner("Extracting and evaluating..."):
                    student_text = get_text_from_files(student_files)
                    student_qna = split_answers_by_question(student_text)
                    results, avg = compare_answers(st.session_state["model_qna"], student_qna)

                    st.subheader(f"🔍 Results for {st.session_state['student_name']}")
                    for q_num, score in results:
                        if isinstance(score, (float, int)):
                            st.metric(f"Q{q_num}", f"{score}%")
                        else:
                            st.warning(f"Q{q_num}: {score}")

                    st.success(f"🎯 Final Suggested Marks: {avg / 100 * 10:.2f} / 10")

                    # Save to session state
                    result_row = {
                        "Student": st.session_state["student_name"],
                        "Total (%)": avg,
                        "Marks (/10)": round(avg / 100 * 10, 2)
                    }
                    for q_num, score in results:
                        result_row[f"Q{q_num}"] = score
                    st.session_state["results"].append(result_row)

                    st.session_state["student_evaluated"] = True
    else:
        st.success("✅ Student evaluated and added to summary.")

# --- Reset Buttons ---
col1, col2 = st.columns(2)

with col1:
    if st.button("➕ Add Next Student (Clear Student Input)"):
        st.session_state["student_name"] = ""
        st.session_state["student_evaluated"] = False
        st.session_state["student_form_counter"] += 1

with col2:
    if st.button("🔄 Reset Entire App (Start Over)"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.experimental_rerun()

# --- STEP 3: Class Summary & Export ---
if st.session_state["results"]:
    st.header("📋 Class Evaluation Summary")
    df = pd.DataFrame(st.session_state["results"])
    st.dataframe(df, use_container_width=True)

    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        "📥 Download Full Result (CSV)",
        data=csv,
        file_name="class_results.csv",
        mime='text/csv'
    )
