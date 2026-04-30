
import streamlit as st
import pandas as pd
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="AI Resume Analyzer", page_icon="📄", layout="wide")

SKILLS = [
    "python", "sql", "machine learning", "scikit-learn", "tensorflow", "pytorch",
    "nlp", "llm", "rag", "prompt engineering", "data pipelines", "statistics",
    "pandas", "numpy", "aws", "azure", "gcp", "communication", "deep learning",
    "streamlit", "vector database", "faiss", "langchain"
]

def extract_pdf_text(uploaded_file):
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

def extract_skills(text):
    text_lower = text.lower()
    return sorted([skill for skill in SKILLS if skill in text_lower])

def match_score(resume_text, job_text):
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform([resume_text, job_text])
    score = cosine_similarity(matrix[0:1], matrix[1:2])[0][0]
    return round(score * 100, 2)

def generate_recommendations(missing_skills):
    if not missing_skills:
        return ["Your resume already matches the main job requirements well."]
    return [
        f"Add a project or bullet showing hands-on experience with {skill}."
        for skill in missing_skills[:8]
    ]

st.title("📄 AI Resume Analyzer with RAG-style Skill Matching")
st.write("Upload your resume and paste a job description to get a match score, missing skills, and improvement suggestions.")

col1, col2 = st.columns(2)

with col1:
    resume_file = st.file_uploader("Upload Resume PDF", type=["pdf"])

with col2:
    job_description = st.text_area("Paste Job Description", height=300)

if st.button("Analyze Resume"):
    if not resume_file or not job_description.strip():
        st.warning("Please upload a resume and paste a job description.")
    else:
        resume_text = extract_pdf_text(resume_file)

        resume_skills = extract_skills(resume_text)
        job_skills = extract_skills(job_description)
        missing_skills = sorted(list(set(job_skills) - set(resume_skills)))

        score = match_score(resume_text, job_description)

        st.subheader("✅ Match Score")
        st.metric("Resume Match", f"{score}%")

        st.subheader("Resume Skills Found")
        st.write(", ".join(resume_skills) if resume_skills else "No tracked skills found.")

        st.subheader("Job Skills Found")
        st.write(", ".join(job_skills) if job_skills else "No tracked skills found.")

        st.subheader("Missing Skills")
        st.write(", ".join(missing_skills) if missing_skills else "No major missing skills detected.")

        st.subheader("Improvement Suggestions")
        for rec in generate_recommendations(missing_skills):
            st.write(f"- {rec}")

        st.subheader("Resume Bullet Suggestions")
        if "rag" in missing_skills or "llm" in missing_skills:
            st.write("- Built a RAG-based resume analyzer using Python, TF-IDF similarity, skill extraction, and Streamlit.")
        if "python" in job_skills:
            st.write("- Used Python to process resume/job text and calculate similarity-based match scoring.")
        if "communication" in job_skills:
            st.write("- Presented technical findings through a simple dashboard for recruiter-friendly interpretation.")
