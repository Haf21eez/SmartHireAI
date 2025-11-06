import streamlit as st
from PyPDF2 import PdfReader
import docx
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
import nltk

# Download NLTK stopwords once
nltk.download('stopwords', quiet=True)

# -------------------------------------------
# Utility Functions
# -------------------------------------------
def extract_text_from_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def extract_text_from_docx(file):
    doc = docx.Document(file)
    text = " ".join([p.text for p in doc.paragraphs])
    return text

def clean_text(text):
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    stop_words = set(stopwords.words("english"))
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

def get_resume_text(uploaded_file):
    if uploaded_file.name.endswith(".pdf"):
        return extract_text_from_pdf(uploaded_file)
    elif uploaded_file.name.endswith(".docx"):
        return extract_text_from_docx(uploaded_file)
    elif uploaded_file.name.endswith(".txt"):
        return uploaded_file.read().decode("utf-8", errors="ignore")
    else:
        return ""

def compute_similarity(resume_text, jd_text):
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform([resume_text, jd_text])
    similarity = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
    return round(similarity * 100, 2)

def extract_skills(text):
    skill_list = [
        "python", "java", "c++", "javascript", "html", "css", "react", "node",
        "express", "django", "flask", "sql", "mysql", "mongodb", "pandas",
        "numpy", "aws", "docker", "git", "linux", "tailwind", "typescript"
    ]
    found = [s for s in skill_list if s in text.lower()]
    return list(set(found))

def summarize_resume(text, skills, similarity):
    lines = []
    lines.append("Candidate Resume Summary:")
    lines.append(f"- Match Score: {similarity}%")
    if skills:
        lines.append(f"- Skills Found: {', '.join(skills)}")
    else:
        lines.append("- No recognized skills found in resume.")
    if similarity > 80:
        lines.append("- Excellent match for the job description!")
    elif similarity > 50:
        lines.append("- Moderate match, might need minor skill updates.")
    else:
        lines.append("- Low match — consider improving skills alignment.")
    return "\n".join(lines)

# -------------------------------------------
# Streamlit UI
# -------------------------------------------
st.set_page_config(page_title="SmartHire AI", layout="wide")
st.title("🤖 SmartHire AI — Intelligent Resume Screening Assistant")
st.markdown("### Upload a resume and a job description to analyze candidate-job fit.")

# Sidebar
with st.sidebar:
    st.header("Upload Files")
    uploaded_resume = st.file_uploader("Upload Resume", type=["pdf", "docx", "txt"])
    jd_text = st.text_area("Paste Job Description", height=200)
    analyze_btn = st.button("🔍 Analyze Resume")

# -------------------------------------------
# Main logic
# -------------------------------------------
if analyze_btn:
    if not uploaded_resume:
        st.warning("⚠️ Please upload a resume file.")
    elif not jd_text.strip():
        st.warning("⚠️ Please paste a job description.")
    else:
        with st.spinner("Analyzing resume... please wait ⏳"):
            resume_text = get_resume_text(uploaded_resume)
            cleaned_resume = clean_text(resume_text)
            cleaned_jd = clean_text(jd_text)

            similarity = compute_similarity(cleaned_resume, cleaned_jd)
            resume_skills = extract_skills(cleaned_resume)
            jd_skills = extract_skills(cleaned_jd)
            missing_skills = list(set(jd_skills) - set(resume_skills))

            summary = summarize_resume(resume_text, resume_skills, similarity)

        st.success("✅ Analysis Complete!")

        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("📄 Resume Summary")
            st.text_area("AI Summary", value=summary, height=200)

            st.markdown("**🎯 Match Score:**")
            st.progress(similarity / 100)
            st.metric("Match Percentage", f"{similarity}%")

        with col2:
            st.markdown("**🧠 Skills Found in Resume**")
            if resume_skills:
                st.write(", ".join(resume_skills))
            else:
                st.write("No skills detected.")

            st.markdown("**📌 Skills Required (from JD)**")
            if jd_skills:
                st.write(", ".join(jd_skills))
            else:
                st.write("No skills listed in JD.")

            st.markdown("**❌ Missing Skills**")
            if missing_skills:
                st.error(", ".join(missing_skills))
            else:
                st.success("All key JD skills covered!")

        st.download_button(
            label="📥 Download Summary Report",
            data=summary,
            file_name="SmartHireAI_Report.txt",
            mime="text/plain"
        )

st.markdown("---")
st.caption("Developed by Abdul Hafeez — Eaevon IT Service & Internship")
