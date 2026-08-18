import streamlit as st
import matplotlib.pyplot as plt
import re
import pandas as pd
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from io import BytesIO

st.set_page_config(page_title="AI Resume ATS Pro", layout="wide")

st.title("📄 AI Resume ATS Pro System")
st.markdown("### Advanced Resume Evaluation & Intelligent Candidate Screening")

#File Upload

resume_files = st.file_uploader(
    "Upload Multiple Resumes (PDF or TXT)",
    type=["pdf", "txt"],
    accept_multiple_files=True
)

jd_file = st.file_uploader("Upload Job Description (TXT)", type=["txt"])

threshold = st.slider("Set Shortlisting Threshold (%)", 0, 100, 60)

#Text Extraction

def extract_text(file):
    if file.type == "application/pdf":
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text.lower()
    else:
        return file.read().decode("utf-8").lower()

#Resume Validation

def is_valid_resume(text):
    resume_keywords = [
        "education", "experience", "skills", "projects",
        "internship", "certification", "objective", "summary"
    ]
    score = sum([1 for word in resume_keywords if word in text])
    return score >= 3  # minimum 3 resume sections required

#Experience Detection

def detect_experience(text):
    years = re.findall(r'(\d+)\s+years?', text)
    if years:
        return max([int(y) for y in years])
    return 0

#Project Count

def count_projects(text):
    keywords = ["project", "developed", "built", "implemented"]
    count = 0
    for word in keywords:
        count += text.count(word)
    return count

#NLP Similarity

def calculate_similarity(resume, jd):
    vectorizer = TfidfVectorizer(stop_words="english")
    vectors = vectorizer.fit_transform([resume, jd])
    similarity = cosine_similarity(vectors[0:1], vectors[1:2])
    return float(similarity[0][0]) * 100

#Logic

if resume_files and jd_file:

    jd_text = extract_text(jd_file)
    results = []
    rejected_files = []

    for resume_file in resume_files:

        resume_text = extract_text(resume_file)

        #Validate Resume
        if not is_valid_resume(resume_text):
            rejected_files.append(resume_file.name)
            continue

        similarity_score = calculate_similarity(resume_text, jd_text)
        experience_years = detect_experience(resume_text)
        project_count = count_projects(resume_text)

        final_score = (
            similarity_score * 0.6 +
            min(experience_years * 5, 20) +
            min(project_count * 2, 20)
        )

        final_score = int(min(final_score, 100))

        results.append({
            "Resume Name": resume_file.name,
            "ATS Score (%)": final_score,
            "Similarity (%)": int(similarity_score),
            "Experience (Years)": experience_years,
            "Project Count": project_count
        })

    #Show Rejected Projects
    if rejected_files:
        st.error("⚠️ Non-Resume Files Detected:")
        for file in rejected_files:
            st.write(f"❌ {file}")

    if results:

        df = pd.DataFrame(results)
        df = df.sort_values(by="ATS Score (%)", ascending=False)
        df.insert(0, "Rank", range(1, len(df) + 1))

        st.markdown("## 📊 Candidate Comparison Dashboard")
        st.dataframe(df, use_container_width=True)

        #Best Candidate

        best_candidate = df.iloc[0]
        st.success(f"🏆 Top Candidate: {best_candidate['Resume Name']} "
                   f"(Score: {best_candidate['ATS Score (%)']}%)")

        #Professional Graph

        st.markdown("## 📈 Candidate Score Distribution")

        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(df["Resume Name"], df["ATS Score (%)"])

        ax.set_ylim(0, 100)
        ax.set_ylabel("ATS Score (%)")
        ax.set_xlabel("Candidates")
        ax.set_title("Resume Ranking Overview")
        ax.tick_params(axis='x', rotation=45)

        #Hihglight Top Candidate
        bars[0].set_linewidth(2)

        st.pyplot(fig)

        #Shortlist

        shortlisted = df[df["ATS Score (%)"] >= threshold]

        st.markdown(f"## 🎯 Shortlisted Candidates (≥ {threshold}%)")

        if not shortlisted.empty:
            st.success(f"{len(shortlisted)} Candidates Selected")

            st.dataframe(shortlisted, use_container_width=True)

            #Convert to Excel
            output = BytesIO()
            shortlisted.to_excel(output, index=False, engine='xlsxwriter')
            output.seek(0)

            st.download_button(
                label="📥 Download Shortlisted Candidates (Excel)",
                data=output,
                file_name="Shortlisted_Candidates.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        else:
            st.warning("No candidates meet the selected threshold.")
