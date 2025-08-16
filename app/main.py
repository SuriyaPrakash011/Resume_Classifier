import streamlit as st
import pickle
import PyPDF2
import docx2txt

# Load trained model & vectorizer
model = pickle.load(open("models/resume_model.pkl", "rb"))
vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))

# Function to extract text from resumes
def extract_text(uploaded_file):
    text = ""
    if uploaded_file.name.endswith(".pdf"):
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    elif uploaded_file.name.endswith(".docx"):
        text = docx2txt.process(uploaded_file)
    else:
        st.error("Unsupported file format. Please upload PDF or DOCX.")
    return text

# Streamlit UI
st.title("📄 AI-Powered Resume Screener")
st.write("Upload your resume (PDF or DOCX) and let AI predict your best-fit job role.")

uploaded_file = st.file_uploader("Upload Resume", type=["pdf", "docx"])

if uploaded_file is not None:
    resume_text = extract_text(uploaded_file)

    if resume_text.strip() != "":
        # Transform text
        resume_vec = vectorizer.transform([resume_text])

        # Predict category
        prediction = model.predict(resume_vec)[0]

        st.subheader("✅ Predicted Job Role:")
        st.success(prediction)
    else:
        st.warning("No text could be extracted from this file.")
