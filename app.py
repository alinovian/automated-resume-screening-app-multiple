import os
import streamlit as st
import pickle
import docx
import PyPDF2
import re
import time
from datetime import datetime

# Load Pre-Trained Model, TF-IDF vectorizer, Label Encoder
svm_model = pickle.load(open('classifier_svm.pkl', 'rb'))
tfidf = pickle.load(open('tfidf_model.pkl', 'rb'))
le = pickle.load(open('label_encoder.pkl', 'rb'))


# Function to clean resume text
def cleanResume(text):
    cleanText = text.lower()
    cleanText = re.sub(r'http\S+|www\S+', '', cleanText)
    cleanText = re.sub(r'\b(rt|cc)\b', '', cleanText)
    cleanText = re.sub(r'#\S+', '', cleanText)
    cleanText = re.sub(r'@\S+', '', cleanText)
    cleanText = re.sub(r'[!"#$%&\'()*+,-./:;<=>?@\[\]^_`{|}~]', ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7F]+', ' ', cleanText)
    cleanText = re.sub(r'\d+', '', cleanText)
    cleanText = re.sub(r'\s+', ' ', cleanText).strip()
    return cleanText


# Function to extract text from PDF
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ''
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


# Function to extract text from DOCX
def extract_text_from_docx(file):
    doc = docx.Document(file)
    text = ''
    for paragraph in doc.paragraphs:
        text += paragraph.text + '\n'
    return text


# Function to extract text from TXT with explicit encoding handling
def extract_text_from_txt(file):
    try:
        text = file.read().decode('utf-8')
    except UnicodeDecodeError:
        text = file.read().decode('latin-1')
    return text


# Function to handle file upload and extraction
def handle_file_upload(uploaded_file):
    file_extension = uploaded_file.name.split('.')[-1].lower()
    if file_extension == 'pdf':
        text = extract_text_from_pdf(uploaded_file)
    elif file_extension == 'docx':
        text = extract_text_from_docx(uploaded_file)
    elif file_extension == 'txt':
        text = extract_text_from_txt(uploaded_file)
    else:
        raise ValueError("Unsupported file type. Please upload a PDF, DOCX, or TXT file.")
    return text


# Function to predict the category of a resume
def pred(input_resume):
    cleaned_text = cleanResume(input_resume)
    vectorized_text = tfidf.transform([cleaned_text])
    vectorized_text = vectorized_text.toarray()
    predicted_category = svm_model.predict(vectorized_text)
    predicted_category_name = le.inverse_transform(predicted_category)
    return predicted_category_name[0]


# Streamlit app layout
def main():
    st.set_page_config(page_title="Automated Resume Screening App", page_icon="🚀", layout="wide")

    st.markdown(
        "<h1 style='text-align: center; color: #4CAF50;'>Automated Resume Category Prediction</h1>",
        unsafe_allow_html=True,
    )
    st.markdown("<p style='text-align: center;'>Upload multiple resumes in PDF, DOCX, or TXT format to predict and organize them by job category.</p>", unsafe_allow_html=True)

    # File upload section (multiple files)
    uploaded_files = st.file_uploader("Upload Resumes", type=["pdf", "docx", "txt"], accept_multiple_files=True)

    if uploaded_files:
        # Create a folder to store categorized resumes
        base_folder = "Categorized_Resumes"
        if not os.path.exists(base_folder):
            os.makedirs(base_folder)

        st.markdown("### 📜 File Processing and Categorization")
        progress = st.progress(0)  # Progress bar

        for i, uploaded_file in enumerate(uploaded_files):
            try:
                # Extract text from file
                resume_text = handle_file_upload(uploaded_file)

                # Predict category
                category = pred(resume_text)

                # Create category folder if not exists
                category_folder = os.path.join(base_folder, category)
                if not os.path.exists(category_folder):
                    os.makedirs(category_folder)

                # Ensure file name uniqueness
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                original_name = uploaded_file.name
                save_path = os.path.join(category_folder, f"{timestamp}_{original_name}")

                # Save file to the corresponding folder
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                st.success(f"File '{original_name}' successfully categorized as '{category}' and saved to '{category_folder}'.")

            except Exception as e:
                st.error(f"Error processing file '{uploaded_file.name}': {str(e)}")

            # Update progress bar
            progress.progress((i + 1) / len(uploaded_files))

        st.success("All files have been processed successfully!")


if __name__ == "__main__":
    main()