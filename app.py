import streamlit as st
from src.qa_engine import QASystem
import tempfile
import os

st.set_page_config(page_title="Document Q&A", layout="wide")
st.title("📄 Document Question Answering System")

qa = QASystem()

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
if uploaded_file:
    # Save file to a temporary location instead of data/
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        pdf_path = tmp_file.name

    st.success(f"PDF uploaded successfully: {uploaded_file.name}")
    qa.process_pdf(pdf_path)

    query = st.text_input("Ask a question about the document:")
    if st.button("Get Answer") and query:
        answer, results = qa.answer_query(query)
        st.subheader("Answer:")
        st.write(answer)

        st.subheader("Relevant Passages:")
        for passage, score in results:
            st.write(f"- {passage} (score: {score:.4f})")

