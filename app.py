import streamlit as st
from src.qa_engine import QASystem
import os

# Initialize the QA system
@st.cache_resource
def init_qa():
    return QASystem()

qa = init_qa()

st.set_page_config(page_title="Document Question Answering", layout="wide")
st.title("📄 Document Question Answering System")

# Upload PDF
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    # Save uploaded PDF to temporary location
    pdf_path = os.path.join("data", uploaded_file.name)
    os.makedirs("data", exist_ok=True)
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success(f"Uploaded {uploaded_file.name}")

    # Process PDF
    qa.process_pdf(pdf_path)
    st.info("PDF processed. You can now ask questions related to the document.")

    # Ask question with a button
    st.markdown("### Ask a question related to the document:")
    if "question" not in st.session_state:
        st.session_state.question = ""

    st.session_state.question = st.text_input(
        "Type your question here:", st.session_state.question
    )

    if st.button("Get Answer"):
        question = st.session_state.question.strip()
        if question:
            try:
                answer, results = qa.answer_query(question)
                st.markdown("### Answer:")
                st.write(answer)

                if results:
                    st.markdown("### Top Relevant Passages:")
                    for i, (text, score) in enumerate(results):
                        st.write(f"**{i+1}.** {text}  _(Score: {score:.2f})_")
            except Exception as e:
                st.error(f"❌ Error: {e}")
        else:
            st.warning("Please enter a question first.")

else:
    st.info("Please upload a PDF to get started.")
