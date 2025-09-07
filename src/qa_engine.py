from .pdf_reader import extract_text_from_pdf
from .text_processor import chunk_text
from .vector_store import VectorStore


class QASystem:
    def __init__(self):
        # Initialize VectorStore (already locked to CPU)
        self.vector_store = VectorStore()

    def process_pdf(self, pdf_path):
        """
        Extract text from PDF, split into chunks, and store embeddings.
        """
        text = extract_text_from_pdf(pdf_path)

        if not text.strip():
            raise ValueError("❌ No text extracted from the PDF. Please upload a valid document.")

        chunks = chunk_text(text)
        if not chunks:
            raise ValueError("❌ Failed to split document into chunks.")

        self.vector_store.build_store(chunks)

    def answer_query(self, query):
        """
        Search for the query in the document.
        Returns a friendly message if no relevant passages are found.
        """
        results = self.vector_store.search(query)

        if results and len(results) > 0:
            best_answer = results[0][0]  # Take the top matching chunk
        else:
            best_answer = (
                "❌ Sorry, I couldn't find an answer in the document. "
                "Please check your question or try another related to the document content."
            )

        return best_answer, results
