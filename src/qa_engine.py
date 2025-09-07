from .pdf_reader import extract_text_from_pdf
from .text_processor import chunk_text
from .vector_store import VectorStore

class QASystem:
    def __init__(self):
        # Initialize the vector store
        self.vector_store = VectorStore()

    def process_pdf(self, pdf_path):
        """
        Extract text from PDF, split into chunks, and store embeddings.
        """
        text = extract_text_from_pdf(pdf_path)
        chunks = chunk_text(text)
        self.vector_store.build_store(chunks)

    def answer_query(self, query):
        """
        Search for the query in the document.
        Returns a friendly message if no relevant passages are found.
        """
        results = self.vector_store.search(query)

        if results:
            best_answer = results[0][0]
        else:
            best_answer = (
                "❌ Sorry, I couldn't find an answer in the document. "
                "Please check your question or try another related to the document content."
            )

        return best_answer, results
