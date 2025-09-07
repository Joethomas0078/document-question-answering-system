from sentence_transformers import SentenceTransformer, util
import torch

class VectorStore:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        # Force CPU to avoid Streamlit Cloud GPU issues
        self.device = "cpu"
        self.model = SentenceTransformer(model_name, device=self.device)

        self.chunks = []
        self.embeddings = None

    def build_store(self, chunks):
        """Store text chunks and their embeddings."""
        if not chunks:
            self.chunks = []
            self.embeddings = None
            return

        self.chunks = chunks
        self.embeddings = self.model.encode(
            chunks,
            convert_to_tensor=True,
            device=self.device
        )

    def search(self, query, top_k=3, score_threshold=0.3):
        """Search the vector store and return results above score_threshold."""
        if self.embeddings is None or len(self.embeddings) == 0:
            return []

        query_embedding = self.model.encode(
            query,
            convert_to_tensor=True,
            device=self.device
        )
        scores = util.pytorch_cos_sim(query_embedding, self.embeddings)[0]
        top_results = torch.topk(scores, k=min(top_k, len(scores)))

        results = []
        for score, idx in zip(top_results.values, top_results.indices):
            if float(score) >= score_threshold:
                results.append((self.chunks[idx], float(score)))

        return results
