import json
import numpy as np
from typing import List, Dict, Any

# Import the free model tool
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False


class SemanticSearchApp:
    def __init__(self, dataset_path: str):
        """Loads the JSON dataset and prepares the numbers for searching."""
        print("Loading dataset...")
        with open(dataset_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        print(f"Loaded {len(self.data)} video segments.")

        # Extract embeddings into a NumPy array
        self.embeddings = np.array([item["ada_v2"] for item in self.data], dtype=np.float32)

        # Pre-normalize database embeddings for fast cosine similarity 
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        self.normalized_embeddings = self.embeddings / norms

    @staticmethod
    def construct_youtube_link(video_id: str, seconds: int) -> str:
        """Creates the clickable YouTube link."""
        return f"https://www.youtube.com/watch?v={video_id}&t={seconds}s"

    def compute_cosine_similarity(self, query_vector: np.ndarray) -> np.ndarray:
        """Calculates how closely the user's question matches the videos."""
        query_vector = np.array(query_vector, dtype=np.float32)

        # FIX: Pad query vector with zeros if dimensions don't match dataset (1536)
        target_dim = self.normalized_embeddings.shape[1]
        if len(query_vector) < target_dim:
            query_vector = np.pad(query_vector, (0, target_dim - len(query_vector)))
        elif len(query_vector) > target_dim:
            query_vector = query_vector[:target_dim]

        norm = np.linalg.norm(query_vector)
        if norm > 0:
            query_vector = query_vector / norm

        # The dot product finds similarity scores
        similarities = np.dot(self.normalized_embeddings, query_vector)
        return similarities

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """Finds the top 5 most relevant video segments."""
        similarities = self.compute_cosine_similarity(query_embedding)

        # Get top 5 highest scores
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            item = self.data[idx]
            results.append({
                "title": item.get("title", "N/A"),
                "speaker": item.get("speaker", "N/A"),
                "summary": item.get("summary", "N/A"),
                "timestamp": item.get("start", "00:00:00"),
                "seconds": item.get("seconds", 0),
                "similarity_score": float(similarities[idx]),
                "youtube_link": self.construct_youtube_link(item["videoId"], item.get("seconds", 0))
            })

        return results

    def display_results(self, results: List[Dict[str, Any]]) -> None:
        """Prints the final results neatly on your screen."""
        print("\n" + "=" * 80)
        print(f"{'TOP SEMANTIC SEARCH RESULTS':^80}")
        print("=" * 80 + "\n")

        for i, res in enumerate(results, 1):
            print(f"Result #{i}")
            print(f"  * Title           : {res['title']}")
            print(f"  * Speaker(s)      : {res['speaker']}")
            print(f"  * Similarity Score: {res['similarity_score']:.4f}")
            print(f"  * Timestamp       : {res['timestamp']}")
            print(f"  * YouTube Link    : {res['youtube_link']}")
            print(f"  * Summary         : {res['summary']}")
            print("-" * 80)


# =====================================================================
# Main Application Run
# =====================================================================

if __name__ == "__main__":
    DATASET_FILE = "backend\\embedding_index_3m.json"

    app = SemanticSearchApp(DATASET_FILE)

    if HAS_SENTENCE_TRANSFORMERS:
        print("\n--- Running Free Model Query Embeddings ---")
        model = SentenceTransformer("all-MiniLM-L6-v2")
        
        user_query = "What are Jupyter Notebooks?"
        print(f"Query: '{user_query}'")
        
        query_vec = model.encode(user_query)
        top_results = app.search(query_embedding=query_vec, top_k=5)
        app.display_results(top_results)
    else:
        print("\nOption A: Pipeline Search Mode")
        sample_query_vector = app.embeddings[0]
        top_results = app.search(query_embedding=sample_query_vector, top_k=5)
        app.display_results(top_results)