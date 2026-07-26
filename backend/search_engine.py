#This file contains the search engine class that performs semantic search using cosine similarity. 
# It takes a query vector and returns the top matching results from the dataset.

import numpy as np
from typing import List, Dict, Any

class SemanticSearcher:
    def __init__(self, data: List[Dict], normalized_embeddings: np.ndarray):
        self.data = data
        self.normalized_embeddings = normalized_embeddings

    @staticmethod
    def construct_youtube_link(video_id: str, seconds: int) -> str:
        #Creates the clickable YouTube link
        return f"https://www.youtube.com/watch?v={video_id}&t={seconds}s"

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        #Calculates cosine similarity and returns the top matches.
        query_vector = np.array(query_vector, dtype=np.float32)

        # Pad query vector with zeros if dimensions don't match dataset (1536)
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