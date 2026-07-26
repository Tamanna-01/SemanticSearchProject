#This file contains the EmbeddingModel class that uses the SentenceTransformer library to 
# convert user queries into embedding vectors for semantic search.

from sentence_transformers import SentenceTransformer

class EmbeddingModel:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Downloads and loads the free embedding model."""
        self.model = SentenceTransformer(model_name)

    def encode(self, text: str):
        """Turns the user's text question into an embedding vector."""
        return self.model.encode(text)