# This file loads the JSON dataset and prepares the numbers for searching


import json
import numpy as np

class DataLoader:
    def __init__(self, dataset_path: str):
        print("Loading dataset...")
        with open(dataset_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        print(f"Loaded {len(self.data)} video segments.")

        # Extracting embeddings into a NumPy array
        self.embeddings = np.array([item["ada_v2"] for item in self.data], dtype=np.float32)

        # Pre-normalizing database embeddings for fast cosine similarity 
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        self.normalized_embeddings = self.embeddings / norms