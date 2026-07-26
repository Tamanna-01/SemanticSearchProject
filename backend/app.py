from data_handler import DataLoader
from model_handler import EmbeddingModel
from search_engine import SemanticSearcher

def display_results(results):
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


if __name__ == "__main__":
    DATASET_FILE = "backend\\embedding_index_3m.json"

    # 1. Loading the dataset
    loader = DataLoader(DATASET_FILE)

    # 2. Initializing the search engine with the loaded data
    searcher = SemanticSearcher(loader.data, loader.normalized_embeddings)

    # 3. Loading the free embedding model
    model = EmbeddingModel()

    # 4. Defining the query and getting the embedding
    user_query = "Can you use RStudio with Azure ML?"
    print(f"\nQuery: '{user_query}'")
    
    query_vec = model.encode(user_query)

    # 5. Performing the search and displaying the results
    top_results = searcher.search(query_vector=query_vec, top_k=5)
    display_results(top_results)