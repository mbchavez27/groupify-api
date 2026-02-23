import pandas as pd
from services.nlp_processor import setup_nltk, prepare_dataframe
from services.ml_model import (
    cluster_unique_words,
    get_top_keywords_per_cluster,
    embedding_model,
)
from services.matchmaker import assign_members


def run_pipeline(csv_path: str, text_column: str, num_clusters: int):
    print("1. Initializing NLTK...")
    setup_nltk()

    print(f"2. Loading data from {csv_path}...")
    dataset = pd.read_csv(csv_path, on_bad_lines="warn")

    print("3. Cleaning text and extracting keywords...")
    processed_df, unique_keywords_df = prepare_dataframe(dataset, text_column)

    print(f"4. Clustering unique words into {num_clusters} clusters...")
    kmeans, word_embeddings, clustered_words_df = cluster_unique_words(
        unique_keywords_df, num_clusters
    )

    print("5. Generating cluster labels...")
    house_labels = get_top_keywords_per_cluster(
        kmeans, word_embeddings, clustered_words_df
    )
    for cluster_id, words in house_labels.items():
        print(f"   Cluster {cluster_id}: {', '.join(words)}")

    print("6. Assigning members to clusters (Load Balancing)...")
    final_df = assign_members(processed_df, embedding_model, kmeans, max_diff=20)

    final_df["cluster_label"] = final_df["assigned_cluster"].apply(
        lambda x: ", ".join(house_labels.get(x, []))
    )

    output_filename = "output/" + "clustered_members_output.csv"
    final_df.to_csv(output_filename, index=False)
    print(f"\nPipeline Complete! Results saved to {output_filename}")


if __name__ == "__main__":
    TARGET_CSV = "data/" + "lscs_members.csv"
    COLUMN_NAME = "interests"
    NUMBER_OF_CLUSTERS = 5

    run_pipeline(TARGET_CSV, COLUMN_NAME, NUMBER_OF_CLUSTERS)
