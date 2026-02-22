import numpy as np
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer

# Load the model once when this file is imported
print("Loading all-mpnet-base-v2 model...")
embedding_model = SentenceTransformer("all-mpnet-base-v2")


def cluster_unique_words(keywords_df, n_clusters):
    """Generates embeddings and clusters the unique words."""
    words = keywords_df["keywords_list"].tolist()
    embeddings = embedding_model.encode(words, convert_to_tensor=False)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    keywords_df["cluster"] = kmeans.fit_predict(embeddings)

    return kmeans, embeddings, keywords_df


def get_top_keywords_per_cluster(kmeans, embeddings, keywords_df, top_n=5):
    """Extracts the top representative words for each cluster."""
    cluster_top_words = {}

    for cluster_id in range(kmeans.n_clusters):
        members_idx = np.where(keywords_df["cluster"] == cluster_id)[0]
        member_vecs = embeddings[members_idx]

        if len(members_idx) == 0:
            cluster_top_words[cluster_id] = []
            continue

        dists = np.linalg.norm(
            member_vecs - kmeans.cluster_centers_[cluster_id], axis=1
        )
        sorted_idx = members_idx[np.argsort(dists)]

        top_words = keywords_df.iloc[sorted_idx[:top_n]]["keywords_list"].tolist()
        cluster_top_words[cluster_id] = top_words

    return cluster_top_words
