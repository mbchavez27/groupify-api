import numpy as np


def assign_members(dataset_sorted, model, kmeans, max_diff=20):
    n_clusters = kmeans.n_clusters
    n_members = len(dataset_sorted)
    embedding_dim = kmeans.cluster_centers_.shape[1]

    vectors_list = []
    for kw_list in dataset_sorted["keywords_list"]:
        if kw_list:
            # Re-encode the member's keywords to compare against cluster centroids
            vec = model.encode(kw_list, convert_to_tensor=False)
            vectors_list.append(
                np.mean(vec, axis=0) if len(vec) > 0 else np.zeros(embedding_dim)
            )
        else:
            vectors_list.append(np.zeros(embedding_dim))

    vectors = np.vstack(vectors_list)
    distances = np.linalg.norm(
        vectors[:, None, :] - kmeans.cluster_centers_[None, :, :], axis=2
    )
    preferences = np.argsort(distances, axis=1)

    assigned = np.full(n_members, -1)
    cluster_counts = np.zeros(n_clusters, dtype=int)

    # Greedy assignment with max difference check
    for member_idx in range(n_members):
        for pref in preferences[member_idx]:
            if cluster_counts[pref] <= min(cluster_counts) + max_diff:
                assigned[member_idx] = pref
                cluster_counts[pref] += 1
                break

        # Fallback
        if assigned[member_idx] == -1:
            min_cluster = np.argmin(cluster_counts)
            assigned[member_idx] = min_cluster
            cluster_counts[min_cluster] += 1

    dataset_sorted["assigned_cluster"] = assigned
    return dataset_sorted
