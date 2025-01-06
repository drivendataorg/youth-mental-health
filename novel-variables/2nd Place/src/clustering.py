from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans,AgglomerativeClustering,DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import numpy as np
import re

# Step 1: Load variables from a text file
def load_variables_from_file(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            variables = [line.strip() for line in file if line.strip()]  # Remove empty lines
        return variables
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return []


def clean_variables(variables):
    cleaned_variables = []
    for var in variables:
        # Step 1: Replace underscores with spaces
        var = var.replace("_", " ")

        # Step 2: Remove special characters (keep alphanumeric and spaces)
        var = re.sub(r"[^a-zA-Z0-9\s]", "", var)

        # Step 3: Convert to lowercase
        var = var.lower()

        # Step 4: Trim extra spaces
        var = " ".join(var.split())

        cleaned_variables.append(var)

    return cleaned_variables

# Step 2: Generate semantic embeddings using Sentence-BERT
def generate_embeddings(variables, model_name='all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(variables)
    return embeddings

# Step 3: Find the best eps
def find_best_eps(embeddings, min_samples=5):
    # Step 1: Compute k-Nearest Neighbors (k = min_samples - 1)
    neighbors = NearestNeighbors(n_neighbors=min_samples).fit(embeddings)
    distances, indices = neighbors.kneighbors(embeddings)

    # Step 2: Sort the distances to the k-th nearest neighbor
    sorted_distances = np.sort(distances[:, -1])

    # Step 3: Plot the distances to find the elbow point
    plt.figure(figsize=(10, 6))
    plt.plot(sorted_distances)
    plt.title("k-NN Distance Plot")
    plt.xlabel("Points sorted by distance to k-th nearest neighbor")
    plt.ylabel("Distance to k-th nearest neighbor")
    plt.grid()
    plt.show()

# Step 4: Clustering with DBSCAN with eps
def cluster_with_precomputed_cosine(embeddings, eps=0.2, min_samples=5):
    # Step 1: Normalize embeddings to ensure unit vectors
    normalized_embeddings = normalize(embeddings, norm='l2', axis=1)

    # Step 2: Compute cosine similarity matrix
    cosine_sim = cosine_similarity(normalized_embeddings)

    # Step 3: Convert cosine similarity to cosine distance and clip to avoid negatives
    cosine_dist = np.clip(1 - cosine_sim, 0, None)  # Clipping to ensure non-negative values

    # Step 4: Use a clustering algorithm that supports precomputed distances
    clustering_model = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
    labels = clustering_model.fit_predict(cosine_dist)
    return labels

# Step 5: Visualize clusters with PCA
def visualize_clusters(embeddings, labels, variables):
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(embeddings)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=labels, cmap='viridis')
    plt.colorbar(scatter)
    plt.title("Clusters of Variables (PCA)")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    
    # Annotate points with variable names
    for i, variable in enumerate(variables):
        plt.annotate(variable, (reduced_embeddings[i, 0], reduced_embeddings[i, 1]), fontsize=8, alpha=0.7)
    
    plt.show()

# Save clusters into files
def save_clusters_to_files(clusters, output_directory="clusters"):
    import os

    # Create the output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)
    
    for cluster, items in clusters.items():
        file_path = os.path.join(output_directory, f"cluster_{cluster}.txt")
        with open(file_path, "w", encoding="utf-8") as file:
            for item in items:
                file.write(f"{item}\n")  # Write each variable to the file
        print(f"Cluster {cluster} saved to {file_path}")


def save_clusters_to_single_file(clusters, output_file="clusters.txt"):
    with open(output_file, "w", encoding="utf-8") as file:
        for cluster, items in clusters.items():
            # Write cluster header
            file.write(f"Cluster {cluster}:\n")
            
            # Write each item in the cluster
            for item in items:
                file.write(f"{item}\n")
            
            # Add a separator for readability
            file.write("\n" + "-" * 40 + "\n\n")
    
    print(f"All clusters saved to {output_file}")

# Main function
def main():
    # File path to the text file containing variables
    file_path = r"../data/outputs/llm_with_clustering/cleaned_variable_list.txt"  # Change this to your file path
    output_file = r"../data/outputs/llm_with_clustering/clusters.txt"

    # Load variables from file
    variables = load_variables_from_file(file_path)
    if not variables:
        return  # Exit if no variables are loaded
    
    print("Variables Loaded:")
    print(variables)

    print("Clean Variables")
    cleanVariables = clean_variables(variables)
    print(cleanVariables)

    # Generate embeddings
    print("\nGenerating embeddings...")
    embeddings = generate_embeddings(cleanVariables)
    

    #find_best_eps(embeddings)

    # Number of clusters (set manually or use heuristics like the elbow method)
    print(f"\nClustering using Precomputed Cosine Distance Matrix")
    labels = cluster_with_precomputed_cosine(embeddings)

    # Visualize clusters
    print("\nVisualizing clusters...")
    visualize_clusters(embeddings, labels, variables)

    # Display and save cluster groups
    clusters = {i: [] for i in range(max(labels) + 1)}  # Create clusters dictionary for valid labels
    clusters["noise"] = []  # Separate group for noise points

    for i, label in enumerate(labels):
        if label == -1:  # Noise points
            clusters["noise"].append(cleanVariables[i])
        else:
            clusters[label].append(cleanVariables[i])

    # Print Cluster Groups
    print("\nCluster Groups:")
    for cluster, items in clusters.items():
        print(f"Cluster {cluster}:")
        print(", ".join(items))
        print()
    
    # Save clusters to files
    print("\nSaving clusters to files...")
    #save_clusters_to_files(clusters)
    save_clusters_to_single_file(clusters,output_file)

# Run the script
if __name__ == "__main__":
    main()