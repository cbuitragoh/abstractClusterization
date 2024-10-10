import pandas as pd
import joblib
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import numpy as np
import os
from dotenv import load_dotenv
load_dotenv()

# Load abstracts
def load_abstracts_as_list():
    """
    Load abstracts from a CSV file using environment variable
    and return them as a list.
    """
    return pd.read_csv(
        os.getenv("FINAL_DATA_PATH")
    )['AbstractNarration'].tolist()

# Generate embeddings from desired model
def generate_embeddings(
        abstracts: list[str],
        model_name: str
):
    """
    Generate embeddings from a list of abstracts using 
    the SentenceTransformer model.

    Args:
        abstracts (list): A list of abstract texts.

    Returns:
        numpy.ndarray: An array of embeddings 
            corresponding to the input abstracts.
    """
    model = SentenceTransformer(model_name)
    embeddings = model.encode(abstracts)

    return model, embeddings


# Clustering simple approach defining arbitrary cluster number
def create_kmeans_model(
        embeddings: np.ndarray,
        num_clusters: int = 5
):
    """
    Creates a KMeans clustering model with the specified number of clusters
    and fits it to the given embeddings.

    Args:
        embeddings (numpy.ndarray): The embeddings to cluster.
        num_clusters (int): The number of clusters to create.

    Returns:
        sklearn.cluster.KMeans: The fitted KMeans model.
        labels: numpy.ndarray
    """

    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(embeddings)
    labels = kmeans.fit_predict(embeddings)

    return kmeans, labels


# Save the model and the embeddings
def save_models(kmeans, model):
    joblib.dump(kmeans, f'{os.getenv("MODELS_PATH")}\kmeans_model.pkl')
    joblib.dump(model, f'{os.getenv("MODELS_PATH")}\sentence_model.pkl')


# Save the abstracts and their cluster labels
# to a CSV file using environment variable
def save_clustered_abstracts(data, labels):
    clustered_abstracts = pd.DataFrame(columns=['AbstractNarration', 'label'])
    clustered_abstracts['AbstractNarration'] = data
    clustered_abstracts['label'] = labels
    clustered_abstracts.to_csv(os.getenv("CLUSTERED_ABSTRACTS_PATH"), index=False)



if __name__ == "__main__":
    # Load abstracts
    abstracts = load_abstracts_as_list()

    # Generate embeddings
    model, embeddings = generate_embeddings(abstracts, 'all-MiniLM-L6-v2')

    # Clustering simple approach defining arbitrary cluster number
    kmeans, labels = create_kmeans_model(embeddings)

    # Save the model and the embeddings
    save_models(kmeans, model)

    # Save the abstracts and their cluster labels 
    # to a CSV file using environment variable
    save_clustered_abstracts(abstracts, labels)

    print("training finished successfully!")
    
