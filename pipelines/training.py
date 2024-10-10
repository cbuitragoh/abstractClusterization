import pandas as pd
import joblib
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import numpy as np
import os
from dotenv import load_dotenv
load_dotenv()

if __name__ == "__main__":
    # Load abstracts
    abstracts = pd.read_csv(os.getenv("CSV_DATA_PATH"))['AbstractNarration']

    # Generate embeddings
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(abstracts)

    # Clustering simple approach defining arbitrary cluster number
    num_clusters = 5  
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings)

    # Save the model and the embeddings
    joblib.dump(kmeans, f'{os.getenv("MODELS_PATH")}\kmeans_model.pkl')
    joblib.dump(model, f'{os.getenv("MODELS_PATH")}\sentence_model.pkl')

    # Save the abstracts and their cluster labels to a CSV file using environment variable
    abstracts['label'] = labels
    abstracts.to_csv(os.getenv("CLUSTERED_ASBTRACTS_PATH"), index=False)
    
