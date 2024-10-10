import joblib
from sentence_transformers import SentenceTransformer
import numpy as np
import os
from dotenv import load_dotenv
load_dotenv()

# Load the saved models
kmeans = joblib.load(f'{os.getenv("MODELS_PATH")}/kmeans_model.pkl')
model = joblib.load(f'{os.getenv("MODELS_PATH")}/sentence_model.pkl')

def predict_topic(new_abstracts):
    # Generate embeddings for new abstracts
    new_embeddings = model.encode(new_abstracts)

    # Predict clusters
    predicted_labels = kmeans.predict(new_embeddings)
    
    return predicted_labels

if __name__ == "__main__":
    print("Enter abstracts to classify (type 'exit' to stop):")
    user_input = []
    
    while True:
        abstract = input("Abstract: ")
        if abstract.lower() == 'exit':
            break
        user_input.append(abstract)

    if user_input:
        predicted_labels = predict_topic(user_input)

        # Display predictions
        for abstract, label in zip(user_input, predicted_labels):
            print(f"Abstract: {abstract}\nPredicted Topic: {label}\n")
    else:
        print("No abstracts were provided for prediction.")
