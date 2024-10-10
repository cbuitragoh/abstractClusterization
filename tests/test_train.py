import os
import pandas as pd
import joblib
import pytest
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from pipelines.training import main
from dotenv import load_dotenv
load_dotenv()

@pytest.fixture
def setup_environment(monkeypatch):
    # Set up environment variables for testing
    monkeypatch.setenv("RAW_DATA_PATH", "test_data.csv")
    monkeypatch.setenv("MODELS_PATH", "models")
    monkeypatch.setenv("CLUSTERED_ABSTRACTS_PATH", "clustered_abstracts.csv")
    
    # Create a mock CSV file for testing
    test_data = pd.DataFrame({
        'AbstractNarration': [
            "Research on AI in healthcare.",
            "Study of climate change impacts.",
            "AI and machine learning applications.",
            "Impact of technology on society.",
            "Climate models and predictions."
        ]
    })
    test_data.to_csv("test_data.csv", index=False)

    yield  # This will run the tests

    # Clean up after tests
    os.remove("test_data.csv")
    if os.path.exists("models"):
        os.rmdir("models") 
    if os.path.exists("clustered_abstracts.csv"):
        os.remove("clustered_abstracts.csv")

def test_main_function(setup_environment):
    # Call the main function
    main()

    # Check if model files are created
    assert os.path.exists("models/kmeans_model.pkl")
    assert os.path.exists("models/sentence_model.pkl")

    # Check if the clustered abstracts CSV file is created
    assert os.path.exists("clustered_abstracts.csv")

    # Load the clustered abstracts and check the labels
    clustered_abstracts = pd.read_csv("clustered_abstracts.csv")
    assert 'label' in clustered_abstracts.columns
    assert len(clustered_abstracts) == 5  

    # Check if the clustering made sense (simple test)
    assert clustered_abstracts['label'].nunique() == 5  

def test_embeddings_generation():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    sample_data = ["AI in healthcare.", "Climate change impacts."]
    embeddings = model.encode(sample_data)
    
    # Ensure we get embeddings for all samples
    assert len(embeddings) == len(sample_data)  
    # Check the shape of embeddings (based on model used)
    assert embeddings[0].shape == (384,)  
