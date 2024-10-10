import os
import joblib
import pytest
import numpy as np
from pipelines.inference import predict_topic
from dotenv import load_dotenv
load_dotenv()

@pytest.fixture(scope="module", autouse=True)
def setup_models():
    # Load models for testing
    global kmeans, model
    kmeans = joblib.load(f'{os.getenv("MODELS_PATH")}\kmeans_model.pkl')
    model = joblib.load(f'{os.getenv("MODELS_PATH")}\sentence_model.pkl')

def test_predict_topic_valid_input():
    new_abstracts = [
        "Research on the impact of AI in healthcare.",
        "Exploring the effects of climate change on polar bears."
    ]
    predicted_labels = predict_topic(new_abstracts)

    # Check if the output is a list of integers
    assert isinstance(predicted_labels, np.ndarray)
    assert len(predicted_labels) == len(new_abstracts)
    assert all(isinstance(label, int) for label in predicted_labels)

def test_predict_topic_empty_input():
    new_abstracts = []
    with pytest.raises(Exception):
        predict_topic(new_abstracts)

def test_predict_topic_single_input():
    new_abstract = ["AI in space exploration."]
    predicted_labels = predict_topic(new_abstract)

    # Check if the output is a single label
    assert isinstance(predicted_labels, np.ndarray)
    assert len(predicted_labels) == 1
    assert isinstance(predicted_labels[0], int)

