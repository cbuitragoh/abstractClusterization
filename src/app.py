from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
from sentence_transformers import SentenceTransformer

# Load the saved models from src/model/ folder not using global
# models folder because that is for experiments
kmeans = joblib.load('model/kmeans_model.pkl')
model = joblib.load('model/sentence_model.pkl')

# Create a FastAPI app
app = FastAPI()

# Define request body schema
class AbstractsRequest(BaseModel):
    abstracts: list[str]

# Define response schema
class PredictionResponse(BaseModel):
    abstract: str
    predicted_label: int

@app.post("/predict", response_model=list[PredictionResponse])
async def predict_topic(request: AbstractsRequest):
    if not request.abstracts:
        raise HTTPException(status_code=400, detail="No abstracts provided.")
    
    # Generate embeddings for new abstracts
    new_embeddings = model.encode(request.abstracts)

    # Predict clusters
    predicted_labels = kmeans.predict(new_embeddings)

    # Prepare response
    response = [
        PredictionResponse(abstract=abstract, predicted_label=int(label))
        for abstract, label in zip(request.abstracts, predicted_labels)
    ]
    
    return response

