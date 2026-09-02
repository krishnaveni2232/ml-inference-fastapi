from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn as nn
import joblib


# Define the Neural Network
class TextClassifier(nn.Module):
    def __init__(self, input_size):
        super(TextClassifier, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(32, 2)
        )

    def forward(self, x):
        return self.network(x)


# Load preprocessing objects
tfidf = joblib.load("tfidf_vectorizer.pkl")
label_encoder = joblib.load("label_encoder.pkl")


# Create model
input_size = len(tfidf.get_feature_names_out())
model = TextClassifier(input_size)

# Load trained weights
model.load_state_dict(
    torch.load("champion_model.pth", map_location=torch.device("cpu"))
)

model.eval()


# Create FastAPI application
app = FastAPI(
    title="Real-Time ML Inference API",
    description="Sentiment classification API using a trained neural network",
    version="1.0.0"
)


# Request schema
class PredictionRequest(BaseModel):
    text: str


# Prediction endpoint
@app.post("/predict")
def predict(request: PredictionRequest):

    # Convert text to TF-IDF vector
    vector = tfidf.transform([request.text]).toarray()

    # Convert to PyTorch tensor
    input_tensor = torch.tensor(vector, dtype=torch.float32)

    # Make prediction
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)

    predicted_index = torch.argmax(probabilities, dim=1).item()
    predicted_label = label_encoder.inverse_transform([predicted_index])[0]

    return {
        "text": request.text,
        "prediction": predicted_label,
        "probabilities": {
            label_encoder.classes_[0]: float(probabilities[0][0]),
            label_encoder.classes_[1]: float(probabilities[0][1])
        }
    }


@app.get("/")
def root():
    return {
        "message": "ML Inference API is running"
    }
