# Real-Time ML Inference REST API

## Project Overview

This project deploys a trained sentiment classification model as a production-ready REST API using FastAPI and Docker.

The API accepts text input and returns the predicted sentiment along with prediction probabilities.

## ML Model

The system uses a TF-IDF vectorizer and a PyTorch neural network for binary sentiment classification.

### Model Architecture

- Input Layer: TF-IDF features
- Dense Layer: 64 neurons
- Batch Normalization
- ReLU Activation
- Dropout: 0.3
- Dense Layer: 32 neurons
- Batch Normalization
- ReLU Activation
- Dropout: 0.3
- Output Layer: 2 classes

## API Endpoint

### POST /predict

Request:

```json
{
  "text": "This product is excellent"
}
## Docker Deployment

The FastAPI application is containerized using Docker.

### Build Docker Image

```bash
docker build -t ml-inference-api .
docker run -d -p 8000:8000 --name ml-inference-container ml-inference-api
### API Testing

The API was successfully tested using Swagger UI at:

```text
http://localhost:8000/docs
{
  "text": "This product is excellent",
  "prediction": "positive",
  "probabilities": {
    "negative": 0.1665,
    "positive": 0.8335
  }
}
