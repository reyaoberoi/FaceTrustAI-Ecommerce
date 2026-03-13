# FaceTrust AI - Backend

FastAPI backend for biometric payment verification.

## Setup

1. Install Python 3.8+

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the server:
```bash
python main.py
```

Server runs at http://localhost:8000

## API Endpoints

### POST /api/register-user
Register a new user with face image
```json
{
  "user_id": "user123",
  "name": "John Doe",
  "image": "base64_encoded_image"
}
```

### POST /api/verify-payment
Verify payment with biometric authentication
```json
{
  "user_id": "user123",
  "image": "base64_encoded_image",
  "amount": 100.00
}
```

### GET /api/health
Health check endpoint

## Services

- Face Recognition: DeepFace with Facenet512
- Liveness Detection: Multi-heuristic approach
- Deepfake Detection: Frequency and texture analysis
- Emotion Analysis: DeepFace emotion detection
- Risk Scoring: Weighted scoring algorithm

## Data Storage

User data stored in `data/users/` directory as JSON files.
