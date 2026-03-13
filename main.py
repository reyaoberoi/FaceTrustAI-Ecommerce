from fastapi import FastAPI, UploadFile, File, Form
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
import io
from PIL import Image
import numpy as np
from services.face_recognition import FaceRecognitionService
from services.liveness_detection import LivenessDetectionService
from services.deepfake_detection import DeepfakeDetectionService
from services.emotion_analysis import EmotionAnalysisService
from services.risk_scoring import RiskScoringEngine
from services.user_storage import UserStorageService
import random
otp_store = {}

app = FastAPI(title="FaceTrust AI API")

origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
face_service = FaceRecognitionService()
liveness_service = LivenessDetectionService()
deepfake_service = DeepfakeDetectionService()
emotion_service = EmotionAnalysisService()
risk_engine = RiskScoringEngine()
storage_service = UserStorageService()

class UserRegistration(BaseModel):
    user_id: str
    name: str
    image: str

class PaymentVerification(BaseModel):
    user_id: str
    image: str
    amount: float

def decode_image(image_data: str) -> np.ndarray:
    """Decode base64 image to numpy array"""
    if ',' in image_data:
        image_data = image_data.split(',')[1]
    
    image_bytes = base64.b64decode(image_data)
    image = Image.open(io.BytesIO(image_bytes))
    return np.array(image)

@app.get("/")
def read_root():
    return {"message": "FaceTrust AI API", "status": "running"}

from mangum import Mangum
handler = Mangum(app)

@app.post("/api/register-user")
async def register_user(
    user_id: str = Form(...),
    name: str = Form(...),
    image: UploadFile = File(...)
):
    """Register a new user with face image"""
    try:
        image_bytes = await image.read()

        image_pil = Image.open(io.BytesIO(image_bytes))
        image_np = np.array(image_pil)

        success = storage_service.register_user(
            user_id=user_id,
            name=name,
            face_image=image_np
        )

        if success:
            return {
                "message": f"User {name} registered successfully",
                "user_id": user_id
            }
        else:
            raise HTTPException(status_code=400, detail="Registration failed")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/verify-payment")
async def verify_payment(
    user_id: str = Form(...),
    amount: float = Form(...),
    image: UploadFile = File(...)
):

    try:

        image_bytes = await image.read()
        image_pil = Image.open(io.BytesIO(image_bytes))
        image = np.array(image_pil)

        user_data = storage_service.get_user(user_id)

        if not user_data:
            return {
                "decision": "BLOCKED",
                "message": "User not found",
                "risk_score": 0.0
            }

        face_match_score = face_service.verify_face(
            image,
            user_data['face_embedding']
        )

        liveness_score = liveness_service.detect_liveness(image)

        deepfake_confidence = deepfake_service.detect_deepfake(image)

        emotion_result = emotion_service.analyze_emotion(image)

        risk_score = risk_engine.calculate_risk_score(
            face_match_score=face_match_score,
            liveness_score=liveness_score,
            deepfake_confidence=deepfake_confidence,
            emotion_stability_score=emotion_result['stability_score']
        )

        decision, message = risk_engine.make_decision(risk_score, amount)

        # OTP CASE
        if decision == "OTP_REQUIRED":

            otp = random.randint(100000, 999999)
            otp_store[user_id] = otp

            print("Generated OTP:", otp)

            return {
                "decision": "OTP_REQUIRED",
                "message": "OTP sent. Please verify.",
                "risk_score": risk_score,
                "details": {
                    "face_match": face_match_score,
                    "liveness": liveness_score,
                    "deepfake": deepfake_confidence,
                    "emotion": emotion_result['dominant_emotion'],
                    "emotion_stability": emotion_result['stability_score']
                }
            }

        # NORMAL RESPONSE
        return {
            "decision": decision,
            "message": message,
            "risk_score": risk_score,
            "details": {
                "face_match": face_match_score,
                "liveness": liveness_score,
                "deepfake": deepfake_confidence,
                "emotion": emotion_result['dominant_emotion'],
                "emotion_stability": emotion_result['stability_score']
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

import random

@app.post("/api/send-otp")
def send_otp(user_id: str = Form(...)):
    otp = random.randint(100000, 999999)
    otp_store[user_id] = otp

    # For demo we return OTP directly
    # In real system you would send via SMS/email
    return {
        "message": "OTP generated",
        "user_id": user_id,
        "otp": otp
    }


@app.post("/api/verify-otp")
def verify_otp(user_id: str = Form(...), otp: str = Form(...)):
    stored_otp = otp_store.get(user_id)

    if stored_otp and str(stored_otp) == otp:
        return {
            "decision": "APPROVED",
            "message": "OTP verified. Payment approved."
        }

    return {
        "decision": "BLOCKED",
        "message": "Invalid OTP"
    }        

@app.get("/api/health")
def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
