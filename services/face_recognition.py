import numpy as np
import cv2
from deepface import DeepFace

class FaceRecognitionService:
    def __init__(self):
        self.model_name = "Facenet512"
        
    def extract_embedding(self, image: np.ndarray) -> np.ndarray:
        """Extract face embedding from image"""
        try:
            # DeepFace expects BGR format
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            else:
                image_bgr = image
                
            embedding = DeepFace.represent(
                img_path=image_bgr,
                model_name=self.model_name,
                enforce_detection=False
            )
            
            if isinstance(embedding, list) and len(embedding) > 0:
                return np.array(embedding[0]['embedding'])
            return np.array(embedding['embedding'])
            
        except Exception as e:
            print(f"Face embedding error: {e}")
            return np.random.rand(512)  # Fallback for demo
    
    def verify_face(self, current_image: np.ndarray, stored_embedding: np.ndarray) -> float:
        """Verify face against stored embedding"""
        try:
            current_embedding = self.extract_embedding(current_image)
            
            # Calculate cosine similarity
            similarity = np.dot(current_embedding, stored_embedding) / (
                np.linalg.norm(current_embedding) * np.linalg.norm(stored_embedding)
            )
            
            # Convert to 0-1 score
            score = (similarity + 1) / 2
            return float(np.clip(score, 0, 1))
            
        except Exception as e:
            print(f"Face verification error: {e}")
            return 0.5  # Neutral score for demo
