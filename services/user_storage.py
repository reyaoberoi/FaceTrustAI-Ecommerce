import json
import os
import numpy as np
from pathlib import Path
from services.face_recognition import FaceRecognitionService

class UserStorageService:
    def __init__(self, storage_path: str = "data/users"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.face_service = FaceRecognitionService()
    
    def register_user(self, user_id: str, name: str, face_image: np.ndarray) -> bool:
        """Register a new user with face embedding"""
        try:
            # Extract face embedding
            embedding = self.face_service.extract_embedding(face_image)
            
            # Store user data
            user_data = {
                'user_id': user_id,
                'name': name,
                'face_embedding': embedding.tolist()
            }
            
            user_file = self.storage_path / f"{user_id}.json"
            with open(user_file, 'w') as f:
                json.dump(user_data, f)
            
            return True
            
        except Exception as e:
            print(f"User registration error: {e}")
            return False
    
    def get_user(self, user_id: str) -> dict:
        """Retrieve user data"""
        try:
            user_file = self.storage_path / f"{user_id}.json"
            
            if not user_file.exists():
                return None
            
            with open(user_file, 'r') as f:
                user_data = json.load(f)
            
            # Convert embedding back to numpy array
            user_data['face_embedding'] = np.array(user_data['face_embedding'])
            
            return user_data
            
        except Exception as e:
            print(f"User retrieval error: {e}")
            return None
    
    def delete_user(self, user_id: str) -> bool:
        """Delete user data"""
        try:
            user_file = self.storage_path / f"{user_id}.json"
            
            if user_file.exists():
                user_file.unlink()
                return True
            
            return False
            
        except Exception as e:
            print(f"User deletion error: {e}")
            return False
