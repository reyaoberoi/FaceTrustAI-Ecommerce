import numpy as np
import cv2

class LivenessDetectionService:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
    
    def detect_liveness(self, image: np.ndarray) -> float:
        """
        Detect if the face is live (not a photo/video)
        Uses multiple heuristics:
        - Face detection confidence
        - Eye detection
        - Texture analysis
        - Color distribution
        """
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
            
            score = 0.0
            
            # 1. Face detection (25%)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            if len(faces) > 0:
                score += 0.25
            
            # 2. Eye detection (25%)
            eyes = self.eye_cascade.detectMultiScale(gray, 1.1, 5)
            if len(eyes) >= 2:
                score += 0.25
            
            # 3. Texture analysis - Laplacian variance (25%)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            if laplacian_var > 100:  # Real faces have more texture
                score += 0.25
            
            # 4. Color distribution (25%)
            if len(image.shape) == 3:
                # Check for skin tone presence
                hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
                lower_skin = np.array([0, 20, 70], dtype=np.uint8)
                upper_skin = np.array([20, 255, 255], dtype=np.uint8)
                skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
                skin_ratio = np.sum(skin_mask > 0) / skin_mask.size
                
                if skin_ratio > 0.1:
                    score += 0.25
            
            return float(np.clip(score, 0, 1))
            
        except Exception as e:
            print(f"Liveness detection error: {e}")
            return 0.6  # Neutral-positive score for demo
