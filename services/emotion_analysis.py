import numpy as np
from deepface import DeepFace

class EmotionAnalysisService:
    def __init__(self):
        self.suspicious_emotions = ['angry', 'fear', 'disgust']
        self.neutral_emotions = ['neutral', 'calm']
        self.positive_emotions = ['happy']
    
    def analyze_emotion(self, image: np.ndarray) -> dict:
        """
        Analyze facial emotion
        Returns dominant emotion and stability score
        """
        try:
            # Analyze emotion using DeepFace
            result = DeepFace.analyze(
                img_path=image,
                actions=['emotion'],
                enforce_detection=False
            )
            
            if isinstance(result, list):
                result = result[0]
            
            emotions = result['emotion']
            dominant_emotion = result['dominant_emotion']
            
            # Calculate stability score
            stability_score = self._calculate_stability(dominant_emotion, emotions)
            
            return {
                'dominant_emotion': dominant_emotion,
                'emotions': emotions,
                'stability_score': stability_score
            }
            
        except Exception as e:
            print(f"Emotion analysis error: {e}")
            return {
                'dominant_emotion': 'neutral',
                'emotions': {'neutral': 100},
                'stability_score': 0.8
            }
    
    def _calculate_stability(self, dominant_emotion: str, emotions: dict) -> float:
        """
        Calculate emotional stability score
        Higher score = more stable/appropriate for payment
        """
        score = 0.5  # Base score
        
        # Penalize suspicious emotions
        if dominant_emotion in self.suspicious_emotions:
            score -= 0.3
        
        # Reward neutral/positive emotions
        if dominant_emotion in self.neutral_emotions:
            score += 0.3
        elif dominant_emotion in self.positive_emotions:
            score += 0.2
        
        # Check emotion confidence
        dominant_confidence = emotions.get(dominant_emotion, 0) / 100.0
        if dominant_confidence > 0.6:
            score += 0.2
        
        return float(np.clip(score, 0, 1))
