class RiskScoringEngine:
    def __init__(self):
        # Weights for risk calculation
        self.weights = {
            'face_match': 0.4,
            'liveness': 0.2,
            'deepfake': 0.2,
            'emotion': 0.2
        }
        
        # Thresholds
        self.approve_threshold = 0.8
        self.otp_threshold = 0.5
    
    def calculate_risk_score(
        self,
        face_match_score: float,
        liveness_score: float,
        deepfake_confidence: float,
        emotion_stability_score: float
    ) -> float:
        """
        Calculate overall risk score using weighted average
        
        Formula:
        risk_score = (face_match * 0.4) + (liveness * 0.2) + 
                     (deepfake * 0.2) + (emotion * 0.2)
        """
        risk_score = (
            face_match_score * self.weights['face_match'] +
            liveness_score * self.weights['liveness'] +
            deepfake_confidence * self.weights['deepfake'] +
            emotion_stability_score * self.weights['emotion']
        )
        
        return round(risk_score, 3)
    
    def make_decision(self, risk_score: float, amount: float) -> tuple:
        """
        Make payment decision based on risk score and amount
        
        Returns: (decision, message)
        - APPROVED: High confidence, proceed with payment
        - OTP_REQUIRED: Medium confidence, require additional verification
        - BLOCKED: Low confidence, block transaction
        """
        # Adjust thresholds based on amount
        approve_threshold = self.approve_threshold
        otp_threshold = self.otp_threshold
        
        if amount > 1000:
            approve_threshold += 0.05
            otp_threshold += 0.05
        elif amount > 5000:
            approve_threshold += 0.1
            otp_threshold += 0.1
        
        if risk_score >= approve_threshold:
            return (
                "APPROVED",
                f"Payment of ${amount:.2f} approved successfully!"
            )
        elif risk_score >= otp_threshold:
            return (
                "OTP_REQUIRED",
                f"Additional verification required for ${amount:.2f}. Please enter OTP."
            )
        else:
            return (
                "BLOCKED",
                f"Payment of ${amount:.2f} blocked due to security concerns."
            )
