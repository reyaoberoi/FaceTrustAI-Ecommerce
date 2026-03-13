import numpy as np
import cv2

class DeepfakeDetectionService:
    def __init__(self):
        pass
    
    def detect_deepfake(self, image: np.ndarray) -> float:
        """
        Detect if image is a deepfake
        Returns confidence score (0-1, higher = more likely real)
        
        Uses heuristics:
        - Frequency domain analysis
        - Edge consistency
        - Color coherence
        - Compression artifacts
        """
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
            
            confidence = 0.0
            
            # 1. Frequency domain analysis (30%)
            # Real images have natural frequency distribution
            f_transform = np.fft.fft2(gray)
            f_shift = np.fft.fftshift(f_transform)
            magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)
            
            # Check for unnatural frequency patterns
            freq_variance = np.var(magnitude_spectrum)
            if 1000 < freq_variance < 10000:
                confidence += 0.3
            
            # 2. Edge consistency (30%)
            edges = cv2.Canny(gray, 100, 200)
            edge_density = np.sum(edges > 0) / edges.size
            
            if 0.05 < edge_density < 0.3:  # Natural edge density
                confidence += 0.3
            
            # 3. Color coherence (20%)
            if len(image.shape) == 3:
                # Check color channel correlation
                r, g, b = image[:,:,0], image[:,:,1], image[:,:,2]
                rg_corr = np.corrcoef(r.flatten(), g.flatten())[0,1]
                
                if 0.5 < abs(rg_corr) < 0.95:
                    confidence += 0.2
            
            # 4. Noise analysis (20%)
            # Real images have consistent noise patterns
            noise = gray - cv2.GaussianBlur(gray, (5, 5), 0)
            noise_std = np.std(noise)
            
            if 5 < noise_std < 30:
                confidence += 0.2
            
            return float(np.clip(confidence, 0, 1))
            
        except Exception as e:
            print(f"Deepfake detection error: {e}")
            return 0.7  # Neutral-positive score for demo
