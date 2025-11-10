"""
Improved MediaPipe ASL Predictor
Enhanced version with better preprocessing for higher accuracy
"""

import os
import cv2
import numpy as np
import joblib
from typing import Dict, Optional

# Try to import mediapipe
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("Warning: MediaPipe not available. Install with: pip install mediapipe")


class ImprovedMediaPipePredictor:
    """Enhanced MediaPipe predictor with better accuracy"""
    
    def __init__(self, model_path='sign_language_model.pkl'):
        """Initialize with improved settings"""
        if not MEDIAPIPE_AVAILABLE:
            raise ImportError("MediaPipe not installed")
        
        # Load model (use the hand landmarks model, not the pixel-based model)
        model_data = joblib.load(model_path)
        self.model = model_data['model']
        
        # Handle different label formats
        if 'labels' in model_data:
            # New format: list of labels
            self.labels = model_data['labels']
        elif 'label_map' in model_data:
            # Old format: dict {index: letter}
            label_map = model_data['label_map']
            max_idx = max(label_map.keys())
            self.labels = [label_map.get(i, f'Class_{i}') for i in range(max_idx + 1)]
        else:
            raise ValueError("Model doesn't have labels or label_map")
        
        # Initialize MediaPipe with IMPROVED settings
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.3,  # Lower for better detection
            min_tracking_confidence=0.3,
            model_complexity=1  # Higher complexity = better accuracy
        )
        
        print(f"✓ Improved MediaPipe predictor loaded with {len(self.labels)} classes")
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Enhanced image preprocessing for better hand detection
        """
        # 1. Increase contrast
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
        
        # 2. Denoise
        denoised = cv2.fastNlMeansDenoisingColored(enhanced, None, 10, 10, 7, 21)
        
        # 3. Sharpen
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(denoised, -1, kernel)
        
        return sharpened
    
    def extract_landmarks(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Enhanced landmark extraction with better normalization"""
        # Preprocess first
        processed = self.preprocess_image(image)
        
        # Process with MediaPipe
        results = self.hands.process(processed)
        
        if not results.multi_hand_landmarks:
            return None
        
        # Extract landmarks
        hand_landmarks = results.multi_hand_landmarks[0]
        landmarks = []
        
        for landmark in hand_landmarks.landmark:
            landmarks.extend([landmark.x, landmark.y, landmark.z])
        
        # Enhanced normalization
        landmarks = np.array(landmarks)
        landmarks = self._improved_normalize(landmarks)
        
        return landmarks
    
    def _improved_normalize(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Improved normalization for better accuracy
        - Scale invariant
        - Position invariant
        - Rotation partially invariant
        """
        # Reshape to (21, 3)
        landmarks = landmarks.reshape(21, 3)
        
        # 1. Center at wrist (landmark 0)
        wrist = landmarks[0]
        centered = landmarks - wrist
        
        # 2. Scale by hand size (wrist to middle finger tip)
        middle_finger_tip = centered[12]  # Middle finger tip
        hand_size = np.linalg.norm(middle_finger_tip[:2])
        
        if hand_size > 0:
            scaled = centered / hand_size
        else:
            scaled = centered
        
        # 3. Normalize z-axis separately
        scaled[:, 2] = (scaled[:, 2] - scaled[:, 2].mean()) / (scaled[:, 2].std() + 1e-6)
        
        return scaled.flatten()
    
    def predict(self, image_or_path, top_k: int = 3) -> Dict:
        """
        Make prediction with enhanced preprocessing
        """
        # Load image
        if isinstance(image_or_path, str):
            image = cv2.imread(image_or_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_or_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = image_or_path
        
        # Ensure RGB
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # Extract features
        features = self.extract_landmarks(image)
        
        if features is None:
            return {
                'success': False,
                'error': 'No hand detected. Please ensure your hand is clearly visible.',
                'top_prediction': None,
                'confidence': 0.0,
                'predictions': []
            }
        
        # Predict with probability calibration
        raw_probas = self.model.predict_proba([features])[0]
        
        # Apply temperature scaling for better calibration
        temperature = 1.5  # Smooths probabilities
        calibrated_probas = np.exp(np.log(raw_probas + 1e-10) / temperature)
        calibrated_probas = calibrated_probas / calibrated_probas.sum()
        
        # Get top-k predictions
        top_indices = np.argsort(calibrated_probas)[-top_k:][::-1]
        
        predictions = [
            {
                'letter': self.labels[idx],
                'confidence': float(calibrated_probas[idx]),
                'confidence_percent': f"{calibrated_probas[idx]*100:.1f}%"
            }
            for idx in top_indices
        ]
        
        return {
            'success': True,
            'top_prediction': self.labels[top_indices[0]],
            'confidence': float(calibrated_probas[top_indices[0]]),
            'predictions': predictions
        }


# Quick function for compatibility
def predict_with_improved_mediapipe(image, model_path='models/asl_model.pkl'):
    """Quick prediction function"""
    predictor = ImprovedMediaPipePredictor(model_path)
    return predictor.predict(image)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python improved_mediapipe_predictor.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    print("\n" + "="*60)
    print("Testing Improved MediaPipe Predictor")
    print("="*60)
    
    predictor = ImprovedMediaPipePredictor()
    result = predictor.predict(image_path)
    
    if result['success']:
        print(f"\n✓ Prediction successful!")
        print(f"Top Prediction: {result['top_prediction']}")
        print(f"Confidence: {result['confidence']*100:.1f}%")
        print(f"\nTop 3 Predictions:")
        for i, pred in enumerate(result['predictions'], 1):
            print(f"  {i}. {pred['letter']:3s} - {pred['confidence_percent']}")
    else:
        print(f"\n✗ Prediction failed: {result['error']}")
    
    print("="*60)
