"""
ASL MediaPipe + Random Forest Prediction Service
Uses hand landmarks for robust recognition
"""

import os
import cv2
import numpy as np
import joblib
import mediapipe as mp
from typing import Dict, List, Optional
from PIL import Image


class MediaPipeASLPredictor:
    """ASL Prediction using MediaPipe hand landmarks + Random Forest"""
    
    def __init__(self, model_path='models/asl_model.pkl'):
        """
        Initialize the predictor
        
        Args:
            model_path: Path to the trained model
        """
        # Load model
        model_data = joblib.load(model_path)
        self.model = model_data['model']
        self.labels = model_data['labels']
        
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.3,  # Lowered for better detection
            min_tracking_confidence=0.3
        )
        
        print(f"MediaPipe ASL model loaded with {len(self.labels)} classes")
    
    def extract_landmarks(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract hand landmarks from image
        
        Args:
            image: RGB image as numpy array
            
        Returns:
            Flattened array of 63 features (21 landmarks × 3 coordinates)
        """
        # Process image
        results = self.hands.process(image)
        
        if not results.multi_hand_landmarks:
            return None
        
        # Extract landmarks
        hand_landmarks = results.multi_hand_landmarks[0]
        landmarks = []
        
        for landmark in hand_landmarks.landmark:
            landmarks.extend([landmark.x, landmark.y, landmark.z])
        
        # Normalize landmarks
        landmarks = np.array(landmarks)
        landmarks = self._normalize_landmarks(landmarks)
        
        return landmarks
    
    def _normalize_landmarks(self, landmarks: np.ndarray) -> np.ndarray:
        """Normalize landmarks to be scale and position invariant"""
        # Reshape to (21, 3)
        landmarks = landmarks.reshape(21, 3)
        
        # Center at wrist (index 0)
        wrist = landmarks[0]
        centered = landmarks - wrist
        
        # Scale by palm size (distance from wrist to middle finger MCP)
        palm_size = np.linalg.norm(centered[9][:2])  # Middle finger MCP
        if palm_size > 0:
            centered = centered / palm_size
        
        # Flatten back
        return centered.flatten()
    
    def predict(self, image_path_or_array, top_k: int = 3) -> Dict:
        """
        Make prediction on an image
        
        Args:
            image_path_or_array: Path to image file or numpy array
            top_k: Number of top predictions to return
            
        Returns:
            Dictionary with prediction results
        """
        # Load image
        if isinstance(image_path_or_array, str):
            image = cv2.imread(image_path_or_array)
            if image is None:
                raise ValueError(f"Could not load image from {image_path_or_array}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif isinstance(image_path_or_array, Image.Image):
            # Convert PIL Image to numpy
            image = np.array(image_path_or_array)
            if len(image.shape) == 2:  # Grayscale
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image = image_path_or_array
        
        # Extract features
        features = self.extract_landmarks(image)
        
        if features is None:
            return {
                'success': False,
                'error': 'No hand detected',
                'top_prediction': None,
                'confidence': 0.0,
                'predictions': []
            }
        
        # Predict
        probabilities = self.model.predict_proba([features])[0]
        
        # Get top-k predictions
        top_indices = np.argsort(probabilities)[-top_k:][::-1]
        
        predictions = [
            {
                'letter': self.labels[idx],
                'confidence': float(probabilities[idx])
            }
            for idx in top_indices
        ]
        
        return {
            'success': True,
            'top_prediction': self.labels[top_indices[0]],
            'confidence': float(probabilities[top_indices[0]]),
            'predictions': predictions
        }


# CLI for testing
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test MediaPipe ASL model')
    parser.add_argument('--image', type=str, required=True, help='Path to image file')
    parser.add_argument('--model', type=str, default='models/asl_model.pkl', help='Path to model file')
    args = parser.parse_args()
    
    # Load predictor
    predictor = MediaPipeASLPredictor(model_path=args.model)
    
    # Make prediction
    result = predictor.predict(args.image)
    
    # Print results
    if result['success']:
        print(f"\nPredicted letter: {result['top_prediction']}")
        print(f"Confidence: {result['confidence']*100:.2f}%")
        print("\nTop predictions:")
        for pred in result['predictions']:
            print(f"  {pred['letter']}: {pred['confidence']*100:.2f}%")
    else:
        print(f"\nError: {result['error']}")
