"""
Improved Random Forest Predictor
Uses the newly trained model with PCA features
"""

import os
import numpy as np
import cv2
import joblib
from typing import Dict, Optional
from PIL import Image

class ImprovedRFPredictor:
    """Random Forest predictor using PCA features"""
    
    def __init__(self, model_path='models/asl_model.pkl'):
        """Load the improved model with scaler and PCA"""
        print(f"Loading improved model from {model_path}...")
        
        # Load model package
        model_data = joblib.load(model_path)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.pca = model_data['pca']
        self.labels = model_data['labels']
        self.accuracy = model_data.get('accuracy', 0)
        
        print(f"✓ Improved RF model loaded")
        print(f"  Accuracy: {self.accuracy*100:.2f}%")
        print(f"  Classes: {len(self.labels)}")
        print(f"  Features: {self.pca.n_components_}")
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image to 28x28 grayscale"""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Resize to 28x28
        resized = cv2.resize(gray, (28, 28))
        
        # Flatten and normalize
        pixels = resized.flatten() / 255.0
        
        return pixels
    
    def extract_features(self, image: np.ndarray) -> np.ndarray:
        """Extract features from image"""
        # Get base pixels
        pixels = self.preprocess_image(image)
        
        # Add statistical features (same as training)
        img_28 = (pixels * 255).reshape(28, 28)
        mean_feat = np.mean(img_28)
        std_feat = np.std(img_28)
        max_feat = np.max(img_28)
        
        # Combine features
        features = np.concatenate([pixels, [mean_feat, std_feat, max_feat]])
        
        return features
    
    def predict(self, image_or_path, top_k: int = 3) -> Dict:
        """
        Make prediction on image
        
        Args:
            image_or_path: Image path or numpy array
            top_k: Number of top predictions
            
        Returns:
            Prediction results
        """
        # Load image
        if isinstance(image_or_path, str):
            image = cv2.imread(image_or_path)
            if image is None:
                raise ValueError(f"Could not load: {image_or_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif isinstance(image_or_path, Image.Image):
            image = np.array(image_or_path)
        else:
            image = image_or_path
        
        # Extract features
        features = self.extract_features(image)
        
        # Scale features
        features_scaled = self.scaler.transform([features])
        
        # Apply PCA
        features_pca = self.pca.transform(features_scaled)
        
        # Predict
        probas = self.model.predict_proba(features_pca)[0]
        
        # Get top-k
        top_indices = np.argsort(probas)[-top_k:][::-1]
        
        predictions = [
            {
                'letter': self.labels[idx],
                'confidence': float(probas[idx]),
                'confidence_percent': f"{probas[idx]*100:.1f}%"
            }
            for idx in top_indices
        ]
        
        return {
            'success': True,
            'top_prediction': self.labels[top_indices[0]],
            'confidence': float(probas[top_indices[0]]),
            'predictions': predictions
        }


# CLI testing
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python predict_rf_improved.py <image_path>")
        sys.exit(1)
    
    predictor = ImprovedRFPredictor()
    result = predictor.predict(sys.argv[1])
    
    print(f"\n✓ Prediction: {result['top_prediction']}")
    print(f"✓ Confidence: {result['confidence']*100:.1f}%")
    print(f"\nTop 3:")
    for i, pred in enumerate(result['predictions'], 1):
        print(f"  {i}. {pred['letter']} - {pred['confidence_percent']}")
