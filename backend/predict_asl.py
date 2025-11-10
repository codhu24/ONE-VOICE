"""
ASL Alphabet Prediction Service
Provides prediction functionality for ASL alphabet recognition
"""

import os
import json
import numpy as np
from tensorflow import keras
from PIL import Image
import argparse

class ASLPredictor:
    """ASL Alphabet Prediction Class"""
    
    def __init__(self, model_path='models/asl_model.h5', class_indices_path='models/class_indices.json'):
        """
        Initialize the predictor
        
        Args:
            model_path: Path to the trained Keras model
            class_indices_path: Path to class indices JSON file
        """
        print(f"Loading model from {model_path}...")
        self.model = keras.models.load_model(model_path)
        print("Model loaded successfully!")
        
        # Load class indices
        with open(class_indices_path, 'r') as f:
            self.class_indices = json.load(f)
        
        # Create reverse mapping (index -> class name)
        self.index_to_class = {v: k for k, v in self.class_indices.items()}
        self.num_classes = len(self.class_indices)
        
        print(f"Loaded {self.num_classes} classes: {list(self.class_indices.keys())}")
    
    def preprocess_image(self, image_path_or_array, target_size=(200, 200)):
        """
        Preprocess image for prediction
        
        Args:
            image_path_or_array: Path to image file or numpy array
            target_size: Target size for resizing
            
        Returns:
            Preprocessed image array
        """
        # Load image
        if isinstance(image_path_or_array, str):
            img = Image.open(image_path_or_array)
        elif isinstance(image_path_or_array, np.ndarray):
            img = Image.fromarray(image_path_or_array)
        else:
            img = image_path_or_array
        
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize
        img = img.resize(target_size)
        
        # Convert to array and normalize
        img_array = np.array(img) / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    def predict(self, image_path_or_array, top_k=3):
        """
        Predict ASL letter from image
        
        Args:
            image_path_or_array: Path to image file or numpy array
            top_k: Number of top predictions to return
            
        Returns:
            Dictionary with prediction results
        """
        # Preprocess image
        img_array = self.preprocess_image(image_path_or_array)
        
        # Make prediction
        predictions = self.model.predict(img_array, verbose=0)[0]
        
        # Get top k predictions
        top_indices = np.argsort(predictions)[-top_k:][::-1]
        
        results = {
            'predictions': [],
            'top_prediction': None,
            'confidence': None
        }
        
        for idx in top_indices:
            class_name = self.index_to_class[idx]
            confidence = float(predictions[idx])
            
            results['predictions'].append({
                'letter': class_name,
                'confidence': confidence,
                'confidence_percent': f"{confidence * 100:.2f}%"
            })
        
        # Set top prediction
        results['top_prediction'] = results['predictions'][0]['letter']
        results['confidence'] = results['predictions'][0]['confidence']
        
        return results
    
    def predict_batch(self, image_paths, top_k=3):
        """
        Predict ASL letters from multiple images
        
        Args:
            image_paths: List of image paths
            top_k: Number of top predictions to return
            
        Returns:
            List of prediction results
        """
        results = []
        for image_path in image_paths:
            result = self.predict(image_path, top_k=top_k)
            result['image_path'] = image_path
            results.append(result)
        return results


def main():
    """Command-line interface for testing predictions"""
    parser = argparse.ArgumentParser(description='ASL Alphabet Prediction')
    parser.add_argument('--image', type=str, required=True, help='Path to image file')
    parser.add_argument('--model', type=str, default='models/asl_model.h5', help='Path to model file')
    parser.add_argument('--top-k', type=int, default=3, help='Number of top predictions to show')
    
    args = parser.parse_args()
    
    # Check if image exists
    if not os.path.exists(args.image):
        print(f"Error: Image file not found: {args.image}")
        return
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        print("Please train the model first using: python train_asl_model.py")
        return
    
    # Create predictor
    predictor = ASLPredictor(model_path=args.model)
    
    # Make prediction
    print(f"\nPredicting ASL letter from: {args.image}")
    print("=" * 60)
    
    results = predictor.predict(args.image, top_k=args.top_k)
    
    print(f"\nTop Prediction: {results['top_prediction']}")
    print(f"Confidence: {results['confidence']*100:.2f}%")
    print(f"\nTop {args.top_k} Predictions:")
    print("-" * 60)
    
    for i, pred in enumerate(results['predictions'], 1):
        print(f"{i}. {pred['letter']:10s} - {pred['confidence_percent']:>7s} confidence")
    
    print("=" * 60)


if __name__ == '__main__':
    main()
