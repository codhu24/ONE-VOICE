"""
Ensemble Predictor for ASL Recognition
Combines predictions from multiple models for higher accuracy
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import Counter
import time

try:
    from predict_asl import ASLPredictor
    CNN_AVAILABLE = True
except (ImportError, AttributeError) as e:
    CNN_AVAILABLE = False
    print("Warning: CNN model not available (TensorFlow issue)")

try:
    from predict_vit import ViTPredictor
    VIT_AVAILABLE = True
except (ImportError, AttributeError) as e:
    VIT_AVAILABLE = False
    print("Warning: ViT model not available (PyTorch issue)")

try:
    from improved_mediapipe_predictor import ImprovedMediaPipePredictor
    MEDIAPIPE_AVAILABLE = True
    USE_IMPROVED = True
    print("✓ Using IMPROVED MediaPipe predictor")
except ImportError:
    try:
        from predict_mediapipe import MediaPipeASLPredictor
        MEDIAPIPE_AVAILABLE = True
        USE_IMPROVED = False
        print("Using standard MediaPipe predictor")
    except ImportError:
        MEDIAPIPE_AVAILABLE = False
        USE_IMPROVED = False
        print("Warning: MediaPipe model not available.")

try:
    from predict_rf_improved import ImprovedRFPredictor
    RF_IMPROVED_AVAILABLE = True
    print("✓ Improved RF predictor available")
except (ImportError, AttributeError) as e:
    RF_IMPROVED_AVAILABLE = False
    print("Warning: Improved RF predictor not available")

try:
    from enhanced_preprocessing import EnhancedPreprocessor
    ENHANCED_PREPROCESSING_AVAILABLE = True
except (ImportError, AttributeError) as e:
    ENHANCED_PREPROCESSING_AVAILABLE = False
    EnhancedPreprocessor = None
    print("Warning: Enhanced preprocessing not available")


class EnsemblePredictor:
    """
    Ensemble predictor that combines predictions from multiple models
    
    Methods:
    1. Voting: Majority vote from all models
    2. Weighted Average: Weight by model accuracy
    3. Confidence Threshold: Only use high-confidence predictions
    """
    
    def __init__(
        self,
        use_cnn: bool = True,
        use_vit: bool = True,
        use_mediapipe: bool = True,
        weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize ensemble predictor
        
        Args:
            use_cnn: Use CNN model
            use_vit: Use ViT model
            use_mediapipe: Use MediaPipe model
            weights: Model weights for weighted voting (default: equal weights)
        """
        self.models = {}
        
        # Try to load preprocessor if available
        if ENHANCED_PREPROCESSING_AVAILABLE and EnhancedPreprocessor:
            self.preprocessor = EnhancedPreprocessor()
        else:
            self.preprocessor = None
        
        # Load models
        if use_cnn and CNN_AVAILABLE:
            try:
                self.models['cnn'] = ASLPredictor()
                print("✓ CNN model loaded")
            except Exception as e:
                print(f"✗ Failed to load CNN model: {e}")
        
        if use_vit and VIT_AVAILABLE:
            try:
                self.models['vit'] = ViTPredictor()
                print("✓ ViT model loaded")
            except Exception as e:
                print(f"✗ Failed to load ViT model: {e}")
        
        if use_mediapipe and MEDIAPIPE_AVAILABLE:
            try:
                if USE_IMPROVED:
                    self.models['mediapipe'] = ImprovedMediaPipePredictor()
                    print("✓ MediaPipe model loaded (IMPROVED version)")
                else:
                    self.models['mediapipe'] = MediaPipeASLPredictor()
                    print("✓ MediaPipe model loaded (standard version)")
            except Exception as e:
                print(f"✗ Failed to load MediaPipe model: {e}")
        
        # Add improved RF model if available (uses pixel features + PCA)
        if RF_IMPROVED_AVAILABLE:
            try:
                self.models['rf_improved'] = ImprovedRFPredictor()
                print("✓ Improved RF model loaded (75-85% accuracy)")
            except Exception as e:
                print(f"✗ Failed to load improved RF model: {e}")
        
        # Set weights (based on empirical accuracy)
        # MediaPipe: ~70%, ViT: ~95%, CNN: ~92%, RF_Improved: ~80%
        self.weights = weights or {
            'mediapipe': 0.25,      # Hand landmarks
            'vit': 0.30,            # Vision Transformer
            'cnn': 0.25,            # Convolutional NN
            'rf_improved': 0.20     # Improved Random Forest
        }
        
        # Normalize weights for available models
        available_models = list(self.models.keys())
        total_weight = sum(self.weights.get(m, 0) for m in available_models)
        if total_weight > 0:
            self.weights = {
                m: self.weights.get(m, 0) / total_weight 
                for m in available_models
            }
        
        print(f"\nEnsemble initialized with {len(self.models)} models")
        print(f"Weights: {self.weights}")
    
    def predict(
        self,
        image,
        method: str = 'weighted',
        top_k: int = 3,
        min_confidence: float = 0.3
    ) -> Dict:
        """
        Make ensemble prediction
        
        Args:
            image: Input image (path, numpy array, or PIL Image)
            method: 'weighted', 'voting', or 'confidence_threshold'
            top_k: Number of top predictions to return
            min_confidence: Minimum confidence threshold
            
        Returns:
            Dictionary with prediction results
        """
        if not self.models:
            return {
                'success': False,
                'error': 'No models available',
                'top_prediction': None,
                'confidence': 0.0,
                'predictions': []
            }
        
        start_time = time.time()
        
        # Get predictions from all models
        all_predictions = {}
        
        for model_name, model in self.models.items():
            try:
                # Preprocess for specific model (if preprocessor available)
                if self.preprocessor:
                    processed_img = self.preprocessor.preprocess_for_model(
                        image, 
                        model_type=model_name,
                        enhance=True
                    )
                else:
                    # Use image as-is if no preprocessor
                    processed_img = image
                
                # Get prediction
                result = model.predict(processed_img, top_k=5)
                
                # Handle different result formats
                if isinstance(result, dict):
                    if result.get('success') is False:
                        continue  # Skip if model failed
                    all_predictions[model_name] = result
                else:
                    all_predictions[model_name] = result
                    
            except Exception as e:
                print(f"Error in {model_name} prediction: {e}")
                continue
        
        if not all_predictions:
            return {
                'success': False,
                'error': 'All models failed to predict',
                'top_prediction': None,
                'confidence': 0.0,
                'predictions': []
            }
        
        # Combine predictions based on method
        if method == 'voting':
            result = self._voting_ensemble(all_predictions, top_k)
        elif method == 'confidence_threshold':
            result = self._confidence_threshold_ensemble(all_predictions, min_confidence, top_k)
        else:  # weighted (default)
            result = self._weighted_ensemble(all_predictions, top_k)
        
        # Add metadata
        result['processing_time_ms'] = int((time.time() - start_time) * 1000)
        result['num_models'] = len(all_predictions)
        result['models_used'] = list(all_predictions.keys())
        
        return result
    
    def _weighted_ensemble(self, all_predictions: Dict, top_k: int) -> Dict:
        """Weighted average of model predictions"""
        # Collect all letters and their weighted confidences
        letter_scores = {}
        
        for model_name, pred_result in all_predictions.items():
            weight = self.weights.get(model_name, 0)
            
            # Extract predictions
            predictions = pred_result.get('predictions', [])
            if not predictions:
                continue
            
            for pred in predictions:
                letter = pred.get('letter', '')
                confidence = pred.get('confidence', 0)
                
                if letter not in letter_scores:
                    letter_scores[letter] = 0
                letter_scores[letter] += confidence * weight
        
        # Sort by score
        sorted_letters = sorted(
            letter_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
        
        if not sorted_letters:
            return {
                'success': False,
                'error': 'No predictions available',
                'top_prediction': None,
                'confidence': 0.0,
                'predictions': []
            }
        
        # Format results
        predictions = [
            {
                'letter': letter,
                'confidence': float(score),
                'confidence_percent': f"{score * 100:.2f}%"
            }
            for letter, score in sorted_letters
        ]
        
        return {
            'success': True,
            'top_prediction': predictions[0]['letter'],
            'confidence': predictions[0]['confidence'],
            'predictions': predictions,
            'method': 'weighted_ensemble'
        }
    
    def _voting_ensemble(self, all_predictions: Dict, top_k: int) -> Dict:
        """Majority voting from all models"""
        # Get top prediction from each model
        votes = []
        
        for model_name, pred_result in all_predictions.items():
            top_pred = pred_result.get('top_prediction')
            if top_pred:
                votes.append(top_pred)
        
        if not votes:
            return {
                'success': False,
                'error': 'No votes available',
                'top_prediction': None,
                'confidence': 0.0,
                'predictions': []
            }
        
        # Count votes
        vote_counts = Counter(votes)
        most_common = vote_counts.most_common(top_k)
        
        # Calculate confidence as vote percentage
        total_votes = len(votes)
        predictions = [
            {
                'letter': letter,
                'confidence': float(count / total_votes),
                'confidence_percent': f"{(count / total_votes) * 100:.2f}%",
                'votes': count
            }
            for letter, count in most_common
        ]
        
        return {
            'success': True,
            'top_prediction': predictions[0]['letter'],
            'confidence': predictions[0]['confidence'],
            'predictions': predictions,
            'method': 'voting_ensemble'
        }
    
    def _confidence_threshold_ensemble(
        self, 
        all_predictions: Dict, 
        min_confidence: float,
        top_k: int
    ) -> Dict:
        """Only use high-confidence predictions"""
        high_conf_letters = []
        
        for model_name, pred_result in all_predictions.items():
            predictions = pred_result.get('predictions', [])
            
            for pred in predictions:
                letter = pred.get('letter', '')
                confidence = pred.get('confidence', 0)
                
                if confidence >= min_confidence:
                    high_conf_letters.append((letter, confidence))
        
        if not high_conf_letters:
            # Fallback to weighted ensemble
            return self._weighted_ensemble(all_predictions, top_k)
        
        # Group by letter and average confidence
        letter_confidences = {}
        for letter, conf in high_conf_letters:
            if letter not in letter_confidences:
                letter_confidences[letter] = []
            letter_confidences[letter].append(conf)
        
        # Average confidences
        averaged = [
            (letter, np.mean(confs))
            for letter, confs in letter_confidences.items()
        ]
        
        # Sort by confidence
        sorted_preds = sorted(averaged, key=lambda x: x[1], reverse=True)[:top_k]
        
        predictions = [
            {
                'letter': letter,
                'confidence': float(conf),
                'confidence_percent': f"{conf * 100:.2f}%"
            }
            for letter, conf in sorted_preds
        ]
        
        return {
            'success': True,
            'top_prediction': predictions[0]['letter'],
            'confidence': predictions[0]['confidence'],
            'predictions': predictions,
            'method': 'confidence_threshold_ensemble'
        }


class TemporalSmoother:
    """
    Smooth predictions over time for video streams
    Reduces jitter and improves stability
    """
    
    def __init__(self, window_size: int = 5, min_consensus: int = 3):
        """
        Args:
            window_size: Number of frames to consider
            min_consensus: Minimum occurrences for a letter to be accepted
        """
        self.window_size = window_size
        self.min_consensus = min_consensus
        self.history = []
    
    def add_prediction(self, letter: str, confidence: float) -> Optional[str]:
        """
        Add a prediction and return smoothed result
        
        Args:
            letter: Predicted letter
            confidence: Prediction confidence
            
        Returns:
            Smoothed letter if consensus reached, else None
        """
        # Add to history
        self.history.append((letter, confidence))
        
        # Keep only last N frames
        if len(self.history) > self.window_size:
            self.history = self.history[-self.window_size:]
        
        # Need minimum frames for smoothing
        if len(self.history) < self.min_consensus:
            return None
        
        # Count occurrences
        recent_letters = [l for l, c in self.history]
        letter_counts = Counter(recent_letters)
        
        # Check for consensus
        most_common_letter, count = letter_counts.most_common(1)[0]
        
        if count >= self.min_consensus:
            return most_common_letter
        
        return None
    
    def reset(self):
        """Reset history"""
        self.history = []


# CLI Testing
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python ensemble_predictor.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    print("=" * 70)
    print("Testing Ensemble Predictor")
    print("=" * 70)
    
    # Create ensemble
    ensemble = EnsemblePredictor()
    
    # Test all methods
    methods = ['weighted', 'voting', 'confidence_threshold']
    
    for method in methods:
        print(f"\n{'='*70}")
        print(f"Method: {method.upper()}")
        print('='*70)
        
        result = ensemble.predict(image_path, method=method)
        
        if result['success']:
            print(f"\n✓ Prediction successful")
            print(f"Top Prediction: {result['top_prediction']}")
            print(f"Confidence: {result['confidence']*100:.2f}%")
            print(f"Processing Time: {result['processing_time_ms']}ms")
            print(f"Models Used: {', '.join(result['models_used'])}")
            print(f"\nTop {len(result['predictions'])} Predictions:")
            for i, pred in enumerate(result['predictions'], 1):
                print(f"  {i}. {pred['letter']} - {pred['confidence_percent']}")
        else:
            print(f"\n✗ Prediction failed: {result['error']}")
    
    print("\n" + "=" * 70)
