"""
Enhanced Image Preprocessing for ASL Recognition
Improves accuracy through advanced preprocessing techniques
"""

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    print("Warning: OpenCV not available for preprocessing")

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    print("Warning: NumPy not available")

try:
    from PIL import Image, ImageEnhance, ImageFilter
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("Warning: Pillow not available. Install with: pip install Pillow")

from typing import Union, Tuple


class EnhancedPreprocessor:
    """Advanced preprocessing for sign language images"""
    
    def __init__(self):
        self.target_sizes = {
            'cnn': (200, 200),
            'vit': (224, 224),
            'mediapipe': None  # Original size for MediaPipe
        }
    
    def preprocess_for_model(
        self, 
        image: Union[np.ndarray, Image.Image, str],
        model_type: str = 'cnn',
        enhance: bool = True
    ) -> Union[np.ndarray, Image.Image]:
        """
        Preprocess image for specific model
        
        Args:
            image: Input image (path, numpy array, or PIL Image)
            model_type: 'cnn', 'vit', or 'mediapipe'
            enhance: Apply enhancement techniques
            
        Returns:
            Preprocessed image ready for model
        """
        # Load image
        if isinstance(image, str):
            img = Image.open(image)
        elif isinstance(image, np.ndarray):
            img = Image.fromarray(image)
        else:
            img = image
        
        # Apply enhancements
        if enhance:
            img = self.enhance_image(img)
        
        # Crop to hand region (optional but improves accuracy)
        img = self.auto_crop_hand(img)
        
        # Resize for specific model
        target_size = self.target_sizes.get(model_type)
        if target_size:
            img = img.resize(target_size, Image.Resampling.LANCZOS)
        
        # Convert based on model needs
        if model_type == 'mediapipe':
            # MediaPipe needs RGB numpy array
            if img.mode != 'RGB':
                img = img.convert('RGB')
            return np.array(img)
        else:
            # CNN and ViT work with PIL Images or arrays
            if img.mode != 'RGB':
                img = img.convert('RGB')
            return img
    
    def enhance_image(self, img: Image.Image) -> Image.Image:
        """
        Apply multiple enhancement techniques
        
        Improvements:
        - Contrast enhancement: Makes hand features more distinct
        - Sharpness: Clarifies edges and fingers
        - Brightness normalization: Handles varying lighting
        - Noise reduction: Smooths background artifacts
        """
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # 1. Contrast enhancement (boost hand vs background)
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.3)  # 30% more contrast
        
        # 2. Sharpness enhancement (clearer edges)
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(1.5)  # 50% sharper
        
        # 3. Brightness normalization
        img_array = np.array(img)
        mean_brightness = np.mean(img_array)
        target_brightness = 128
        
        if mean_brightness < 100:  # Too dark
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(target_brightness / mean_brightness)
        elif mean_brightness > 180:  # Too bright
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(target_brightness / mean_brightness)
        
        # 4. Denoise (reduce background noise)
        img = img.filter(ImageFilter.MedianFilter(size=3))
        
        return img
    
    def auto_crop_hand(self, img: Image.Image) -> Image.Image:
        """
        Automatically crop to hand region using skin detection
        Falls back to center crop if hand not found
        """
        img_array = np.array(img)
        
        # Convert to HSV for skin detection
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        
        # Expanded skin color range for better detection
        lower_skin1 = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin1 = np.array([20, 255, 255], dtype=np.uint8)
        lower_skin2 = np.array([170, 20, 70], dtype=np.uint8)
        upper_skin2 = np.array([180, 255, 255], dtype=np.uint8)
        
        # Create masks
        mask1 = cv2.inRange(hsv, lower_skin1, upper_skin1)
        mask2 = cv2.inRange(hsv, lower_skin2, upper_skin2)
        mask = cv2.bitwise_or(mask1, mask2)
        
        # Clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find largest contour (likely the hand)
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            
            # Only crop if hand region is significant
            if area > 5000:
                x, y, w, h = cv2.boundingRect(largest_contour)
                
                # Add padding (20% of dimensions)
                padding_x = int(w * 0.2)
                padding_y = int(h * 0.2)
                
                x = max(0, x - padding_x)
                y = max(0, y - padding_y)
                w = min(img.width - x, w + 2 * padding_x)
                h = min(img.height - y, h + 2 * padding_y)
                
                # Crop to hand region
                return img.crop((x, y, x + w, y + h))
        
        # Fallback: center crop
        width, height = img.size
        crop_size = min(width, height)
        left = (width - crop_size) // 2
        top = (height - crop_size) // 2
        return img.crop((left, top, left + crop_size, top + crop_size))
    
    def augment_for_robustness(self, img: Image.Image) -> list[Image.Image]:
        """
        Create multiple augmented versions for test-time augmentation
        Averaging predictions from these improves accuracy
        """
        augmented = [img]  # Original
        
        # Slight rotation variations
        augmented.append(img.rotate(5, expand=False, fillcolor='white'))
        augmented.append(img.rotate(-5, expand=False, fillcolor='white'))
        
        # Brightness variations
        enhancer = ImageEnhance.Brightness(img)
        augmented.append(enhancer.enhance(0.9))
        augmented.append(enhancer.enhance(1.1))
        
        return augmented


def preprocess_for_cnn(image: Union[np.ndarray, Image.Image, str]) -> Image.Image:
    """Quick function for CNN preprocessing"""
    preprocessor = EnhancedPreprocessor()
    return preprocessor.preprocess_for_model(image, model_type='cnn', enhance=True)


def preprocess_for_vit(image: Union[np.ndarray, Image.Image, str]) -> Image.Image:
    """Quick function for ViT preprocessing"""
    preprocessor = EnhancedPreprocessor()
    return preprocessor.preprocess_for_model(image, model_type='vit', enhance=True)


def preprocess_for_mediapipe(image: Union[np.ndarray, Image.Image, str]) -> np.ndarray:
    """Quick function for MediaPipe preprocessing"""
    preprocessor = EnhancedPreprocessor()
    return preprocessor.preprocess_for_model(image, model_type='mediapipe', enhance=True)


# Test the preprocessor
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python enhanced_preprocessing.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    print("Testing Enhanced Preprocessing...")
    print("=" * 60)
    
    preprocessor = EnhancedPreprocessor()
    
    # Test for each model type
    for model_type in ['cnn', 'vit', 'mediapipe']:
        print(f"\nPreprocessing for {model_type.upper()}...")
        processed = preprocessor.preprocess_for_model(image_path, model_type=model_type)
        
        if isinstance(processed, Image.Image):
            print(f"Output: PIL Image, size={processed.size}, mode={processed.mode}")
            processed.save(f'test_output_{model_type}.jpg')
        else:
            print(f"Output: Numpy array, shape={processed.shape}, dtype={processed.dtype}")
            cv2.imwrite(f'test_output_{model_type}.jpg', cv2.cvtColor(processed, cv2.COLOR_RGB2BGR))
    
    print("\n" + "=" * 60)
    print("Preprocessing complete! Check test_output_*.jpg files")
