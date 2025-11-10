"""
Image Preprocessing Utilities for ASL Recognition
Improves accuracy through alignment and enhancement
"""

import cv2
import numpy as np
from typing import Tuple, Optional


class ImagePreprocessor:
    """Preprocessing utilities for ASL hand images"""
    
    @staticmethod
    def detect_and_crop_hand(image: np.ndarray, padding: int = 20) -> Optional[np.ndarray]:
        """
        Detect hand region and crop to focus area
        
        Args:
            image: Input image (RGB or BGR)
            padding: Padding around detected hand region
            
        Returns:
            Cropped image focusing on hand, or None if no hand detected
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # Apply binary threshold to detect hand (assuming light skin on dark background)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Find largest contour (assumed to be hand)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get bounding box
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Add padding
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(image.shape[1] - x, w + 2 * padding)
        h = min(image.shape[0] - y, h + 2 * padding)
        
        # Crop image
        cropped = image[y:y+h, x:x+w]
        
        return cropped if cropped.size > 0 else None
    
    @staticmethod
    def enhance_contrast(image: np.ndarray) -> np.ndarray:
        """
        Enhance image contrast using CLAHE
        
        Args:
            image: Input image (RGB or BGR)
            
        Returns:
            Contrast-enhanced image
        """
        # Convert to LAB color space
        if len(image.shape) == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
        else:
            l = image.copy()
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)
        
        # Merge channels
        if len(image.shape) == 3:
            enhanced = cv2.merge([l_enhanced, a, b])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
        else:
            enhanced = l_enhanced
        
        return enhanced
    
    @staticmethod
    def normalize_size(image: np.ndarray, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
        """
        Resize image while maintaining aspect ratio
        
        Args:
            image: Input image
            target_size: Target (width, height)
            
        Returns:
            Resized image with padding if needed
        """
        h, w = image.shape[:2]
        target_w, target_h = target_size
        
        # Calculate scaling factor
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Create canvas with padding
        if len(image.shape) == 3:
            canvas = np.zeros((target_h, target_w, image.shape[2]), dtype=image.dtype)
        else:
            canvas = np.zeros((target_h, target_w), dtype=image.dtype)
        
        # Calculate padding
        x_offset = (target_w - new_w) // 2
        y_offset = (target_h - new_h) // 2
        
        # Place resized image on canvas
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        return canvas
    
    @staticmethod
    def remove_background(image: np.ndarray) -> np.ndarray:
        """
        Remove background using simple thresholding
        
        Args:
            image: Input image (RGB)
            
        Returns:
            Image with background removed (white background)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Apply binary threshold
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Apply morphological operations to clean mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Create white background
        result = np.ones_like(image) * 255
        
        # Copy foreground
        result[mask > 0] = image[mask > 0]
        
        return result
    
    @staticmethod
    def preprocess_for_asl(image: np.ndarray, 
                          crop_hand: bool = True,
                          enhance: bool = True,
                          remove_bg: bool = False,
                          target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
        """
        Complete preprocessing pipeline for ASL recognition
        
        Args:
            image: Input image (RGB)
            crop_hand: Whether to detect and crop hand region
            enhance: Whether to enhance contrast
            remove_bg: Whether to remove background
            target_size: Target size for final image
            
        Returns:
            Preprocessed image
        """
        processed = image.copy()
        
        # Remove background if requested
        if remove_bg:
            processed = ImagePreprocessor.remove_background(processed)
        
        # Crop to hand region if requested
        if crop_hand:
            cropped = ImagePreprocessor.detect_and_crop_hand(processed)
            if cropped is not None:
                processed = cropped
        
        # Enhance contrast if requested
        if enhance:
            processed = ImagePreprocessor.enhance_contrast(processed)
        
        # Normalize size
        processed = ImagePreprocessor.normalize_size(processed, target_size)
        
        return processed


# Test utility
if __name__ == "__main__":
    import argparse
    from PIL import Image
    
    parser = argparse.ArgumentParser(description='Test image preprocessing')
    parser.add_argument('--image', type=str, required=True, help='Path to image file')
    parser.add_argument('--output', type=str, default='preprocessed.jpg', help='Output path')
    args = parser.parse_args()
    
    # Load image
    image = cv2.imread(args.image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Preprocess
    preprocessor = ImagePreprocessor()
    processed = preprocessor.preprocess_for_asl(
        image,
        crop_hand=True,
        enhance=True,
        remove_bg=False
    )
    
    # Save result
    result = cv2.cvtColor(processed, cv2.COLOR_RGB2BGR)
    cv2.imwrite(args.output, result)
    
    print(f"Preprocessed image saved to: {args.output}")
