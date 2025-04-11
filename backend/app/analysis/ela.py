"""
Error Level Analysis (ELA) module for detecting image tampering.
"""
import os
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
from typing import Tuple, Optional

class ELAAnalyzer:
    def __init__(self, quality: int = 90, scale_factor: int = 15):
        """
        Initialize ELA analyzer.
        
        Args:
            quality (int): JPEG save quality (1-100) for recompression
            scale_factor (int): Multiplication factor to make ELA differences more visible
        """
        if not 1 <= quality <= 100:
            raise ValueError("Quality must be between 1 and 100")
        self.quality = quality
        self.scale_factor = scale_factor

    def analyze(self, image_path: str | Path) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform Error Level Analysis on an image.
        
        Args:
            image_path (str | Path): Path to the image file
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple containing:
                - Original image as RGB numpy array
                - ELA result as grayscale numpy array
        
        Raises:
            FileNotFoundError: If image file doesn't exist
            ValueError: If image can't be processed
        """
        # Convert path to Path object
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Read original image
        try:
            original = Image.open(image_path)
            if original.mode != 'RGB':
                original = original.convert('RGB')
        except Exception as e:
            raise ValueError(f"Failed to open image: {e}")

        # Create temporary path for resaved image
        temp_path = image_path.parent / f"temp_{image_path.name}"
        
        try:
            # Save with specified quality
            original.save(temp_path, 'JPEG', quality=self.quality)
            
            # Open resaved image
            resaved = Image.open(temp_path)
            
            # Convert both to numpy arrays
            original_array = np.array(original)
            resaved_array = np.array(resaved)
            
            # Calculate absolute difference and scale
            ela = np.abs(original_array - resaved_array) * self.scale_factor
            
            # Convert to grayscale for better visualization
            ela_gray = cv2.cvtColor(ela, cv2.COLOR_RGB2GRAY)
            
            return original_array, ela_gray
            
        except Exception as e:
            raise ValueError(f"Error during ELA analysis: {e}")
        
        finally:
            # Clean up temporary file
            if temp_path.exists():
                os.remove(temp_path)
    
    def get_threshold_mask(self, ela_result: np.ndarray, 
                          threshold: int = 50) -> np.ndarray:
        """
        Create a binary mask of potentially tampered regions.
        
        Args:
            ela_result (np.ndarray): ELA result from analyze()
            threshold (int): Threshold value for binary mask
            
        Returns:
            np.ndarray: Binary mask where 1 indicates potential tampering
        """
        return (ela_result > threshold).astype(np.uint8)

    def overlay_heatmap(self, original: np.ndarray, 
                       ela_result: np.ndarray, 
                       alpha: float = 0.5) -> np.ndarray:
        """
        Create a heatmap overlay of ELA results on the original image.
        
        Args:
            original (np.ndarray): Original image array
            ela_result (np.ndarray): ELA result from analyze()
            alpha (float): Transparency of the overlay (0-1)
            
        Returns:
            np.ndarray: Image with ELA heatmap overlay
        """
        # Create heatmap
        heatmap = cv2.applyColorMap(ela_result, cv2.COLORMAP_JET)
        
        # Ensure original is in BGR for OpenCV
        if len(original.shape) == 3 and original.shape[2] == 3:
            original_bgr = cv2.cvtColor(original, cv2.COLOR_RGB2BGR)
        else:
            original_bgr = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
            
        # Blend images
        return cv2.addWeighted(original_bgr, 1-alpha, heatmap, alpha, 0)
