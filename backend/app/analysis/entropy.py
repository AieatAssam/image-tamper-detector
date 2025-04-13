"""
Entropy-based analysis module for detecting AI-generated images.

This implementation is based on the methodology described in:
'Detecting AI-Generated Images Using Entropy Analysis'
by Fred Rohrer (https://blog.frohrer.com/detecting-ai-generated-images-using-entropy-analysis/)
"""

import numpy as np
import cv2
from pathlib import Path
import io
from typing import Tuple, Optional, Union
from dataclasses import dataclass
from skimage.morphology import disk
from skimage import filters

@dataclass
class EntropyFeatures:
    """Container for entropy analysis features."""
    entropy_red: np.ndarray    # Local entropy map for red channel
    entropy_green: np.ndarray  # Local entropy map for green channel
    entropy_blue: np.ndarray   # Local entropy map for blue channel
    matching_mask: np.ndarray  # Mask of pixels with similar entropy across channels
    uniformity_mask: np.ndarray  # Mask of pixels with uniform entropy patterns

class EntropyAnalyzer:
    def __init__(self, 
                 radius: int = 4,  # Balanced radius for local entropy calculation
                 tolerance: float = 0.12,  # Tolerance for entropy matching
                 matching_threshold: float = 0.35,  # Threshold for AI detection
                 uniformity_threshold: float = 0.2):  # Threshold for local uniformity
        """
        Initialize Entropy analyzer for AI-generated image detection.
        
        Args:
            radius: Radius for the local entropy calculation window
            tolerance: Tolerance for considering entropy values similar across channels
            matching_threshold: Threshold for the proportion of matching pixels to classify as AI-generated
            uniformity_threshold: Threshold for local entropy uniformity
        """
        if radius < 1:
            raise ValueError("Radius must be at least 1")
        if not 0 < tolerance < 1:
            raise ValueError("Tolerance must be between 0 and 1")
        if not 0 < matching_threshold < 1:
            raise ValueError("Matching threshold must be between 0 and 1")
        if not 0 < uniformity_threshold < 1:
            raise ValueError("Uniformity threshold must be between 0 and 1")
            
        self.radius = radius
        self.tolerance = tolerance
        self.matching_threshold = matching_threshold
        self.uniformity_threshold = uniformity_threshold
        self.kernel_size = 2 * radius + 1
        self.selem = disk(radius)  # Structural element for entropy calculation
        
    def _normalize_entropy(self, entropy_map: np.ndarray) -> np.ndarray:
        """Normalize entropy map to uint8 range."""
        # Scale to 0-255 range
        min_val = np.min(entropy_map)
        max_val = np.max(entropy_map)
        if max_val > min_val:
            normalized = ((entropy_map - min_val) * 255 / (max_val - min_val)).astype(np.uint8)
        else:
            normalized = np.zeros_like(entropy_map, dtype=np.uint8)
        return normalized
        
    def _compute_uniformity_mask(self, entropy_maps: list[np.ndarray]) -> np.ndarray:
        """Compute mask of regions with uniform entropy patterns."""
        # Stack entropy maps
        entropy_stack = np.stack(entropy_maps, axis=-1)
        
        # Compute mean entropy across channels
        mean_entropy = np.mean(entropy_stack, axis=-1).astype(np.float32)
        
        # Use OpenCV's blur for fast local mean computation
        local_mean = cv2.blur(mean_entropy, (self.kernel_size, self.kernel_size))
        local_mean2 = cv2.blur(mean_entropy * mean_entropy, (self.kernel_size, self.kernel_size))
        
        # Compute local variance and standard deviation
        local_var = local_mean2 - local_mean * local_mean
        local_std = np.sqrt(np.maximum(local_var, 0))
        
        # Normalize local standard deviation
        local_std = self._normalize_entropy(local_std)
        
        # Create uniformity mask
        uniformity_mask = local_std < (self.uniformity_threshold * 255)
        return uniformity_mask
        
    def _load_image_from_bytes(self, image_bytes: bytes) -> np.ndarray:
        """Load an image from bytes."""
        try:
            # Read image bytes into numpy array
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("Failed to decode image bytes")
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            raise ValueError(f"Error loading image from bytes: {e}")

    def analyze(self, image_input: Union[str, Path, bytes]) -> Tuple[np.ndarray, EntropyFeatures]:
        """
        Analyze an image using entropy-based detection.
        
        Args:
            image_input: Can be one of:
                - Path to the image file (str or Path)
                - Bytes of the image file (bytes)
            
        Returns:
            Tuple containing:
                - Original image as RGB numpy array
                - EntropyFeatures object containing analysis results
                
        Raises:
            FileNotFoundError: If image file doesn't exist
            ValueError: If image can't be processed
        """
        try:
            # Handle different input types
            if isinstance(image_input, (str, Path)):
                image_path = Path(image_input)
                if not image_path.exists():
                    raise FileNotFoundError(f"Image not found: {image_path}")
                # Read image using OpenCV
                image = cv2.imread(str(image_path))
                if image is None:
                    raise ValueError("Failed to read image")
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            elif isinstance(image_input, bytes):
                image_rgb = self._load_image_from_bytes(image_input)
            else:
                raise ValueError("Invalid input type. Must be string, Path, or bytes")
            
            # Convert to uint8 if necessary
            if image_rgb.dtype != np.uint8:
                image_rgb = cv2.convertScaleAbs(image_rgb)
            
            # Calculate local entropy for each channel
            entropy_red = filters.rank.entropy(image_rgb[..., 0], self.selem)
            entropy_green = filters.rank.entropy(image_rgb[..., 1], self.selem)
            entropy_blue = filters.rank.entropy(image_rgb[..., 2], self.selem)
            
            # Normalize entropy maps to uint8
            entropy_red = self._normalize_entropy(entropy_red)
            entropy_green = self._normalize_entropy(entropy_green)
            entropy_blue = self._normalize_entropy(entropy_blue)
            
            # Compare entropy across channels
            entropy_diff_rg = np.abs(entropy_red - entropy_green)
            entropy_diff_rb = np.abs(entropy_red - entropy_blue)
            entropy_diff_gb = np.abs(entropy_green - entropy_blue)
            
            # Create mask where entropy differences are within tolerance
            tolerance_scaled = self.tolerance * 255  # Scale tolerance to uint8 range
            matching_mask = (
                (entropy_diff_rg < tolerance_scaled) & 
                (entropy_diff_rb < tolerance_scaled) & 
                (entropy_diff_gb < tolerance_scaled)
            )
            
            # Compute uniformity mask
            uniformity_mask = self._compute_uniformity_mask([
                entropy_red, entropy_green, entropy_blue
            ])
            
            return image_rgb, EntropyFeatures(
                entropy_red=entropy_red,
                entropy_green=entropy_green,
                entropy_blue=entropy_blue,
                matching_mask=matching_mask,
                uniformity_mask=uniformity_mask
            )
            
        except Exception as e:
            raise ValueError(f"Error during entropy analysis: {e}")
            
    def detect_ai_generated(self, 
                          image_input: Union[str, Path, bytes],
                          overlay_alpha: float = 0.6) -> Tuple[bool, np.ndarray, float]:
        """
        Detect if an image is likely AI-generated and return visualization.
        
        Args:
            image_input: Can be one of:
                - Path to the image file (str or Path)
                - Bytes of the image file (bytes)
            overlay_alpha: Transparency of the visualization overlay (0-1)
            
        Returns:
            Tuple containing:
                - Boolean indicating if image is likely AI-generated
                - Visualization with suspicious regions highlighted
                - Proportion of pixels with matching entropy
                
        Raises:
            FileNotFoundError: If image file doesn't exist
            ValueError: If image can't be processed
        """
        # Analyze the image
        image_rgb, features = self.analyze(image_input)
        
        # Calculate proportion of matching pixels with uniform entropy
        matching_proportion = float(np.mean(
            features.matching_mask & features.uniformity_mask
        ))
        
        # AI-generated images tend to have lower proportions of matching entropy patterns
        is_ai_generated = matching_proportion < self.matching_threshold
        
        # Create visualization
        visualization = image_rgb.copy()
        
        # Create overlay for matching regions (red highlight)
        overlay = np.zeros_like(visualization)
        overlay[features.matching_mask & features.uniformity_mask] = [255, 0, 0]  # Red for matching regions
        
        # Blend overlay with original image
        visualization = cv2.addWeighted(
            visualization,
            1.0,
            overlay,
            overlay_alpha,
            0
        )
        
        return is_ai_generated, visualization, matching_proportion 