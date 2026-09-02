"""
Error Level Analysis (ELA) module for detecting image tampering.

The raw residual follows the methodology described in:
'A Picture's Worth: Digital Image Analysis and Forensics'
by Dr. Neal Krawetz, presented at Black Hat USA 2007.
The legacy feature classifier below remains a repository-specific heuristic.
"""

import numpy as np
from PIL import Image
import cv2
from pathlib import Path
import io
import logging
from typing import Tuple, Optional, List, Union
from dataclasses import dataclass

logger = logging.getLogger(__name__)

MAX_ANALYSIS_SIDE: int | None = None

@dataclass
class TamperingFeatures:
    """Container for various tampering detection features."""
    edge_discontinuity: float  # Measure of edge consistency
    texture_variance: float   # Variance in texture patterns
    noise_consistency: float  # Consistency of noise patterns
    compression_artifacts: float  # Level of compression artifacts

class ELAAnalyzer:
    def __init__(self,
                 quality: int = 95,
                 resave_quality: Optional[int] = None,
                 max_image_size: Optional[int] = MAX_ANALYSIS_SIDE):
        """
        Initialize ELA analyzer with enhanced feature detection.
        
        Args:
            quality: JPEG quality level for the controlled resave (1-100)
            resave_quality: Backwards-compatible override for the controlled resave
            max_image_size: Optional explicit processing bound
        """
        if not 1 <= quality <= 100:
            raise ValueError("Quality values must be between 1 and 100")
        if resave_quality is not None and not 1 <= resave_quality <= 100:
            raise ValueError("Quality values must be between 1 and 100")
        if max_image_size is not None and max_image_size < 1:
            raise ValueError("Maximum image size must be positive")

        self.quality = quality
        self.resave_quality = quality if resave_quality is None else resave_quality
        self.max_image_size = max_image_size

    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """Preprocess image for analysis."""
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # An explicit bound is a caller policy, not part of paper-defined ELA.
        if self.max_image_size is not None and max(image.size) > self.max_image_size:
            ratio = self.max_image_size / max(image.size)
            new_size = tuple(int(dim * ratio) for dim in image.size)
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        return image

    def _load_image_from_bytes(self, image_bytes: bytes) -> Image.Image:
        """Load an image from bytes."""
        try:
            return Image.open(io.BytesIO(image_bytes))
        except Exception as e:
            raise ValueError(f"Error loading image from bytes: {e}")

    def analyze(self, image_input: Union[str, Path, bytes, Image.Image]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform Error Level Analysis on an image.
        
        Args:
            image_input: Can be one of:
                - Path to the image file (str or Path)
                - Bytes of the image file (bytes)
            
        Returns:
            Tuple containing:
                - Original image as RGB numpy array
                - ELA result as RGB numpy array showing error levels
        """
        try:
            # Handle different input types
            if isinstance(image_input, (str, Path)):
                image_path = Path(image_input)
                if not image_path.exists():
                    raise FileNotFoundError(f"Image not found: {image_path}")
                original = Image.open(image_path)
            elif isinstance(image_input, bytes):
                original = self._load_image_from_bytes(image_input)
            elif isinstance(image_input, Image.Image):
                original = image_input.copy()
            else:
                raise ValueError("Invalid input type. Must be string, Path, bytes, or PIL image")

            # Decode the supplied image without changing its dimensions.
            original = self._preprocess_image(original)
            input_array = np.asarray(original, dtype=np.uint8)

            # Krawetz ELA: compare the decoded input with one controlled resave.
            resaved_buffer = io.BytesIO()
            original.save(resaved_buffer, format="JPEG", quality=self.resave_quality)
            resaved_buffer.seek(0)
            resaved = Image.open(resaved_buffer).convert("RGB")
            resaved_array = np.asarray(resaved, dtype=np.uint8)
            ela = cv2.absdiff(input_array, resaved_array)

            return input_array, ela
            
        except Exception as e:
            raise ValueError(f"Error during ELA analysis: {e}")

    def _compute_edge_discontinuity(self, ela_result: np.ndarray) -> float:
        """Compute edge discontinuity score from ELA result."""
        # Convert to grayscale if needed
        if len(ela_result.shape) == 3:
            gray = cv2.cvtColor(ela_result, cv2.COLOR_RGB2GRAY)
        else:
            gray = ela_result
            
        # Detect edges using Canny with more conservative thresholds
        edges = cv2.Canny(gray, 50, 150)
        
        # Compute edge continuity using morphological operations
        kernel = np.ones((3,3), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=1)
        edge_gaps = cv2.subtract(dilated, edges)
        
        # Normalize the score to [0, 1] range with stronger normalization
        score = np.sum(edge_gaps) / (np.sum(edges) + 1e-6)
        return float(min(1.0, score / 4.0))  # Stronger normalization for more reasonable scores

    def _compute_texture_variance(self, ela_result: np.ndarray) -> float:
        """Compute texture variance score from ELA result."""
        if len(ela_result.shape) == 3:
            gray = cv2.cvtColor(ela_result, cv2.COLOR_RGB2GRAY)
        else:
            gray = ela_result
            
        # Compute local binary pattern (LBP) for texture analysis
        kernel_size = 3
        texture_kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
        local_mean = cv2.filter2D(gray, -1, texture_kernel)
        texture_variance = np.var(np.abs(gray - local_mean))
        
        return texture_variance

    def _compute_noise_consistency(self, ela_result: np.ndarray) -> float:
        """Compute noise consistency score from ELA result."""
        if len(ela_result.shape) == 3:
            gray = cv2.cvtColor(ela_result, cv2.COLOR_RGB2GRAY)
        else:
            gray = ela_result
            
        # Apply median filter to estimate noise
        median = cv2.medianBlur(gray, 3)
        noise = cv2.absdiff(gray, median)
        
        # Compute local noise statistics
        kernel_size = 5
        kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
        local_std = cv2.filter2D(noise, -1, kernel)
        
        return np.std(local_std)

    def _compute_compression_artifacts(self, ela_result: np.ndarray) -> float:
        """Compute compression artifacts score from ELA result."""
        if len(ela_result.shape) == 3:
            gray = cv2.cvtColor(ela_result, cv2.COLOR_RGB2GRAY)
        else:
            gray = ela_result
            
        # Detect block artifacts (common in JPEG compression)
        block_size = 8
        h, w = gray.shape
        blocks_h = h // block_size
        blocks_w = w // block_size
        
        if blocks_h < 2 or blocks_w < 2:
            return 0.0

        horizontal = np.abs(
            gray[: (blocks_h - 1) * block_size, 7 : (blocks_w - 1) * block_size : block_size]
            - gray[: (blocks_h - 1) * block_size, 8 : blocks_w * block_size : block_size]
        ).reshape(blocks_h - 1, block_size, blocks_w - 1).mean(axis=1)
        vertical = np.abs(
            gray[7 : (blocks_h - 1) * block_size : block_size, : (blocks_w - 1) * block_size]
            - gray[8 : blocks_h * block_size : block_size, : (blocks_w - 1) * block_size]
        ).reshape(blocks_h - 1, blocks_w - 1, block_size).mean(axis=2)
        return float(np.mean((horizontal, vertical)))

    def detect_tampering(self, 
                        image_input: Union[str, Path, bytes],
                        edge_threshold: float = 0.45,
                        texture_threshold: float = 2000.0,
                        noise_threshold: float = 25.0,
                        compression_threshold: float = 100.0) -> Tuple[bool, np.ndarray, TamperingFeatures]:
        """
        Detect potential tampering in an image using multiple features.
        
        Args:
            image_input: Can be one of:
                - Path to the image file (str or Path)
                - Bytes of the image file (bytes)
            edge_threshold: Threshold for edge discontinuity
            texture_threshold: Lower bound for texture variance
            noise_threshold: Lower bound for noise consistency
            compression_threshold: Upper bound for compression artifacts
            
        Returns:
            Tuple containing:
                - Boolean indicating if tampering was detected
                - Visualization with suspicious regions highlighted in red over grayscale
                - TamperingFeatures object with detailed scores
        """
        # Perform ELA analysis
        original, ela_result = self.analyze(image_input)
        
        # Compute various features
        features = TamperingFeatures(
            edge_discontinuity=self._compute_edge_discontinuity(ela_result),
            texture_variance=self._compute_texture_variance(ela_result),
            noise_consistency=self._compute_noise_consistency(ela_result),
            compression_artifacts=self._compute_compression_artifacts(ela_result)
        )
        
        # Detect tampering based on multiple features
        edge_violation = features.edge_discontinuity > edge_threshold
        texture_violation = features.texture_variance < texture_threshold  # Look for unusually LOW texture
        compression_violation = features.compression_artifacts > compression_threshold
        noise_violation = features.noise_consistency < noise_threshold and compression_violation
        
        logger.debug(
            "ELA violations: edge=%s texture=%s noise=%s compression=%s",
            edge_violation,
            texture_violation,
            noise_violation,
            compression_violation,
        )
        
        # Count violations with special handling for AI-generated content
        # We consider texture and noise violations together as one strong indicator
        violation_count = sum([
            edge_violation,
            compression_violation,
            (texture_violation or noise_violation)  # Count as 1 if either is present
        ])
        
        logger.debug("ELA total violation count: %s", violation_count)
        
        is_tampered = violation_count >= 2  # At least two types of violations needed
        
        # Convert original image to grayscale while preserving 3 channels for overlay
        grayscale = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
        visualization = cv2.cvtColor(grayscale, cv2.COLOR_GRAY2BGR)
        
        if is_tampered:
            # Create suspicious regions mask
            suspicious_mask = np.zeros_like(original, dtype=np.uint8)
            
            # Add edge discontinuities to mask with more selective thresholding
            if edge_violation:
                edges = cv2.Canny(ela_result, 100, 200)  # Increased thresholds for more selective edge detection
                edges = cv2.dilate(edges, np.ones((2,2), np.uint8), iterations=1)  # Slight dilation
                suspicious_mask |= cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            
            # Add texture and noise anomalies to mask with more selective thresholding
            if texture_violation or noise_violation:
                gray_ela = cv2.cvtColor(ela_result, cv2.COLOR_RGB2GRAY)
                # Use more aggressive thresholding to reduce highlighted areas
                threshold = np.mean(gray_ela) + 2 * np.std(gray_ela)  # Increased threshold
                _, thresh = cv2.threshold(gray_ela, threshold, 255, cv2.THRESH_BINARY)
                # Clean up the mask with morphological operations
                thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
                suspicious_mask |= cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
            
            # Create final mask
            final_mask = np.any(suspicious_mask > 0, axis=2)
            
            # Clean up the mask to reduce noise
            kernel = np.ones((3,3), np.uint8)
            final_mask = cv2.morphologyEx(final_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
            final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)
            
            # Create overlay for suspicious regions (bright red in BGR with reduced opacity)
            overlay = np.zeros_like(visualization)
            overlay[final_mask > 0] = [0, 0, 255]  # BGR format: Red = [0, 0, 255]
            
            # Blend overlay with grayscale image using reduced alpha
            visualization = cv2.addWeighted(
                visualization,
                1.0,
                overlay,
                0.4,  # Reduced alpha for better visibility
                0
            )
            
            # Add subtle red tint to suspicious regions instead of full red
            if np.any(final_mask > 0):
                visualization[final_mask > 0] = cv2.addWeighted(
                    visualization[final_mask > 0],
                    0.7,  # Keep more of the original grayscale
                    np.full_like(visualization[final_mask > 0], [0, 0, 255]),
                    0.3,  # Less red
                    0
                )
            
            # Add a thin border around suspicious regions
            kernel = np.ones((2,2), np.uint8)  # Smaller kernel for thinner border
            dilated_mask = cv2.dilate(final_mask, kernel, iterations=1)
            edge_mask = dilated_mask - final_mask
            
            # Add red border with reduced intensity
            visualization[edge_mask > 0] = [0, 0, 200]  # Slightly less bright red
        
        return is_tampered, visualization, features
