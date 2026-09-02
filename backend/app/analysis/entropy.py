"""
Entropy-based analysis module for detecting AI-generated images.

This implementation is based on the methodology described in:
'Detecting AI-Generated Images Using Entropy Analysis'
by Fred Rohrer (https://blog.frohrer.com/detecting-ai-generated-images-using-entropy-analysis/)
"""

import numpy as np
import cv2
from pathlib import Path
import json
from typing import Tuple, Union
from dataclasses import dataclass
from skimage.morphology import disk
from skimage import filters

MAX_ANALYSIS_SIDE = 1024


def _analysis_image(image: np.ndarray) -> np.ndarray:
    longest = max(image.shape[:2])
    if longest <= MAX_ANALYSIS_SIDE:
        return image
    ratio = MAX_ANALYSIS_SIDE / float(longest)
    return cv2.resize(
        image,
        (max(1, round(image.shape[1] * ratio)), max(1, round(image.shape[0] * ratio))),
        interpolation=cv2.INTER_AREA,
    )

@dataclass
class EntropyFeatures:
    """Container for entropy analysis features."""
    entropy_red: np.ndarray    # Local entropy map for red channel
    entropy_green: np.ndarray  # Local entropy map for green channel
    entropy_blue: np.ndarray   # Local entropy map for blue channel
    matching_mask: np.ndarray  # Mask of pixels with similar entropy across channels

def _legacy_threshold() -> float:
    calibration = json.loads(Path(__file__).with_name("calibration.json").read_text())
    return float(calibration["legacy"]["entropy"]["matching_threshold"])


class EntropyAnalyzer:
    def __init__(self,
                 radius: int = 5,
                 tolerance: float = 0.1,
                 matching_threshold: float | None = None):
        """
        Initialize Entropy analyzer for AI-generated image detection.

        Args:
            radius: Radius for the local entropy calculation window
            tolerance: Tolerance for considering entropy values similar across channels
            matching_threshold: Repository threshold for the proportion of matching pixels
        """
        matching_threshold = _legacy_threshold() if matching_threshold is None else matching_threshold
        if radius < 1:
            raise ValueError("Radius must be at least 1")
        if not 0 < tolerance < 1:
            raise ValueError("Tolerance must be between 0 and 1")
        if not 0 < matching_threshold < 1:
            raise ValueError("Matching threshold must be between 0 and 1")
        self.radius = radius
        self.tolerance = tolerance
        self.matching_threshold = matching_threshold
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

    def analyze(self, image_input: Union[str, Path, bytes, np.ndarray]) -> Tuple[np.ndarray, EntropyFeatures]:
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
            elif isinstance(image_input, np.ndarray):
                image_rgb = image_input.copy()
                if image_rgb.ndim == 2:
                    image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_GRAY2RGB)
                elif image_rgb.ndim == 3 and image_rgb.shape[2] == 4:
                    image_rgb = image_rgb[..., :3]
            else:
                raise ValueError("Invalid input type. Must be string, Path, bytes, or numpy array")

            # Convert to uint8 if necessary
            if image_rgb.dtype != np.uint8:
                image_rgb = cv2.convertScaleAbs(image_rgb)

            if isinstance(image_input, np.ndarray):
                image_rgb = _analysis_image(image_rgb)

            # Calculate local entropy for each channel
            raw_entropy_red = filters.rank.entropy(image_rgb[..., 0], self.selem)
            raw_entropy_green = filters.rank.entropy(image_rgb[..., 1], self.selem)
            raw_entropy_blue = filters.rank.entropy(image_rgb[..., 2], self.selem)

            # Normalize entropy maps to uint8
            entropy_red = self._normalize_entropy(raw_entropy_red)
            entropy_green = self._normalize_entropy(raw_entropy_green)
            entropy_blue = self._normalize_entropy(raw_entropy_blue)

            # Compare raw entropy values; normalized uint8 subtraction wraps modulo 256.
            entropy_diff_rg = np.abs(raw_entropy_red.astype(np.float32) - raw_entropy_green.astype(np.float32))
            entropy_diff_rb = np.abs(raw_entropy_red.astype(np.float32) - raw_entropy_blue.astype(np.float32))
            entropy_diff_gb = np.abs(raw_entropy_green.astype(np.float32) - raw_entropy_blue.astype(np.float32))

            # Create mask where entropy differences are within tolerance
            matching_mask = (
                (entropy_diff_rg < self.tolerance) &
                (entropy_diff_rb < self.tolerance) &
                (entropy_diff_gb < self.tolerance)
            )

            return image_rgb, EntropyFeatures(
                entropy_red=entropy_red,
                entropy_green=entropy_green,
                entropy_blue=entropy_blue,
                matching_mask=matching_mask,
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
                - Visualization with suspicious regions highlighted in red over grayscale
                - Proportion of pixels with matching entropy

        Raises:
            FileNotFoundError: If image file doesn't exist
            ValueError: If image can't be processed
        """
        # Analyze the image
        image_rgb, features = self.analyze(image_input)

        # The cited procedure only marks pixels whose raw channel entropies match.
        suspicious_regions = features.matching_mask
        matching_proportion = float(np.mean(suspicious_regions))

        # AI-generated images tend to have lower proportions of matching entropy patterns
        is_ai_generated = matching_proportion < self.matching_threshold

        # Convert original image to grayscale while preserving 3 channels for overlay
        grayscale = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        visualization = cv2.cvtColor(grayscale, cv2.COLOR_GRAY2BGR)  # Note: Using BGR here

        # Create overlay for suspicious regions (bright red highlight in BGR)
        overlay = np.zeros_like(visualization)
        overlay[suspicious_regions] = [0, 0, 255]  # BGR format: Red = [0, 0, 255]

        # Blend overlay with grayscale image
        # Use additive blending to make red regions more visible
        visualization = cv2.addWeighted(
            visualization,
            1.0,
            overlay,
            overlay_alpha,
            0
        )

        # Enhance red channel in suspicious regions to make it more prominent
        visualization[suspicious_regions] = [0, 0, 255]  # BGR format: Red = [0, 0, 255]

        return is_ai_generated, visualization, matching_proportion
