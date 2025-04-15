"""
Minimal Photo Response Non-Uniformity (PRNU) uniformity detector for image tampering detection.
Inspired by https://github.com/polimi-ispl/prnu-python/blob/master/prnu/functions.py
and the Binghamton PRNU toolbox.
"""
import numpy as np
import cv2
from pathlib import Path
import io
from typing import Tuple, Optional, List, Union
from scipy import signal
from scipy.interpolate import griddata
import pywt  # Add pywt for wavelet denoising
import logging  # Add logging for debugging
from scipy.ndimage import median_filter, gaussian_filter

def prnu_uniformity(
    image: Union[str, np.ndarray],
    noise_filter_sigma: float = 3.0,
    window_size: int = 64,
    stride: int = 32,
    variance_threshold: float = 0.001
) -> Tuple[bool, float, np.ndarray]:
    """
    Extracts the noise residual from an image, computes the local variance of the noise,
    and returns a uniformity score. Uniform PRNU (low variance) means not tampered,
    inconsistent (high variance) means tampered.

    Args:
        image: Path to image or numpy array (RGB or BGR)
        noise_filter_sigma: Sigma for Gaussian denoising
        window_size: Size of the sliding window for local variance
        stride: Stride for the sliding window
        variance_threshold: Threshold for variance to flag as tampered

    Returns:
        is_tampered: True if image is likely tampered (inconsistent PRNU)
        uniformity_score: Mean local variance of the noise residual
        noise_map: The extracted noise residual (same shape as input)
    """
    # Load image if path
    if isinstance(image, str):
        img = cv2.imread(image, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Could not load image: {image}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif isinstance(image, np.ndarray):
        img = image.copy()
        if img.ndim == 2:
            img = np.stack([img]*3, axis=-1)
        elif img.shape[2] == 4:
            img = img[..., :3]
    else:
        raise ValueError("Input must be a file path or numpy array")

    img = img.astype(np.float32)
    # Extract noise residual (simple denoising as in prnu-python)
    # Use median filter + gaussian filter as in Binghamton toolbox
    denoised = median_filter(img, size=3)
    denoised = gaussian_filter(denoised, sigma=noise_filter_sigma)
    noise = img - denoised
    # Normalize noise
    noise = noise - np.mean(noise)
    # Compute local variance map
    h, w, c = noise.shape
    var_map = np.zeros((h, w), dtype=np.float32)
    for y in range(0, h - window_size + 1, stride):
        for x in range(0, w - window_size + 1, stride):
            window = noise[y:y+window_size, x:x+window_size, :]
            local_var = np.var(window)
            var_map[y:y+window_size, x:x+window_size] += local_var
    # Normalize by number of times each pixel was covered
    count_map = np.zeros((h, w), dtype=np.float32)
    for y in range(0, h - window_size + 1, stride):
        for x in range(0, w - window_size + 1, stride):
            count_map[y:y+window_size, x:x+window_size] += 1
    count_map[count_map == 0] = 1
    var_map /= count_map
    # Uniformity score: mean of local variance
    uniformity_score = float(np.mean(var_map))
    is_tampered = uniformity_score > variance_threshold
    return is_tampered, uniformity_score, noise

class PRNUAnalyzer:
    def __init__(self,
                 noise_filter_sigma: float = 3.0,
                 window_size: int = 64,
                 stride: int = 32,
                 variance_threshold: float = 0.001):
        """
        Initialize PRNU analyzer for tampering detection.
        Args:
            noise_filter_sigma: Sigma for Gaussian denoising
            window_size: Size of the sliding window for local variance
            stride: Stride for the sliding window
            variance_threshold: Threshold for variance to flag as tampered
        """
        self.noise_filter_sigma = noise_filter_sigma
        self.window_size = window_size
        self.stride = stride
        self.variance_threshold = variance_threshold

    def _load_image(self, image_input: Union[str, Path, bytes, np.ndarray]) -> np.ndarray:
        if isinstance(image_input, (str, Path)):
            img = cv2.imread(str(image_input), cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"Could not load image: {image_input}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif isinstance(image_input, bytes):
            nparr = np.frombuffer(image_input, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Failed to decode image bytes")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif isinstance(image_input, np.ndarray):
            img = image_input.copy()
            if img.ndim == 2:
                img = np.stack([img]*3, axis=-1)
            elif img.shape[2] == 4:
                img = img[..., :3]
        else:
            raise ValueError("Input must be a file path, bytes, or numpy array")
        return img.astype(np.float32)

    def detect_tampering(self,
                        image_input: Union[str, Path, bytes, np.ndarray],
                        overlay_alpha: float = 0.8  # Even higher alpha for a more prominent heatmap
                        ) -> Tuple[bool, np.ndarray, float]:
        """
        Detect image tampering using PRNU uniformity (variance of noise residual).
        Args:
            image_input: Path, bytes, or numpy array for the image.
            overlay_alpha: Transparency for the overlay visualization.
        Returns:
            is_tampered: True if image is likely tampered (inconsistent PRNU)
            visualization: Image with high-variance regions highlighted in red (heatmap style)
            uniformity_score: Mean local variance of the noise residual
        """
        img = self._load_image(image_input)
        # Extract noise residual (median + gaussian filter as in prnu-python)
        denoised = median_filter(img, size=3)
        denoised = gaussian_filter(denoised, sigma=self.noise_filter_sigma)
        noise = img - denoised
        noise = noise - np.mean(noise)
        h, w, c = noise.shape
        var_map = np.zeros((h, w), dtype=np.float32)
        for y in range(0, h - self.window_size + 1, self.stride):
            for x in range(0, w - self.window_size + 1, self.stride):
                window = noise[y:y+self.window_size, x:x+self.window_size, :]
                local_var = np.var(window)
                var_map[y:y+self.window_size, x:x+self.window_size] += local_var
        # Normalize by number of times each pixel was covered
        count_map = np.zeros((h, w), dtype=np.float32)
        for y in range(0, h - self.window_size + 1, self.stride):
            for x in range(0, w - self.window_size + 1, self.stride):
                count_map[y:y+self.window_size, x:x+self.window_size] += 1
        count_map[count_map == 0] = 1
        var_map /= count_map
        # Uniformity score: mean of local variance
        uniformity_score = float(np.mean(var_map))
        is_tampered = uniformity_score > self.variance_threshold
        # Visualization: heatmap overlay (continuous, not binary)
        grayscale = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        visualization = cv2.cvtColor(grayscale, cv2.COLOR_GRAY2BGR)
        # Normalize var_map for heatmap (0 = no highlight, 1 = max highlight)
        norm_var_map = (var_map - np.min(var_map)) / (np.max(var_map) - np.min(var_map) + 1e-8)
        # Create heatmap: red channel intensity proportional to normalized variance
        heatmap = np.zeros_like(visualization, dtype=np.uint8)
        heatmap[..., 2] = (norm_var_map * 255).astype(np.uint8)  # Red channel
        # Overlay heatmap on grayscale image
        visualization = cv2.addWeighted(visualization, 1.0, heatmap, overlay_alpha, 0)
        return is_tampered, visualization, uniformity_score

    def extract_noise_residual(self, image: np.ndarray) -> np.ndarray:
        """
        Extract noise residual from an image using wavelet-based denoising.
        
        Args:
            image (np.ndarray): Input image array
            
        Returns:
            np.ndarray: Noise residual
        """
        # Convert to float32 for processing
        img_float = image.astype(np.float32)
        noise_residual = np.zeros_like(img_float)
        
        for channel in range(3):
            img_channel = img_float[..., channel]
            # Wavelet decomposition
            coeffs = pywt.wavedec2(img_channel, 'db8', level=4)
            # Zero out approximation coefficients (keep only details)
            coeffsH = list(coeffs)
            coeffsH[0] = np.zeros_like(coeffsH[0])
            # Reconstruct noise
            noise = pywt.waverec2(coeffsH, 'db8')
            # Crop to original size
            noise = noise[:img_channel.shape[0], :img_channel.shape[1]]
            # Normalize
            noise = noise - np.mean(noise)
            noise_residual[..., channel] = noise
        logging.debug(f"Noise residual shape: {noise_residual.shape}, mean: {np.mean(noise_residual)}, std: {np.std(noise_residual)}")
        return noise_residual
    
    def compute_prnu_pattern(self, noise_residuals: List[np.ndarray]) -> np.ndarray:
        """
        Compute PRNU pattern from multiple noise residuals.
        
        Args:
            noise_residuals (List[np.ndarray]): List of noise residuals from multiple images
            
        Returns:
            np.ndarray: Estimated PRNU pattern
        """
        if not noise_residuals:
            raise ValueError("No noise residuals provided")
            
        # Average noise residuals
        prnu_pattern = np.mean(noise_residuals, axis=0)
        
        # Normalize pattern
        prnu_pattern = (prnu_pattern - np.mean(prnu_pattern)) / (np.std(prnu_pattern) + 1e-10)
        
        return prnu_pattern
    
    def compute_correlation(self, pattern1: np.ndarray, pattern2: np.ndarray) -> float:
        """
        Compute normalized cross-correlation between two patterns.
        
        Args:
            pattern1 (np.ndarray): First pattern
            pattern2 (np.ndarray): Second pattern
            
        Returns:
            float: Correlation coefficient between -1 and 1
        """
        # Ensure patterns have same shape
        if pattern1.shape != pattern2.shape:
            raise ValueError("Patterns must have same shape")
        # For 3D arrays (RGB), compute correlation for each channel and average
        if len(pattern1.shape) == 3:
            correlations = []
            for channel in range(pattern1.shape[2]):
                p1 = pattern1[..., channel].flatten()
                p2 = pattern2[..., channel].flatten()
                # Use np.corrcoef for clarity
                corr = np.corrcoef(p1, p2)[0, 1]
                correlations.append(corr)
            result = float(np.mean(correlations))
            logging.debug(f"Channel correlations: {correlations}, mean: {result}")
            return np.clip(result, -1, 1)
        else:
            p1 = pattern1.flatten()
            p2 = pattern2.flatten()
            corr = np.corrcoef(p1, p2)[0, 1]
            logging.debug(f"Single channel correlation: {corr}")
            return np.clip(float(corr), -1, 1)
    
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

    def analyze(self, image_input: Union[str, Path, bytes]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform PRNU analysis on an image.
        
        Args:
            image_input: Can be one of:
                - Path to the image file (str or Path)
                - Bytes of the image file (bytes)
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple containing:
                - Original image as RGB numpy array
                - PRNU noise residual
                
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
            
            # Extract noise residual
            noise_residual = self.extract_noise_residual(image_rgb)
            
            return image_rgb, noise_residual
            
        except Exception as e:
            raise ValueError(f"Error during PRNU analysis: {e}")
    
    def create_tampering_heatmap(self,
                            reference_pattern: np.ndarray,
                            test_pattern: np.ndarray,
                            window_size: int = 64,
                            stride: int = 32) -> np.ndarray:
        """
        Create a heatmap showing local correlations between reference and test patterns.
        
        Args:
            reference_pattern: Reference PRNU pattern
            test_pattern: Test image PRNU pattern
            window_size: Size of sliding window
            stride: Step size for sliding window
            
        Returns:
            2D array of correlation values (0-1), where lower values indicate potential tampering
        """
        # Ensure both patterns have the same shape
        if reference_pattern.shape != test_pattern.shape:
            # Resize test pattern to match reference pattern
            test_pattern = cv2.resize(
                test_pattern,
                (reference_pattern.shape[1], reference_pattern.shape[0])
            )
            # Add channel dimension if needed
            if len(test_pattern.shape) == 2 and len(reference_pattern.shape) == 3:
                test_pattern = test_pattern[..., np.newaxis]
            elif len(reference_pattern.shape) == 2 and len(test_pattern.shape) == 3:
                reference_pattern = reference_pattern[..., np.newaxis]
        height, width = reference_pattern.shape[:2]
        heatmap = np.zeros((height, width))
        window_count = np.zeros((height, width))
        # Compute local correlations using sliding window
        for y in range(0, height - window_size + 1, stride):
            for x in range(0, width - window_size + 1, stride):
                ref_window = reference_pattern[y:y+window_size, x:x+window_size]
                test_window = test_pattern[y:y+window_size, x:x+window_size]
                correlations = []
                for c in range(ref_window.shape[2]):
                    ref_channel = ref_window[..., c].flatten()
                    test_channel = test_window[..., c].flatten()
                    # Use np.corrcoef for local correlation
                    if np.std(ref_channel) > 1e-6 and np.std(test_channel) > 1e-6:
                        correlation = np.corrcoef(ref_channel, test_channel)[0, 1]
                        correlations.append(correlation)
                # Use average correlation across channels
                if correlations:
                    correlation = np.mean(correlations)
                    # Update heatmap with weighted contribution (no negative clipping)
                    weight = np.ones((window_size, window_size))
                    heatmap[y:y+window_size, x:x+window_size] += correlation * weight
                    window_count[y:y+window_size, x:x+window_size] += weight
        # Normalize heatmap
        valid_mask = window_count > 0
        heatmap[valid_mask] /= window_count[valid_mask]
        # Fill in any gaps using interpolation
        if np.any(~valid_mask):
            y_coords, x_coords = np.nonzero(valid_mask)
            values = heatmap[valid_mask]
            y_grid, x_grid = np.mgrid[0:height, 0:width]
            heatmap = griddata(
                (y_coords, x_coords),
                values,
                (y_grid, x_grid),
                method='nearest'
            )
        logging.debug(f"Heatmap min: {np.min(heatmap)}, max: {np.max(heatmap)}, mean: {np.mean(heatmap)}")
        return heatmap

    def detect_prnu_inconsistency(self,
                                  image_input: Union[str, Path, bytes],
                                  window_size: int = 64,
                                  stride: int = 32,
                                  correlation_threshold: float = 0.5,
                                  overlay_alpha: float = 0.6) -> Tuple[bool, np.ndarray, np.ndarray]:
        """
        Detect PRNU inconsistency within a single image, useful for splicing or AI-generated image detection.
        This method does not require a reference pattern. It computes the global PRNU pattern from the image itself,
        then checks local consistency across the image.

        Args:
            image_input: Path, bytes, or file-like object for the image.
            window_size: Size of the sliding window for local analysis.
            stride: Step size for the sliding window.
            correlation_threshold: Minimum acceptable correlation for a region to be considered consistent.
            overlay_alpha: Transparency for the overlay visualization.

        Returns:
            Tuple of:
                - Boolean indicating if inconsistency (possible tampering/AI) is detected
                - Visualization with inconsistent regions highlighted
                - Heatmap of local PRNU consistency
        """
        # Load and process image
        image_rgb, noise_residual = self.analyze(image_input)
        height, width = noise_residual.shape[:2]

        # Print debug info about image and windowing
        print(f"[PRNU] Image shape: {image_rgb.shape}, window_size: {window_size}, stride: {stride}")

        # Compute global PRNU pattern (mean of noise residual)
        global_pattern = np.mean(noise_residual, axis=(0, 1), keepdims=True)
        # Normalize global pattern
        global_pattern = (global_pattern - np.mean(global_pattern)) / (np.std(global_pattern) + 1e-10)

        # Prepare heatmap
        heatmap = np.zeros((height, width), dtype=np.float32)
        window_count = np.zeros((height, width), dtype=np.float32)

        # --- Robust windowing logic ---
        # Use np.arange for window positions, always process at least one window
        if height < window_size or width < window_size:
            # If the image is smaller than the window, process one window covering the whole image
            y_positions = np.array([0])
            x_positions = np.array([0])
            actual_window_size_y = height
            actual_window_size_x = width
        else:
            y_positions = np.arange(0, height - window_size + 1, stride)
            if y_positions.size == 0:
                y_positions = np.array([0])
            x_positions = np.arange(0, width - window_size + 1, stride)
            if x_positions.size == 0:
                x_positions = np.array([0])
            actual_window_size_y = window_size
            actual_window_size_x = window_size

        # Slide window and compute local PRNU consistency
        for y in y_positions:
            for x in x_positions:
                # For small images, window covers the whole image
                wy = min(actual_window_size_y, height - y)
                wx = min(actual_window_size_x, width - x)
                local_window = noise_residual[y:y+wy, x:x+wx, :]
                print(f"[PRNU] Processing window at (y={y}, x={x}), shape={local_window.shape}")
                # Compute local PRNU pattern
                local_pattern = np.mean(local_window, axis=(0, 1), keepdims=True)
                # Normalize local pattern
                local_pattern = (local_pattern - np.mean(local_pattern)) / (np.std(local_pattern) + 1e-10)
                # Compute correlation for each channel
                correlations = []
                for c in range(local_pattern.shape[2]):
                    g = global_pattern[..., c].flatten()
                    l = local_pattern[..., c].flatten()
                    if np.std(g) > 1e-6 and np.std(l) > 1e-6:
                        corr = np.corrcoef(g, l)[0, 1]
                        correlations.append(corr)
                # Use mean correlation across channels
                if correlations:
                    correlation = float(np.mean(correlations))
                    heatmap[y:y+wy, x:x+wx] += correlation
                    window_count[y:y+wy, x:x+wx] += 1

        print(f"[PRNU] window_count sum: {np.sum(window_count)} (should be > 0)")

        # Robustness: If window_count is all zeros, set heatmap to zeros and warn
        if np.all(window_count == 0):
            print("[PRNU] WARNING: window_count is all zeros. No windows were processed. Setting heatmap to zeros.")
            heatmap[:] = 0
        else:
            # Normalize heatmap
            valid_mask = window_count > 0
            heatmap[valid_mask] /= window_count[valid_mask]
            # Fill in any gaps using interpolation
            if np.any(~valid_mask):
                y_coords, x_coords = np.nonzero(valid_mask)
                values = heatmap[valid_mask]
                y_grid, x_grid = np.mgrid[0:height, 0:width]
                heatmap = griddata(
                    (y_coords, x_coords),
                    values,
                    (y_grid, x_grid),
                    method='nearest'
                )
        # Robustness: If any NaN in heatmap, set to zero and warn
        if np.isnan(heatmap).any():
            print("[PRNU] WARNING: NaN values in heatmap. Setting NaNs to zero.")
            heatmap = np.nan_to_num(heatmap)
        logging.debug(f"PRNU inconsistency heatmap: min={np.min(heatmap)}, max={np.max(heatmap)}, mean={np.mean(heatmap)}")
        # Create mask for inconsistent regions
        inconsistent_mask = heatmap < correlation_threshold
        inconsistent_percentage = float(np.mean(inconsistent_mask))
        is_inconsistent = bool(inconsistent_percentage > 0.05)
        # Visualization
        grayscale = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        visualization = cv2.cvtColor(grayscale, cv2.COLOR_GRAY2BGR)
        overlay = np.zeros_like(visualization)
        overlay[inconsistent_mask] = [0, 0, 255]
        visualization = cv2.addWeighted(
            visualization,
            1.0,
            overlay,
            overlay_alpha,
            0
        )
        visualization[inconsistent_mask] = [0, 0, 255]
        # Add border for inconsistent regions
        if is_inconsistent:
            kernel = np.ones((3,3), np.uint8)
            dilated_mask = cv2.dilate(inconsistent_mask.astype(np.uint8), kernel, iterations=1)
            edge_mask = dilated_mask - inconsistent_mask.astype(np.uint8)
            visualization[edge_mask == 1] = [0, 0, 255]
        return is_inconsistent, visualization, heatmap
