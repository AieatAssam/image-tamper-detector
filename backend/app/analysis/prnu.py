"""
Photo Response Non-Uniformity (PRNU) analysis module for image tampering detection.
"""
import numpy as np
import cv2
from pathlib import Path
import io
from typing import Tuple, Optional, List, Union
from scipy import signal
from scipy.interpolate import griddata

class PRNUAnalyzer:
    def __init__(self, 
                 wavelet: str = 'db8', 
                 levels: int = 4,
                 sigma: float = 5.0):
        """
        Initialize PRNU analyzer.
        
        Args:
            wavelet (str): Wavelet type for noise extraction
            levels (int): Number of wavelet decomposition levels
            sigma (float): Sigma for Gaussian filtering
        """
        self.wavelet = wavelet
        self.levels = levels
        self.sigma = sigma
        
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
        
        # Process each color channel separately
        noise_residual = np.zeros_like(img_float)
        
        for channel in range(3):
            # Extract channel
            img_channel = img_float[..., channel]
            
            # Apply Gaussian filtering
            img_blur = cv2.GaussianBlur(img_channel, (0, 0), self.sigma)
            
            # Calculate noise residual
            channel_residual = img_channel - img_blur
            
            # Normalize the residual
            channel_residual = channel_residual - np.mean(channel_residual)
            
            noise_residual[..., channel] = channel_residual
            
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
                p1 = pattern1[..., channel]
                p2 = pattern2[..., channel]
                
                # Normalize patterns
                p1_norm = (p1 - np.mean(p1)) / (np.std(p1) + 1e-10)
                p2_norm = (p2 - np.mean(p2)) / (np.std(p2) + 1e-10)
                
                # Compute normalized cross-correlation
                corr = np.sum(p1_norm * p2_norm) / (p1_norm.size - 1)
                correlations.append(corr)
                
            return np.clip(np.mean(correlations), -1, 1)
        else:
            # Normalize patterns
            p1_norm = (pattern1 - np.mean(pattern1)) / (np.std(pattern1) + 1e-10)
            p2_norm = (pattern2 - np.mean(pattern2)) / (np.std(pattern2) + 1e-10)
            
            # Compute normalized cross-correlation
            correlation = np.sum(p1_norm * p2_norm) / (p1_norm.size - 1)
            return np.clip(correlation, -1, 1)
    
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
                # Extract windows
                ref_window = reference_pattern[y:y+window_size, x:x+window_size]
                test_window = test_pattern[y:y+window_size, x:x+window_size]
                
                # Compute correlation for each color channel
                correlations = []
                for c in range(ref_window.shape[2]):
                    ref_channel = ref_window[..., c].flatten()
                    test_channel = test_window[..., c].flatten()
                    
                    # Center the data
                    ref_channel = ref_channel - np.mean(ref_channel)
                    test_channel = test_channel - np.mean(test_channel)
                    
                    # Compute normalized correlation
                    norm_ref = np.linalg.norm(ref_channel)
                    norm_test = np.linalg.norm(test_channel)
                    
                    if norm_ref > 0 and norm_test > 0:
                        correlation = np.dot(ref_channel, test_channel) / (norm_ref * norm_test)
                        correlations.append(max(0, correlation))  # Clip negative correlations
                
                # Use average correlation across channels
                if correlations:
                    correlation = np.mean(correlations)
                    
                    # Update heatmap with weighted contribution
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
        
        return heatmap

    def detect_tampering(self,
                        image_input: Union[str, Path, bytes],
                        reference_pattern: Optional[np.ndarray] = None,
                        correlation_threshold: float = 0.5,
                        window_size: int = 64,
                        stride: int = 32,
                        overlay_alpha: float = 0.6) -> Tuple[bool, np.ndarray]:
        """
        Analyze an image for tampering and return a decision with visualization.
        
        Args:
            image_input: Can be one of:
                - Path to the image file (str or Path)
                - Bytes of the image file (bytes)
            reference_pattern: Reference PRNU pattern (must be a numpy array)
            correlation_threshold: Threshold for correlation values (0-1).
                Lower values indicate potential tampering
            window_size: Size of sliding window for local analysis
            stride: Stride for sliding window
            overlay_alpha: Transparency of the overlay (0-1)
            
        Returns:
            Tuple containing:
                - Boolean indicating if tampering was detected
                - Visualization with tampered regions highlighted
                
        Raises:
            FileNotFoundError: If image file does not exist
            ValueError: If image is invalid or reference pattern has wrong type
        """
        # Analyze the image
        image_rgb, noise_residual = self.analyze(image_input)
        
        # If reference pattern provided, validate its type
        if reference_pattern is not None and not isinstance(reference_pattern, np.ndarray):
            raise ValueError("Reference pattern must be a numpy array")
        
        # If no reference pattern provided, use the image's own pattern
        if reference_pattern is None:
            reference_pattern = noise_residual
        
        # Create tampering heatmap
        heatmap = self.create_tampering_heatmap(
            reference_pattern,
            noise_residual,
            window_size=window_size,
            stride=stride
        )
        
        # Create visualization
        # Convert heatmap to color visualization (red indicates tampering)
        heatmap_vis = np.zeros((image_rgb.shape[0], image_rgb.shape[1], 3), dtype=np.uint8)
        
        # Resize heatmap to match image dimensions
        heatmap_resized = cv2.resize(
            heatmap,
            (image_rgb.shape[1], image_rgb.shape[0]),
            interpolation=cv2.INTER_LINEAR
        )
        
        # Create visualization mask (red for low correlation, indicating tampering)
        heatmap_vis[..., 2] = ((1 - heatmap_resized) * 255).astype(np.uint8)  # Red channel
        
        # Create mask for potentially tampered regions
        tampered_mask = heatmap_resized < correlation_threshold
        
        # Calculate percentage of potentially tampered pixels
        tampered_percentage = float(np.mean(tampered_mask))  # Convert to Python float
        is_tampered = bool(tampered_percentage > 0.05)  # Convert to Python bool
        
        # Create final visualization
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        
        # Create overlay by blending the original image with the heatmap visualization
        overlay = cv2.addWeighted(
            image_bgr,
            1.0,
            heatmap_vis,
            overlay_alpha,
            0
        )
        
        # Create final visualization by combining original and overlay based on tampered regions
        visualization = image_bgr.copy()
        visualization[tampered_mask] = overlay[tampered_mask]
        
        # Add a border around tampered regions for better visibility
        if is_tampered:
            # Create a dilated mask to highlight boundaries
            kernel = np.ones((3,3), np.uint8)
            dilated_mask = cv2.dilate(tampered_mask.astype(np.uint8), kernel, iterations=1)
            edge_mask = dilated_mask - tampered_mask.astype(np.uint8)
            
            # Add red border
            visualization[edge_mask == 1] = [0, 0, 255]  # BGR format
        
        return is_tampered, visualization
