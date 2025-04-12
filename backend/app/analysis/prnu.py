"""
Photo Response Non-Uniformity (PRNU) analysis module for image tampering detection.
"""
import numpy as np
import cv2
from pathlib import Path
from typing import Tuple, Optional, List
from scipy import signal

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
    
    def analyze(self, image_path: str | Path) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform PRNU analysis on an image.
        
        Args:
            image_path (str | Path): Path to the image file
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple containing:
                - Original image as RGB numpy array
                - PRNU noise residual
                
        Raises:
            FileNotFoundError: If image file doesn't exist
            ValueError: If image can't be processed
        """
        # Convert path to Path object
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
            
        try:
            # Read image using OpenCV
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError("Failed to read image")
                
            # Convert to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
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
        Create a tampering heatmap by computing local correlations.
        
        Args:
            reference_pattern (np.ndarray): Reference PRNU pattern
            test_pattern (np.ndarray): Test image PRNU pattern
            window_size (int): Size of sliding window
            stride (int): Stride for sliding window
            
        Returns:
            np.ndarray: Heatmap showing local correlation values
        """
        # Ensure both patterns have the same shape
        if reference_pattern.shape != test_pattern.shape:
            # Resize test pattern to match reference pattern
            if len(reference_pattern.shape) == 3:
                test_pattern = cv2.resize(test_pattern, 
                                        (reference_pattern.shape[1], reference_pattern.shape[0]))
            else:
                raise ValueError("Patterns must have same shape and dimensions")
        
        height, width = reference_pattern.shape[:2]
        heatmap = np.zeros((height, width))
        
        for y in range(0, height - window_size + 1, stride):
            for x in range(0, width - window_size + 1, stride):
                # Extract windows
                ref_window = reference_pattern[y:y+window_size, x:x+window_size]
                test_window = test_pattern[y:y+window_size, x:x+window_size]
                
                # Compute local correlation
                correlation = self.compute_correlation(ref_window, test_window)
                
                # Normalize correlation to [0, 1] range
                correlation = (correlation + 1) / 2
                
                # Update heatmap
                heatmap[y:y+window_size, x:x+window_size] = correlation
                
        return heatmap
