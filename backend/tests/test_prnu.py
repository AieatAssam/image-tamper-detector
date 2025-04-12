"""
Tests for the PRNU analysis module.
"""
import pytest
import numpy as np
import cv2
from pathlib import Path
from ..app.analysis.prnu import PRNUAnalyzer

@pytest.fixture
def prnu_analyzer():
    """Create a PRNUAnalyzer instance for testing."""
    return PRNUAnalyzer()

@pytest.fixture
def original_image_path():
    """Path to an original image for testing."""
    return Path("data/samples/original/landscape_original.jpg")

@pytest.fixture
def tampered_image_path():
    """Path to a tampered image for testing."""
    return Path("data/samples/tampered/landscape_copy_paste.jpg")

@pytest.fixture
def ai_generated_image_path():
    """Path to an AI-generated image for testing."""
    return Path("data/samples/tampered/gpt-4o-generated-receipt-01.png")

def test_init():
    """Test PRNUAnalyzer initialization."""
    analyzer = PRNUAnalyzer(wavelet='db4', levels=3, sigma=3.0)
    assert analyzer.wavelet == 'db4'
    assert analyzer.levels == 3
    assert analyzer.sigma == 3.0

def test_extract_noise_residual(prnu_analyzer, original_image_path):
    """Test noise residual extraction using a real image."""
    # Read the image
    image = cv2.imread(str(original_image_path))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    noise_residual = prnu_analyzer.extract_noise_residual(image_rgb)
    
    # Check shape and type
    assert noise_residual.shape == image_rgb.shape
    assert noise_residual.dtype == np.float32
    
    # Check that noise residual has non-zero values
    assert not np.allclose(noise_residual, 0)
    
    # Check that each channel's residual has approximately zero mean
    for channel in range(3):
        assert np.abs(np.mean(noise_residual[..., channel])) < 1e-5

def test_compute_prnu_pattern(prnu_analyzer, original_image_path):
    """Test PRNU pattern computation using a real image."""
    # Read the image
    image = cv2.imread(str(original_image_path))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create multiple noise residuals by slightly modifying the image
    noise_residuals = []
    for _ in range(3):
        # Add small random noise to create slightly different versions
        noisy_image = image_rgb + np.random.normal(0, 2, image_rgb.shape).astype(np.uint8)
        noise_residual = prnu_analyzer.extract_noise_residual(noisy_image)
        noise_residuals.append(noise_residual)
    
    pattern = prnu_analyzer.compute_prnu_pattern(noise_residuals)
    
    # Check shape and type
    assert pattern.shape == image_rgb.shape
    assert pattern.dtype == np.float32
    
    # Check normalization
    assert np.abs(np.mean(pattern)) < 1e-5
    assert np.abs(np.std(pattern) - 1.0) < 1e-5

def test_compute_correlation_same_image(prnu_analyzer, original_image_path):
    """Test correlation computation with the same image."""
    # Read the image
    image = cv2.imread(str(original_image_path))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Extract noise patterns
    pattern1 = prnu_analyzer.extract_noise_residual(image_rgb)
    pattern2 = prnu_analyzer.extract_noise_residual(image_rgb)  # Same image
    
    correlation = prnu_analyzer.compute_correlation(pattern1, pattern2)
    
    # Correlation should be high for the same image
    assert -1 <= correlation <= 1
    assert correlation > 0.5  # High correlation expected

def test_compute_correlation_different_images(prnu_analyzer, original_image_path, ai_generated_image_path):
    """Test correlation computation with different images."""
    # Read the images
    image1 = cv2.imread(str(original_image_path))
    image1_rgb = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    
    image2 = cv2.imread(str(ai_generated_image_path))
    image2_rgb = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
    
    # Resize second image to match first image's dimensions
    image2_rgb = cv2.resize(image2_rgb, (image1_rgb.shape[1], image1_rgb.shape[0]))
    
    # Extract noise patterns
    pattern1 = prnu_analyzer.extract_noise_residual(image1_rgb)
    pattern2 = prnu_analyzer.extract_noise_residual(image2_rgb)
    
    correlation = prnu_analyzer.compute_correlation(pattern1, pattern2)
    
    # Correlation should be lower for different images
    assert -1 <= correlation <= 1
    assert correlation < 0.5  # Lower correlation expected

def test_create_tampering_heatmap(prnu_analyzer, original_image_path, tampered_image_path):
    """Test tampering heatmap creation with real tampered images."""
    # Read the images
    original = cv2.imread(str(original_image_path))
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    
    tampered = cv2.imread(str(tampered_image_path))
    tampered_rgb = cv2.cvtColor(tampered, cv2.COLOR_BGR2RGB)
    
    # Extract noise patterns
    reference_pattern = prnu_analyzer.extract_noise_residual(original_rgb)
    test_pattern = prnu_analyzer.extract_noise_residual(tampered_rgb)
    
    heatmap = prnu_analyzer.create_tampering_heatmap(
        reference_pattern,
        test_pattern,
        window_size=64,
        stride=32
    )
    
    # Check shape
    assert heatmap.shape[:2] == original_rgb.shape[:2]
    
    # Check that heatmap has values between 0 and 1
    assert np.all(heatmap >= 0)
    assert np.all(heatmap <= 1)
    
    # Check that there are variations in the heatmap
    assert np.std(heatmap) > 0

def test_analyze_invalid_path(prnu_analyzer):
    """Test analyze with invalid image path."""
    with pytest.raises(FileNotFoundError):
        prnu_analyzer.analyze("nonexistent_image.jpg")

def test_compute_prnu_pattern_empty_list(prnu_analyzer):
    """Test compute_prnu_pattern with empty list."""
    with pytest.raises(ValueError):
        prnu_analyzer.compute_prnu_pattern([])

def test_analyze_real_image(prnu_analyzer, original_image_path):
    """Test the complete analysis pipeline with a real image."""
    # Run analysis
    original_image, noise_residual = prnu_analyzer.analyze(original_image_path)
    
    # Check that we got valid outputs
    assert isinstance(original_image, np.ndarray)
    assert isinstance(noise_residual, np.ndarray)
    assert original_image.shape == noise_residual.shape
    assert noise_residual.dtype == np.float32
    
    # Check that noise residual has meaningful values
    assert not np.allclose(noise_residual, 0)
    assert np.all(np.isfinite(noise_residual))  # No NaN or inf values 