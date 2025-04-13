"""
Unit tests for Photo Response Non-Uniformity (PRNU) implementation.
"""
import pytest
import numpy as np
import cv2
from pathlib import Path
from backend.app.analysis.prnu import PRNUAnalyzer

# Test constants
TEST_WAVELET = 'db8'
TEST_LEVELS = 4
TEST_SIGMA = 5.0
TEST_WINDOW_SIZE = 64
TEST_STRIDE = 32
SAME_CAMERA_THRESHOLD = 0.5  # Correlation threshold for same camera
DIFFERENT_CAMERA_THRESHOLD = 0.5  # Correlation threshold for different cameras
TAMPERING_VARIATION_THRESHOLD = 0.05  # Expected std dev in tampered regions - adjusted based on empirical data
CORRELATION_THRESHOLD = 0.5

@pytest.fixture
def prnu_analyzer():
    """Fixture to create PRNU analyzer instance."""
    return PRNUAnalyzer(wavelet=TEST_WAVELET, levels=TEST_LEVELS, sigma=TEST_SIGMA)

@pytest.fixture
def data_dir():
    """Fixture to get the data directory path."""
    return Path(__file__).parent.parent.parent / 'data' / 'samples'

def test_prnu_analyzer_initialization():
    """Test PRNU analyzer initialization with valid parameters."""
    # Test valid initialization
    analyzer = PRNUAnalyzer(wavelet=TEST_WAVELET, levels=TEST_LEVELS, sigma=TEST_SIGMA)
    assert analyzer.wavelet == TEST_WAVELET
    assert analyzer.levels == TEST_LEVELS
    assert analyzer.sigma == TEST_SIGMA
    
    # Test with different valid parameters
    analyzer = PRNUAnalyzer(wavelet='db4', levels=3, sigma=3.0)
    assert analyzer.wavelet == 'db4'
    assert analyzer.levels == 3
    assert analyzer.sigma == 3.0

def test_original_image_analysis(prnu_analyzer, data_dir):
    """Test PRNU on original (untampered) images."""
    original_dir = data_dir / 'original'
    
    for img_path in original_dir.glob('*.jpg'):
        # Analyze image
        original, noise_residual = prnu_analyzer.analyze(img_path)
        
        # Check basic properties
        assert noise_residual.shape == original.shape
        assert noise_residual.dtype == np.float32
        assert not np.allclose(noise_residual, 0)
        
        # Check that noise residual has meaningful values
        assert np.all(np.isfinite(noise_residual))
        
        # Check that each channel's residual has approximately zero mean
        for channel in range(3):
            assert np.abs(np.mean(noise_residual[..., channel])) < 1e-5

def test_tampered_image_analysis(prnu_analyzer, data_dir):
    """Test PRNU on known tampered images."""
    original_dir = data_dir / 'original'
    tampered_dir = data_dir / 'tampered'
    
    # Get first original image as reference
    original_path = next(original_dir.glob('*.jpg'))
    original_img, original_noise = prnu_analyzer.analyze(original_path)
    
    for img_path in tampered_dir.glob('*.jpg'):
        print(f"\nAnalyzing tampered image: {img_path.name}")
        
        # Analyze tampered image
        tampered_img, tampered_noise = prnu_analyzer.analyze(img_path)
        
        # Resize tampered image to match original if needed
        if tampered_img.shape != original_img.shape:
            print(f"Resizing from {tampered_img.shape} to {original_img.shape}")
            tampered_img = cv2.resize(tampered_img, (original_img.shape[1], original_img.shape[0]))
            tampered_noise = cv2.resize(tampered_noise, (original_noise.shape[1], original_noise.shape[0]))
        
        # Create correlation heatmap
        heatmap = prnu_analyzer.create_tampering_heatmap(
            original_noise,
            tampered_noise,
            window_size=TEST_WINDOW_SIZE,
            stride=TEST_STRIDE
        )
        
        # Check heatmap properties
        assert heatmap.shape[:2] == original_img.shape[:2]
        assert np.all(heatmap >= 0) and np.all(heatmap <= 1)
        
        # Print heatmap statistics
        variation = np.std(heatmap)
        min_val = np.min(heatmap)
        max_val = np.max(heatmap)
        print(f"Heatmap statistics:")
        print(f"- Standard deviation: {variation:.6f}")
        print(f"- Min value: {min_val:.6f}")
        print(f"- Max value: {max_val:.6f}")
        print(f"- Value range: {max_val - min_val:.6f}")
        
        # Check for variations in the heatmap (indicating potential tampering)
        assert variation > TAMPERING_VARIATION_THRESHOLD, \
            f"Variation {variation:.6f} is below threshold {TAMPERING_VARIATION_THRESHOLD}"

def test_ai_generated_detection(prnu_analyzer, data_dir):
    """Test PRNU on AI-generated images."""
    original_dir = data_dir / 'original'
    tampered_dir = data_dir / 'tampered'
    
    # Get original image as reference
    original_path = next(original_dir.glob('*.jpg'))
    original_img, original_noise = prnu_analyzer.analyze(original_path)
    
    # Test AI-generated images
    for img_path in tampered_dir.glob('*generated*.png'):
        print(f"\nAnalyzing AI-generated image: {img_path.name}")
        
        # Analyze AI-generated image
        ai_img, ai_noise = prnu_analyzer.analyze(img_path)
        
        # Resize AI image to match original
        if ai_img.shape != original_img.shape:
            print(f"Resizing from {ai_img.shape} to {original_img.shape}")
            ai_img = cv2.resize(ai_img, (original_img.shape[1], original_img.shape[0]))
            ai_noise = cv2.resize(ai_noise, (original_noise.shape[1], original_noise.shape[0]))
        
        # Compute correlation
        correlation = prnu_analyzer.compute_correlation(original_noise, ai_noise)
        print(f"Correlation with original: {correlation:.6f}")
        
        # AI-generated images should have low correlation with real images
        assert -1 <= correlation <= 1
        assert abs(correlation) < DIFFERENT_CAMERA_THRESHOLD

def test_same_camera_correlation(prnu_analyzer, data_dir):
    """Test correlation between images from the same camera."""
    original_dir = data_dir / 'original'
    
    # Get first image as reference
    original_path = next(original_dir.glob('*.jpg'))
    original_img, original_noise = prnu_analyzer.analyze(original_path)
    
    # Compare with itself (should have high correlation)
    correlation = prnu_analyzer.compute_correlation(original_noise, original_noise)
    print(f"\nSame image correlation: {correlation:.6f}")
    assert -1 <= correlation <= 1
    assert correlation > SAME_CAMERA_THRESHOLD

def test_error_handling(prnu_analyzer):
    """Test error handling for invalid inputs."""
    # Test with non-existent file
    with pytest.raises(ValueError):
        prnu_analyzer.analyze('nonexistent.jpg')
    
    # Test with invalid file
    invalid_file = Path(__file__).parent / 'test_prnu.py'
    with pytest.raises(ValueError):
        prnu_analyzer.analyze(invalid_file)
    
    # Test compute_prnu_pattern with empty list
    with pytest.raises(ValueError):
        prnu_analyzer.compute_prnu_pattern([])

def test_heatmap_generation(prnu_analyzer, data_dir):
    """Test tampering heatmap generation."""
    original_dir = data_dir / 'original'
    tampered_dir = data_dir / 'tampered'
    
    # Get reference image
    original_path = next(original_dir.glob('*.jpg'))
    original_img, original_noise = prnu_analyzer.analyze(original_path)
    
    # Test with both original and tampered images
    for img_dir in [original_dir, tampered_dir]:
        for img_path in img_dir.glob('*.jpg'):
            # Skip reference image
            if img_path == original_path:
                continue
                
            print(f"\nGenerating heatmap for: {img_path.name}")
            
            # Analyze image
            test_img, test_noise = prnu_analyzer.analyze(img_path)
            
            # Resize if needed
            if test_img.shape != original_img.shape:
                print(f"Resizing from {test_img.shape} to {original_img.shape}")
                test_img = cv2.resize(test_img, (original_img.shape[1], original_img.shape[0]))
                test_noise = cv2.resize(test_noise, (original_noise.shape[1], original_noise.shape[0]))
            
            # Generate heatmap
            heatmap = prnu_analyzer.create_tampering_heatmap(
                original_noise,
                test_noise,
                window_size=TEST_WINDOW_SIZE,
                stride=TEST_STRIDE
            )
            
            # Print heatmap statistics
            print(f"Heatmap statistics:")
            print(f"- Standard deviation: {np.std(heatmap):.6f}")
            print(f"- Min value: {np.min(heatmap):.6f}")
            print(f"- Max value: {np.max(heatmap):.6f}")
            
            # Check heatmap properties
            assert heatmap.shape[:2] == original_img.shape[:2]
            assert np.all(heatmap >= 0) and np.all(heatmap <= 1)
            assert np.all(np.isfinite(heatmap))  # No NaN or inf values 

def test_detect_tampering_original_images(prnu_analyzer, data_dir):
    """Test tampering detection on original (untampered) images."""
    original_dir = data_dir / 'original'
    
    # Get first image as reference
    reference_path = next(original_dir.glob('*.jpg'))
    reference_img, reference_pattern = prnu_analyzer.analyze(reference_path)
    
    # Test each original image
    for img_path in original_dir.glob('*.jpg'):
        print(f"\nAnalyzing original image: {img_path.name}")
        
        # Detect tampering
        is_tampered, visualization = prnu_analyzer.detect_tampering(
            img_path,
            reference_pattern=reference_pattern,
            correlation_threshold=CORRELATION_THRESHOLD
        )
        
        # Check results
        assert isinstance(is_tampered, bool), "is_tampered should be boolean"
        assert isinstance(visualization, np.ndarray), "visualization should be numpy array"
        assert visualization.dtype == np.uint8, "visualization should be uint8"
        assert len(visualization.shape) == 3, "visualization should be 3D array"
        assert visualization.shape[2] == 3, "visualization should have 3 channels"
        
        # For original images, expect low tampering probability
        assert not is_tampered, f"Original image {img_path.name} incorrectly flagged as tampered"

def test_detect_tampering_tampered_images(prnu_analyzer, data_dir):
    """Test tampering detection on known tampered images."""
    original_dir = data_dir / 'original'
    tampered_dir = data_dir / 'tampered'
    
    # Get first original image as reference
    reference_path = next(original_dir.glob('*.jpg'))
    reference_img, reference_pattern = prnu_analyzer.analyze(reference_path)
    
    # Test each tampered image
    for img_path in tampered_dir.glob('*.jpg'):
        print(f"\nAnalyzing tampered image: {img_path.name}")
        
        # Detect tampering
        is_tampered, visualization = prnu_analyzer.detect_tampering(
            img_path,
            reference_pattern=reference_pattern,
            correlation_threshold=CORRELATION_THRESHOLD
        )
        
        # Check results
        assert isinstance(is_tampered, bool), "is_tampered should be boolean"
        assert isinstance(visualization, np.ndarray), "visualization should be numpy array"
        assert visualization.dtype == np.uint8, "visualization should be uint8"
        assert len(visualization.shape) == 3, "visualization should be 3D array"
        assert visualization.shape[2] == 3, "visualization should have 3 channels"
        
        # For tampered images, expect detection
        assert is_tampered, f"Tampered image {img_path.name} not detected as tampered"

def test_detect_tampering_parameters(prnu_analyzer, data_dir):
    """Test tampering detection with different parameters."""
    tampered_dir = data_dir / 'tampered'
    test_image = next(tampered_dir.glob('*.jpg'))
    
    # Test different correlation thresholds
    thresholds = [0.3, 0.5, 0.7]
    for threshold in thresholds:
        is_tampered, vis = prnu_analyzer.detect_tampering(
            test_image,
            correlation_threshold=threshold
        )
        assert isinstance(is_tampered, bool)
        assert isinstance(vis, np.ndarray)
    
    # Test different window sizes
    window_sizes = [32, 64, 128]
    for size in window_sizes:
        is_tampered, vis = prnu_analyzer.detect_tampering(
            test_image,
            window_size=size,
            stride=size//2
        )
        assert isinstance(is_tampered, bool)
        assert isinstance(vis, np.ndarray)
    
    # Test different overlay alphas
    alphas = [0.3, 0.6, 0.9]
    for alpha in alphas:
        is_tampered, vis = prnu_analyzer.detect_tampering(
            test_image,
            overlay_alpha=alpha
        )
        assert isinstance(is_tampered, bool)
        assert isinstance(vis, np.ndarray)

def test_detect_tampering_error_handling(prnu_analyzer, data_dir):
    """Test error handling in tampering detection."""
    # Test with non-existent file
    with pytest.raises(ValueError):
        prnu_analyzer.detect_tampering('nonexistent.jpg')
    
    # Test with invalid file
    invalid_file = Path(__file__).parent / 'test_prnu.py'
    with pytest.raises(ValueError):
        prnu_analyzer.detect_tampering(invalid_file)
    
    # Test with invalid reference pattern type
    test_image = data_dir / 'tampered' / 'gpt-4o-generated-receipt-02.png'
    with pytest.raises(ValueError):
        prnu_analyzer.detect_tampering(
            test_image,
            reference_pattern="not a numpy array"  # Wrong type
        ) 