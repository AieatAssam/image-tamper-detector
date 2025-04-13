"""
Unit tests for Entropy-based AI image detection implementation.
"""
import pytest
import numpy as np
from pathlib import Path
from PIL import Image
import cv2
from backend.app.analysis.entropy import EntropyAnalyzer, EntropyFeatures

# Test constants
TEST_RADIUS = 4  # Match default radius
TEST_TOLERANCE = 0.12  # Match default tolerance
TEST_MATCHING_THRESHOLD = 0.35  # Match default threshold
TEST_UNIFORMITY_THRESHOLD = 0.2  # Keep same uniformity threshold

@pytest.fixture
def entropy_analyzer():
    """Fixture to create entropy analyzer instance."""
    return EntropyAnalyzer(
        radius=TEST_RADIUS,
        tolerance=TEST_TOLERANCE,
        matching_threshold=TEST_MATCHING_THRESHOLD,
        uniformity_threshold=TEST_UNIFORMITY_THRESHOLD
    )

@pytest.fixture
def data_dir():
    """Fixture to get the data directory path."""
    return Path(__file__).parent.parent.parent / 'data' / 'samples'

def test_entropy_analyzer_initialization():
    """Test entropy analyzer initialization with valid parameters."""
    # Test valid initialization
    analyzer = EntropyAnalyzer(
        radius=TEST_RADIUS,
        tolerance=TEST_TOLERANCE,
        matching_threshold=TEST_MATCHING_THRESHOLD
    )
    assert analyzer.radius == TEST_RADIUS
    assert analyzer.tolerance == TEST_TOLERANCE
    assert analyzer.matching_threshold == TEST_MATCHING_THRESHOLD
    
    # Test with different valid parameters
    analyzer = EntropyAnalyzer(radius=3, tolerance=0.2, matching_threshold=0.4)
    assert analyzer.radius == 3
    assert analyzer.tolerance == 0.2
    assert analyzer.matching_threshold == 0.4
    
    # Test invalid parameters
    with pytest.raises(ValueError):
        EntropyAnalyzer(radius=0)  # Radius must be positive
    with pytest.raises(ValueError):
        EntropyAnalyzer(tolerance=0)  # Tolerance must be positive
    with pytest.raises(ValueError):
        EntropyAnalyzer(tolerance=1.5)  # Tolerance must be less than 1
    with pytest.raises(ValueError):
        EntropyAnalyzer(matching_threshold=0)  # Threshold must be positive
    with pytest.raises(ValueError):
        EntropyAnalyzer(matching_threshold=1.5)  # Threshold must be less than 1

def test_analyze_returns_correct_shapes(entropy_analyzer, data_dir):
    """Test that analyze method returns arrays of correct shape."""
    for img_dir in ['original', 'tampered']:
        dir_path = data_dir / img_dir
        for img_path in dir_path.glob('*.[jp][pn][g]*'):
            image_rgb, features = entropy_analyzer.analyze(img_path)
            
            # Check image shape
            assert len(image_rgb.shape) == 3  # RGB image
            assert image_rgb.dtype == np.uint8
            
            # Check feature shapes
            assert len(features.entropy_red.shape) == 2  # 2D entropy maps
            assert len(features.entropy_green.shape) == 2
            assert len(features.entropy_blue.shape) == 2
            assert len(features.matching_mask.shape) == 2
            assert len(features.uniformity_mask.shape) == 2  # New uniformity mask
            
            # Check all maps have same shape as image
            assert image_rgb.shape[:2] == features.entropy_red.shape
            assert image_rgb.shape[:2] == features.entropy_green.shape
            assert image_rgb.shape[:2] == features.entropy_blue.shape
            assert image_rgb.shape[:2] == features.matching_mask.shape
            assert image_rgb.shape[:2] == features.uniformity_mask.shape  # New uniformity mask

def test_error_handling(entropy_analyzer):
    """Test error handling for invalid inputs."""
    # Test with non-existent file
    with pytest.raises(FileNotFoundError):
        entropy_analyzer.analyze('nonexistent.jpg')
    
    # Test with invalid file
    invalid_file = Path(__file__).parent / 'test_entropy.py'
    with pytest.raises(ValueError):
        entropy_analyzer.analyze(invalid_file)
    
    # Test with non-existent file in detect_ai_generated
    with pytest.raises(FileNotFoundError):
        entropy_analyzer.detect_ai_generated('nonexistent.jpg')
    
    # Test with invalid file in detect_ai_generated
    with pytest.raises(ValueError):
        entropy_analyzer.detect_ai_generated(invalid_file)

def test_detect_ai_generated_original_images(entropy_analyzer, data_dir):
    """Test AI detection on original (non-AI) images."""
    # Test with original landscape image
    original_image = data_dir / "original" / "landscape_original.jpg"
    is_ai_generated, visualization, proportion = entropy_analyzer.detect_ai_generated(original_image)
    
    # Print analysis results for debugging
    print("\nOriginal image analysis:")
    print(f"Matching proportion: {proportion:.3f}")
    print(f"Threshold: {entropy_analyzer.matching_threshold:.3f}")
    print(f"Uniformity threshold: {entropy_analyzer.uniformity_threshold:.3f}")
    
    # For original images, expect high proportion of matching pixels
    assert proportion > entropy_analyzer.matching_threshold
    assert not is_ai_generated, "Original image incorrectly flagged as AI-generated"
    
    # Check visualization properties
    assert isinstance(visualization, np.ndarray)
    assert visualization.dtype == np.uint8
    assert len(visualization.shape) == 3
    assert visualization.shape[2] == 3

def test_detect_ai_generated_ai_images(entropy_analyzer, data_dir):
    """Test AI detection on known AI-generated images."""
    # Test with AI-generated receipt
    ai_image = data_dir / "tampered" / "gpt-4o-generated-receipt-02.png"
    is_ai_generated, visualization, proportion = entropy_analyzer.detect_ai_generated(ai_image)
    
    # Print analysis results for debugging
    print("\nAI-generated image analysis:")
    print(f"Matching proportion: {proportion:.3f}")
    print(f"Threshold: {entropy_analyzer.matching_threshold:.3f}")
    print(f"Uniformity threshold: {entropy_analyzer.uniformity_threshold:.3f}")
    
    # For AI-generated images, expect low proportion of matching pixels
    assert proportion < entropy_analyzer.matching_threshold
    assert is_ai_generated, "AI-generated image not detected"
    
    # Check visualization properties
    assert isinstance(visualization, np.ndarray)
    assert visualization.dtype == np.uint8
    assert len(visualization.shape) == 3
    assert visualization.shape[2] == 3

def test_entropy_computation(entropy_analyzer, data_dir):
    """Test entropy computation for individual channels."""
    # Get a sample image
    image_path = next((data_dir / 'original').glob('*.[jp][pn][g]*'))
    _, features = entropy_analyzer.analyze(image_path)
    
    # Test entropy maps
    for entropy_map in [features.entropy_red, features.entropy_green, features.entropy_blue]:
        assert np.all(np.isfinite(entropy_map))  # No NaN or inf values
        assert np.all(entropy_map >= 0)  # Non-negative entropy
        assert entropy_map.dtype == np.uint8  # Correct data type
    
    # Test matching mask
    assert features.matching_mask.dtype == bool  # Binary mask
    assert np.all((features.matching_mask == 0) | (features.matching_mask == 1))  # Only 0s and 1s
    
    # Test uniformity mask
    assert features.uniformity_mask.dtype == bool  # Binary mask
    assert np.all((features.uniformity_mask == 0) | (features.uniformity_mask == 1))  # Only 0s and 1s

def test_image_preprocessing(entropy_analyzer, data_dir):
    """Test image preprocessing with different input types."""
    # Test with grayscale image
    gray_image = Image.new('L', (100, 100), color=128)
    gray_path = data_dir / 'test_gray.png'
    gray_image.save(gray_path)
    
    try:
        image_rgb, features = entropy_analyzer.analyze(gray_path)
        assert len(image_rgb.shape) == 3  # Should be converted to RGB
        assert image_rgb.shape[2] == 3
    finally:
        # Clean up test file
        gray_path.unlink()
    
    # Test with RGBA image
    rgba_image = Image.new('RGBA', (100, 100), color=(128, 128, 128, 255))
    rgba_path = data_dir / 'test_rgba.png'
    rgba_image.save(rgba_path)
    
    try:
        image_rgb, features = entropy_analyzer.analyze(rgba_path)
        assert len(image_rgb.shape) == 3  # Should be converted to RGB
        assert image_rgb.shape[2] == 3
    finally:
        # Clean up test file
        rgba_path.unlink() 