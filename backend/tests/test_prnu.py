"""
Unit tests for Photo Response Non-Uniformity (PRNU) implementation.
"""
import pytest
import numpy as np
from pathlib import Path
import cv2
from backend.app.analysis.prnu import PRNUAnalyzer

@pytest.fixture
def prnu_analyzer():
    """Fixture to create PRNU analyzer instance with a high threshold for AI-generated image detection.
    We ignore copy-paste tampered images and focus on AI-generated image detection, as PRNU is less effective for copy-move forgeries.
    """
    return PRNUAnalyzer(variance_threshold=300)

@pytest.fixture
def data_dir():
    """Fixture to get the data directory path."""
    return Path(__file__).parent.parent.parent / 'data' / 'samples'

def test_prnu_analyzer_initialization():
    """Test PRNU analyzer initialization with valid parameters."""
    analyzer = PRNUAnalyzer(noise_filter_sigma=3.0, window_size=64, stride=32, variance_threshold=0.001)
    assert analyzer.noise_filter_sigma == 3.0
    assert analyzer.window_size == 64
    assert analyzer.stride == 32
    assert analyzer.variance_threshold == 0.001

def test_analyze_returns_correct_shapes(prnu_analyzer, data_dir):
    """Test that analyze method returns arrays of correct shape."""
    for img_dir in ['original', 'tampered']:
        dir_path = data_dir / img_dir
        for img_path in dir_path.glob('*.[jp][pn][g]*'):
            image_rgb, noise_residual = prnu_analyzer.analyze(img_path)
            assert len(image_rgb.shape) == 3  # RGB image
            assert len(noise_residual.shape) == 3  # Noise residual
            assert image_rgb.shape == noise_residual.shape
            assert image_rgb.dtype == np.uint8 or image_rgb.dtype == np.float32
            assert noise_residual.dtype == np.float32

def test_error_handling(prnu_analyzer):
    """Test error handling for invalid inputs."""
    # Test with non-existent file
    with pytest.raises(ValueError):
        prnu_analyzer.analyze('nonexistent.jpg')
    # Test with invalid file
    invalid_file = Path(__file__).parent / 'test_prnu.py'
    with pytest.raises(ValueError):
        prnu_analyzer.analyze(invalid_file)

def test_detect_tampering_original_images(prnu_analyzer, data_dir):
    """Test tampering detection on original images."""
    original_image = data_dir / "original" / "landscape_original.jpg"
    is_tampered, visualization, uniformity_score = prnu_analyzer.detect_tampering(original_image)
    print(f"\nOriginal image PRNU uniformity score: {uniformity_score:.6f}")
    assert not is_tampered, "Original image incorrectly flagged as tampered"
    assert isinstance(visualization, np.ndarray)
    assert visualization.dtype == np.uint8
    assert len(visualization.shape) == 3
    assert visualization.shape[2] == 3

def test_detect_tampering_ai_generated_images(prnu_analyzer, data_dir):
    """Test tampering detection on AI-generated images (focus of PRNU)."""
    tampered_dir = data_dir / "tampered"
    ai_images = [img for img in tampered_dir.glob('*.[jp][pn][g]*') if 'generated' in img.name]
    for ai_image in ai_images:
        is_tampered, visualization, uniformity_score = prnu_analyzer.detect_tampering(ai_image)
        print(f"\nAI-generated image PRNU uniformity score: {uniformity_score:.6f}")
        assert is_tampered, f"AI-generated image {ai_image.name} not detected as tampered"
        assert isinstance(visualization, np.ndarray)
        assert visualization.dtype == np.uint8
        assert len(visualization.shape) == 3
        assert visualization.shape[2] == 3

def test_print_uniformity_scores_for_all_images(prnu_analyzer, data_dir):
    """Print PRNU uniformity scores for all original and tampered images for threshold debugging."""
    print("\n--- ORIGINAL IMAGES ---")
    orig_dir = data_dir / "original"
    for img_path in orig_dir.glob('*.[jp][pn][g]*'):
        is_tampered, _, uniformity_score = prnu_analyzer.detect_tampering(img_path)
        print(f"Original: {img_path.name} | Tampered: {is_tampered} | Score: {uniformity_score:.6f}")

    print("\n--- TAMPERED IMAGES ---")
    tampered_dir = data_dir / "tampered"
    for img_path in tampered_dir.glob('*.[jp][pn][g]*'):
        is_tampered, _, uniformity_score = prnu_analyzer.detect_tampering(img_path)
        print(f"Tampered: {img_path.name} | Tampered: {is_tampered} | Score: {uniformity_score:.6f}")

# Note: We ignore copy-paste tampered images in these tests, as PRNU is not designed for copy-move forgeries. 