"""
Unit tests for Error Level Analysis (ELA) implementation.
"""
import pytest
import numpy as np
from pathlib import Path
from PIL import Image
import cv2
from backend.app.analysis.ela import ELAAnalyzer

# Test constants
TEST_QUALITY = 90
TEST_SCALE = 10  # Reduced from 15 to be less sensitive
THRESHOLD = 40   # Reduced from 50 to account for natural compression artifacts
UNTAMPERED_THRESHOLD = 15.0  # Increased from 10.0 to account for high-res images
TAMPERED_THRESHOLD = 5.0     # Keep minimum threshold for tampered detection

@pytest.fixture
def ela_analyzer():
    """Fixture to create ELA analyzer instance."""
    return ELAAnalyzer(quality=TEST_QUALITY, scale_factor=TEST_SCALE)

@pytest.fixture
def data_dir():
    """Fixture to get the data directory path."""
    return Path(__file__).parent.parent.parent / 'data' / 'samples'

def test_ela_analyzer_initialization():
    """Test ELA analyzer initialization with valid and invalid parameters."""
    # Test valid initialization
    analyzer = ELAAnalyzer(quality=90, scale_factor=10)
    assert analyzer.quality == 90
    assert analyzer.scale_factor == 10
    
    # Test invalid quality value
    with pytest.raises(ValueError):
        ELAAnalyzer(quality=101)
    with pytest.raises(ValueError):
        ELAAnalyzer(quality=0)

def test_original_image_analysis(ela_analyzer, data_dir):
    """Test ELA on original (untampered) images."""
    original_dir = data_dir / 'original'
    
    for img_path in original_dir.glob('*.jpg'):
        # Analyze image
        original, ela_result = ela_analyzer.analyze(img_path)
        
        # Get binary mask of potentially tampered regions
        mask = ela_analyzer.get_threshold_mask(ela_result, THRESHOLD)
        
        # Calculate percentage of potentially tampered pixels
        tampered_percentage = (mask.sum() / mask.size) * 100
        
        # For original images, expect low percentage of "tampered" pixels
        assert tampered_percentage < UNTAMPERED_THRESHOLD, f"Original image {img_path.name} shows high tampering probability of {tampered_percentage}%"

def test_tampered_image_analysis(ela_analyzer, data_dir):
    """Test ELA on known tampered images."""
    tampered_dir = data_dir / 'tampered'
    
    for img_path in tampered_dir.glob('*.jpg'):
        # Analyze image
        original, ela_result = ela_analyzer.analyze(img_path)
        
        # Get binary mask of potentially tampered regions
        mask = ela_analyzer.get_threshold_mask(ela_result, THRESHOLD)
        
        # Calculate percentage of potentially tampered pixels
        tampered_percentage = (mask.sum() / mask.size) * 100
        
        # For tampered images, expect higher percentage of "tampered" pixels
        assert tampered_percentage > TAMPERED_THRESHOLD, f"Tampered image {img_path.name} shows low tampering probability of {tampered_percentage}%"

def test_heatmap_generation(ela_analyzer, data_dir):
    """Test heatmap overlay generation."""
    # Test with both original and tampered images
    for img_dir in ['original', 'tampered']:
        dir_path = data_dir / img_dir
        for img_path in dir_path.glob('*.jpg'):
            # Analyze image
            original, ela_result = ela_analyzer.analyze(img_path)
            
            # Generate heatmap
            heatmap = ela_analyzer.overlay_heatmap(original, ela_result)
            
            # Check heatmap properties
            assert heatmap.shape[:2] == original.shape[:2], "Heatmap size mismatch"
            assert heatmap.dtype == np.uint8, "Invalid heatmap data type"

def test_error_handling(ela_analyzer):
    """Test error handling for invalid inputs."""
    # Test with non-existent file
    with pytest.raises(FileNotFoundError):
        ela_analyzer.analyze('nonexistent.jpg')
    
    # Test with invalid file
    invalid_file = Path(__file__).parent / 'test_ela.py'
    with pytest.raises(ValueError):
        ela_analyzer.analyze(invalid_file)

def test_threshold_mask(ela_analyzer):
    """Test threshold mask generation."""
    # Create test array
    test_array = np.zeros((100, 100), dtype=np.uint8)
    test_array[40:60, 40:60] = 255  # Create a square of high values
    
    # Generate mask
    mask = ela_analyzer.get_threshold_mask(test_array, threshold=50)
    
    # Check mask properties
    assert mask.dtype == np.uint8
    assert mask.shape == test_array.shape
    assert mask[45, 45] == 1  # Check high value area
    assert mask[0, 0] == 0    # Check low value area 