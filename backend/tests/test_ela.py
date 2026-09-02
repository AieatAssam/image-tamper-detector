"""
Unit tests for Error Level Analysis (ELA) implementation.
"""
import pytest
import numpy as np
from io import BytesIO
from pathlib import Path
from PIL import Image
import cv2
from backend.app.analysis.ela import ELAAnalyzer, MAX_ANALYSIS_SIDE, TamperingFeatures

@pytest.fixture
def ela_analyzer():
    """Fixture to create ELA analyzer instance."""
    return ELAAnalyzer(quality=95, resave_quality=75)

@pytest.fixture
def data_dir():
    """Fixture to get the data directory path."""
    return Path(__file__).parent.parent.parent / 'data' / 'samples'

def test_ela_analyzer_initialization():
    """Test ELA analyzer initialization with valid and invalid parameters."""
    # Test valid initialization
    analyzer = ELAAnalyzer()
    assert analyzer.quality == 95
    assert analyzer.resave_quality == 95
    assert analyzer.max_image_size == MAX_ANALYSIS_SIDE
    
    # Test invalid quality values
    with pytest.raises(ValueError):
        ELAAnalyzer(quality=101)
    with pytest.raises(ValueError):
        ELAAnalyzer(resave_quality=101)

def test_analyze_returns_correct_shapes(ela_analyzer, data_dir):
    """Test that analyze method returns arrays of correct shape."""
    for img_dir in ['original', 'tampered']:
        dir_path = data_dir / img_dir
        for img_path in dir_path.glob('*.[jp][pn][g]*'):
            original, ela_result = ela_analyzer.analyze(img_path)
            
            assert len(original.shape) == 3  # RGB image
            assert len(ela_result.shape) == 3  # RGB ELA result
            assert original.shape == ela_result.shape

def test_error_handling(ela_analyzer):
    """Test error handling for invalid inputs."""
    # Test with non-existent file
    with pytest.raises(ValueError):
        ela_analyzer.analyze('nonexistent.jpg')
    
    # Test with invalid file
    invalid_file = Path(__file__).parent / 'test_ela.py'
    with pytest.raises(ValueError):
        ela_analyzer.analyze(invalid_file)

def test_detect_tampering_original_images(ela_analyzer, data_dir):
    """Test tampering detection on original images."""
    # Test with original landscape image
    original_image = data_dir / "original" / "landscape_original.jpg"
    is_tampered, _, features = ela_analyzer.detect_tampering(original_image)
    
    # Print feature values for debugging
    print("\nOriginal image features:")
    print(f"Edge discontinuity: {features.edge_discontinuity:.3f}")
    print(f"Compression artifacts: {features.compression_artifacts:.3f}")
    print(f"Texture variance: {features.texture_variance:.3f}")
    print(f"Noise consistency: {features.noise_consistency:.3f}")
    
    assert isinstance(is_tampered, bool)
    assert 0 <= features.edge_discontinuity <= 1

def test_detect_tampering_tampered_images(ela_analyzer, data_dir):
    """Test tampering detection on known tampered images."""
    # Test with AI-generated receipt
    tampered_image = data_dir / "tampered" / "gpt-4o-generated-receipt-02.png"
    is_tampered, _, features = ela_analyzer.detect_tampering(tampered_image)
    
    # Print feature values for debugging
    print("\nTampered image features:")
    print(f"Edge discontinuity: {features.edge_discontinuity:.3f}")
    print(f"Compression artifacts: {features.compression_artifacts:.3f}")
    print(f"Texture variance: {features.texture_variance:.3f}")
    print(f"Noise consistency: {features.noise_consistency:.3f}")
    
    assert isinstance(is_tampered, bool)
    assert 0 <= features.edge_discontinuity <= 1

def test_feature_computation(ela_analyzer, data_dir):
    """Test individual feature computation methods."""
    # Get a sample image
    image_path = next((data_dir / 'tampered').glob('*.[jp][pn][g]*'))
    _, ela_result = ela_analyzer.analyze(image_path)
    
    # Test edge discontinuity
    edge_score = ela_analyzer._compute_edge_discontinuity(ela_result)
    assert isinstance(edge_score, float)
    assert 0 <= edge_score <= 1.0
    
    # Test texture variance
    texture_score = ela_analyzer._compute_texture_variance(ela_result)
    assert isinstance(texture_score, float)
    assert texture_score >= 0
    
    # Test noise consistency
    noise_score = ela_analyzer._compute_noise_consistency(ela_result)
    assert isinstance(noise_score, float)
    assert noise_score >= 0
    
    # Test compression artifacts
    compression_score = ela_analyzer._compute_compression_artifacts(ela_result)
    assert isinstance(compression_score, float)
    assert compression_score >= 0

def test_image_preprocessing(ela_analyzer):
    """Test image preprocessing functionality."""
    # Test RGB conversion
    gray_image = Image.new('L', (100, 100), color=128)
    processed = ela_analyzer._preprocess_image(gray_image)
    assert processed.mode == 'RGB'
    
    # The paper-faithful default preserves the JPEG pixel lattice.
    large_image = Image.new('RGB', (3000, 3000), color='white')
    processed = ela_analyzer._preprocess_image(large_image)
    assert processed.size == large_image.size

    bounded = ELAAnalyzer(max_image_size=1024)
    assert max(bounded._preprocess_image(large_image).size) <= 1024


def test_analyze_compares_input_with_one_controlled_resave():
    rng = np.random.default_rng(22)
    source = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    input_buffer = BytesIO()
    Image.fromarray(source).save(input_buffer, format='JPEG', quality=75)

    original, actual = ELAAnalyzer(quality=95).analyze(input_buffer.getvalue())

    resaved_buffer = BytesIO()
    Image.fromarray(original).save(resaved_buffer, format='JPEG', quality=95)
    resaved = np.array(Image.open(BytesIO(resaved_buffer.getvalue())).convert('RGB'))
    expected = cv2.absdiff(original, resaved)
    assert np.array_equal(actual, expected)


def test_compression_artifacts_matches_8px_block_boundaries(ela_analyzer):
    gray = np.arange(32 * 32, dtype=np.uint8).reshape(32, 32)
    expected = []
    for row in range(3):
        for column in range(3):
            block = gray[row * 8:(row + 1) * 8, column * 8:(column + 1) * 8]
            next_horizontal = gray[row * 8:(row + 1) * 8, (column + 1) * 8:(column + 2) * 8]
            next_vertical = gray[(row + 1) * 8:(row + 2) * 8, column * 8:(column + 1) * 8]
            expected.extend([
                np.mean(np.abs(block[:, -1] - next_horizontal[:, 0])),
                np.mean(np.abs(block[-1, :] - next_vertical[0, :])),
            ])
    assert np.isclose(ela_analyzer._compute_compression_artifacts(gray), np.mean(expected))

def test_empty_suspicious_mask_does_not_fail():
    image = Image.new('RGB', (256, 256), color='gray')
    output = BytesIO()
    image.save(output, format='JPEG')

    is_tampered, visualization, _ = ELAAnalyzer().detect_tampering(
        output.getvalue(), edge_threshold=-1.0
    )

    assert is_tampered
    assert visualization.shape == (256, 256, 3)
