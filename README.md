# Image Tamper Detector

A sophisticated image tampering detection system that combines multiple analysis techniques to identify potentially manipulated regions in digital images. The system uses Error Level Analysis (ELA), Photo Response Non-Uniformity (PRNU), Entropy Analysis, and computer vision techniques to detect areas that may have been digitally altered or generated.

## Features

- **Error Level Analysis (ELA)**: Advanced implementation of ELA for detecting image manipulation through compression artifacts
- **Photo Response Non-Uniformity (PRNU)**: Camera sensor pattern noise analysis for detecting:
  - Image splicing and copy-paste manipulation
  - AI-generated content
  - Images from different source cameras
- **Entropy-based AI Detection**: Advanced entropy analysis for identifying AI-generated content:
  - Cross-channel entropy pattern analysis
  - Local uniformity detection
  - Natural vs. artificial pattern discrimination
- **Visual Heatmaps**: Generate visual heatmaps highlighting potentially tampered regions
- **Multi-technique Analysis**: Combines results from multiple detection methods
- **Configurable Parameters**: Adjustable sensitivity and threshold settings
- **Support for Multiple Formats**: Handles various image formats including JPEG and PNG
- **Fast Processing**: Efficient analysis of high-resolution images
- **REST API**: FastAPI-based REST API for easy integration

## Project Structure

```
image-tamper-detector/
├── backend/           # Backend API and core logic
│   ├── app/          # Main application code
│   │   ├── analysis/ # Image analysis algorithms
│   │   │   ├── ela.py     # Error Level Analysis
│   │   │   ├── entropy.py # Entropy-based AI Detection
│   │   │   └── prnu.py    # Photo Response Non-Uniformity
│   │   ├── api/      # API endpoints
│   │   └── utils/    # Utility functions
│   └── tests/        # Unit tests
├── data/             # Test and sample images
│   ├── samples/      # Sample images for testing
│   │   ├── original/ # Original untampered images
│   │   └── tampered/ # Known tampered/AI-generated images
│   └── processed/    # Processed results
└── requirements.txt  # Project dependencies
```

## Installation

### Prerequisites

- Python 3.12 (Python 3.13 is not yet supported by TensorFlow as of April 2025)
- pip (latest version recommended)

### Steps

1. Clone the repository:
```bash
git clone https://github.com/yourusername/image-tamper-detector.git
cd image-tamper-detector
```

2. Ensure you have Python 3.12:
```bash
python --version  # Should output Python 3.12.x
```
If needed, install Python 3.12 using pyenv or your preferred version manager.

3. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

4. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running Tests

To run the test suite:
```bash
python -m pytest backend/tests/
```

For specific test modules:
```bash
python -m pytest backend/tests/test_ela.py    # Test ELA module
python -m pytest backend/tests/test_prnu.py   # Test PRNU module
python -m pytest backend/tests/test_entropy.py # Test Entropy module
```

### API Endpoints

The system provides the following REST API endpoints:

- `POST /api/analyze`: Analyze an image for potential tampering
  - Input: Image file (multipart/form-data)
  - Output: Analysis results including:
    - Tampering probability
    - ELA heatmap
    - PRNU correlation map
    - Entropy analysis visualization
    - Combined analysis results

### Configuration

Key parameters that can be adjusted:

#### ELA Parameters
- `QUALITY`: JPEG compression quality (default: 90)
- `SCALE_FACTOR`: Sensitivity of the analysis (default: 10)
- `THRESHOLD`: Detection threshold for tampering (default: 40)

#### PRNU Parameters
- `WINDOW_SIZE`: Size of analysis window (default: 64)
- `STRIDE`: Window stride for analysis (default: 32)
- `SIGMA`: Gaussian filter sigma (default: 5.0)

#### Entropy Analysis Parameters
- `RADIUS`: Local entropy calculation window size (default: 4)
- `TOLERANCE`: Entropy matching tolerance (default: 0.12)
- `MATCHING_THRESHOLD`: AI detection threshold (default: 0.35)
- `UNIFORMITY_THRESHOLD`: Local pattern uniformity threshold (default: 0.2)

## Technical Details

### Error Level Analysis (ELA)

The system uses Error Level Analysis to detect image tampering by:
1. Recompressing the image at a known quality level
2. Computing the difference between the original and recompressed versions
3. Analyzing error levels to identify inconsistencies that may indicate tampering

### Photo Response Non-Uniformity (PRNU)

PRNU analysis detects image tampering by:
1. Extracting camera sensor noise patterns from images
2. Computing normalized cross-correlation between patterns
3. Generating correlation heatmaps to identify:
   - Regions with inconsistent noise patterns (potential tampering)
   - AI-generated content (lacking proper sensor patterns)
   - Images from different source cameras

### Entropy-based AI Detection

The system uses entropy analysis to detect AI-generated content by:
1. Computing local entropy patterns across color channels
2. Analyzing cross-channel entropy consistency
3. Identifying artificial patterns characteristic of AI generation:
   - Natural images: Higher matching proportions (>35%)
   - AI-generated images: Lower matching proportions (<35%)
4. Generating visualizations of suspicious regions

### Thresholds

- ELA thresholds:
  - Untampered: 15.0% (maximum acceptable difference for original images)
  - Tampered: 5.0% (minimum difference to flag potential tampering)
- PRNU thresholds:
  - Same camera: > 0.5 correlation
  - Different cameras: < 0.5 correlation
  - Tampered regions: Local correlation variations
- Entropy thresholds:
  - Natural images: > 35% matching entropy patterns
  - AI-generated: < 35% matching entropy patterns
  - Local uniformity: > 20% for pattern detection

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[MIT License](LICENSE)

## Acknowledgments

- Based on research in digital image forensics
- Uses OpenCV and NumPy for image processing
- FastAPI for REST API implementation
- SciPy for signal processing 