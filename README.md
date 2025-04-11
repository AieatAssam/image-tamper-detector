# Image Tamper Detector

A sophisticated image tampering detection system that uses Error Level Analysis (ELA) and computer vision techniques to identify potentially manipulated regions in digital images. The system analyzes compression artifacts and error levels to detect areas that may have been digitally altered.

## Features

- **Error Level Analysis (ELA)**: Advanced implementation of ELA for detecting image manipulation
- **Visual Heatmaps**: Generate visual heatmaps highlighting potentially tampered regions
- **Configurable Parameters**: Adjustable sensitivity and threshold settings
- **Support for Multiple Formats**: Handles various image formats including JPEG
- **Fast Processing**: Efficient analysis of high-resolution images
- **REST API**: FastAPI-based REST API for easy integration

## Project Structure

```
image-tamper-detector/
├── backend/           # Backend API and core logic
│   ├── app/          # Main application code
│   │   ├── analysis/ # Image analysis algorithms
│   │   ├── api/      # API endpoints
│   │   └── utils/    # Utility functions
│   └── tests/        # Unit tests
├── data/             # Test and sample images
│   ├── samples/      # Sample images for testing
│   │   ├── original/ # Original untampered images
│   │   └── tampered/ # Known tampered images
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

### API Endpoints

The system provides the following REST API endpoints:

- `POST /api/analyze`: Analyze an image for potential tampering
  - Input: Image file (multipart/form-data)
  - Output: Analysis results including tampering probability and heatmap

### Configuration

Key parameters that can be adjusted:

- `QUALITY`: JPEG compression quality (default: 90)
- `SCALE_FACTOR`: Sensitivity of the analysis (default: 10)
- `THRESHOLD`: Detection threshold for tampering (default: 40)

## Technical Details

### Error Level Analysis (ELA)

The system uses Error Level Analysis to detect image tampering by:
1. Recompressing the image at a known quality level
2. Computing the difference between the original and recompressed versions
3. Analyzing error levels to identify inconsistencies that may indicate tampering

### Thresholds

- Untampered threshold: 15.0% (maximum acceptable difference for original images)
- Tampered threshold: 5.0% (minimum difference to flag potential tampering)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[MIT License](LICENSE)

## Acknowledgments

- Based on research in Error Level Analysis and digital image forensics
- Uses OpenCV and PIL for image processing
- FastAPI for REST API implementation 