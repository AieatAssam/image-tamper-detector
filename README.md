# Image Tamper Detector

**NOTICE**
This project is a functional image analysis system that combines multiple detection techniques. While the core functionality is operational, some detection parameters are still being optimized for improved accuracy. The system is suitable for experimental use and research purposes.

## Overview
A sophisticated image tampering detection system that combines multiple analysis techniques to identify potentially manipulated regions in digital images. The project implements several proven analysis methods and provides a practical platform for image authenticity verification.

The system successfully employs Error Level Analysis (ELA), Photo Response Non-Uniformity (PRNU), Entropy Analysis, and computer vision techniques to detect areas that may have been digitally altered or generated. These techniques are operational and provide useful insights, with ongoing optimization to improve detection accuracy.

## Current Development Status

- ✅ Core analysis algorithms implemented and operational
- ✅ API endpoints fully functional
- ✅ Test framework established and passing
- 🚧 Frontend visualization in progress
- 📈 Detection accuracy being continuously improved
- ✅ Basic analysis operations working as expected
- 🔄 Ongoing optimization for enhanced accuracy

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
├── frontend/         # React/TypeScript frontend
│   ├── src/         # Source code
│   │   ├── components/  # React components
│   │   ├── hooks/      # Custom React hooks
│   │   ├── utils/      # Utility functions
│   │   └── App.tsx     # Main application component
│   ├── public/      # Static assets
│   └── vite.config.ts  # Vite configuration
├── scripts/         # Utility scripts
│   └── start.sh    # Development startup script
├── data/           # Test and sample images
│   ├── samples/    # Sample images for testing
│   │   ├── original/   # Original untampered images
│   │   └── tampered/   # Known tampered/AI-generated images
│   └── processed/  # Processed results
├── docker-compose.yml  # Docker composition config
├── Dockerfile         # Docker build configuration
└── requirements.txt   # Python dependencies
```

## Installation

### Prerequisites

- Python 3.12 (Python 3.13 is not yet supported by TensorFlow as of April 2025)
- pip (latest version recommended)
- Node.js 18+ (for frontend)
- Docker (optional, for containerized setup)

### Backend Setup

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

### Frontend Setup

1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

### Running the Application

#### Development Mode (Recommended)

The project includes a convenient shell script that starts both frontend and backend in development mode with hot-reload enabled:

```bash
# From the root directory
./scripts/start.sh
```

This will:
- Start the backend server on port 8000 with auto-reload
- Start the frontend dev server on port 5173 with hot module replacement
- Watch for changes in both frontend and backend code
- Output logs from both services in a single terminal

The application will be available at:
- Frontend: `http://localhost:5173`
- Backend API: `http://localhost:8000`
- API Documentation: `http://localhost:8000/docs`

#### Running Services Separately

If you need to run the services separately:

1. Start the backend server:
```bash
cd backend
uvicorn app.main:app --reload --port 8000
```

2. Start the frontend development server:
```bash
cd frontend
npm run dev
```

#### Using Docker

1. Build and run using Docker Compose:
```bash
# Build and start both frontend and backend
docker-compose up --build

# Run in detached mode
docker-compose up -d

# Stop the containers
docker-compose down
```

The application will be available at the same ports as in development mode.

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

The following parameters can be adjusted to fine-tune the analysis:

#### ELA Parameters
- `QUALITY`: JPEG compression quality (default: 90)
- `SCALE_FACTOR`: Sensitivity of the analysis (default: 10)
- `THRESHOLD`: Detection threshold for tampering (default: 40)
*Parameters can be adjusted for specific use cases*

#### PRNU Parameters
- `WINDOW_SIZE`: Size of analysis window (default: 64)
- `STRIDE`: Window stride for analysis (default: 32)
- `SIGMA`: Gaussian filter sigma (default: 5.0)
*Adjustable for different image types and conditions*

#### Entropy Analysis Parameters
- `RADIUS`: Local entropy calculation window size (default: 4)
- `TOLERANCE`: Entropy matching tolerance (default: 0.12)
- `MATCHING_THRESHOLD`: AI detection threshold (default: 0.35)
- `UNIFORMITY_THRESHOLD`: Local pattern uniformity threshold (default: 0.2)
*Parameters can be tuned based on specific detection requirements*

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

## Known Limitations

1. **False Positives**: The current implementation may incorrectly flag legitimate images as tampered or AI-generated
2. **Threshold Sensitivity**: Detection thresholds are not yet optimized for real-world use
3. **AI Detection Accuracy**: The entropy-based AI detection system needs further refinement
4. **Performance Optimization**: Some analysis operations may be slower than necessary
5. **Limited Testing**: While test cases exist, more comprehensive testing with diverse image sets is needed

## Development Goals

1. Optimize detection thresholds through machine learning
2. Reduce false positive rate
3. Improve processing speed
4. Enhance AI-generated content detection accuracy
5. Complete frontend visualization components
6. Add comprehensive error handling

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[MIT License](LICENSE)

## Acknowledgments

- Based on research in digital image forensics
- Uses OpenCV and NumPy for image processing
- FastAPI for REST API implementation
- SciPy for signal processing