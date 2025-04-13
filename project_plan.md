# Image Tamper Detection Web Application - Project Plan

## Overview
A web application that performs multiple image analysis techniques to detect and visualize areas of image manipulation or generation. The application will consist of a FastAPI backend and a simple web frontend.

## Technology Stack

### Backend
- FastAPI (MIT License) - Modern, fast web framework for building APIs
- Python 3.9+ 
- OpenCV (Apache 2.0) - For image processing and ELA
- NumPy (BSD-3) - For numerical computations
- scikit-image (BSD-3) - For advanced image processing
- Pillow (HPND) - For basic image handling
- python-multipart (MIT) - For handling file uploads
- slowapi (MIT) - For rate limiting
- uvicorn (BSD) - ASGI server
- c2pa-python (MIT/Apache-2.0) - For C2PA metadata scanning and validation

### Frontend
- HTML5/CSS3
- JavaScript (ES6+)
- Bootstrap 5 (MIT) - For responsive UI
- Chart.js (MIT) - For visualization
- Axios (MIT) - For API calls

## Implementation Checklist

### 1. Project Setup
- [x] Create project directory structure
- [x] Set up virtual environment
- [x] Create requirements.txt with all dependencies
- [x] Initialize git repository
- [x] Create README.md with setup instructions

### 2. Backend Development

#### Core Image Analysis Components
- [x] Implement Error Level Analysis (ELA)
  - [x] Image resaving with different quality levels
  - [x] Difference calculation
  - [x] Threshold-based anomaly detection

- [ ] Implement PRNU (Photo Response Non-Uniformity) Analysis
  - [x] Noise extraction
  - [x] Pattern matching
  - [x] Correlation analysis

- [x] Implement Entropy-based AI Detection
  - [x] Local entropy calculation
  - [x] Cross-channel pattern analysis
  - [x] Threshold-based classification
  - [x] Visualization of suspicious regions
  - [x] Parameter tuning and validation
  - [x] Test cases for natural and AI-generated images

- [ ] Implement C2PA Metadata Analysis
  - [ ] C2PA manifest extraction
  - [ ] Signature validation
  - [ ] Provenance chain verification
  - [ ] Claim verification and parsing
  - [ ] Integration with other analysis results

#### API Development
- [ ] Set up FastAPI application structure
- [ ] Implement file upload endpoint
- [ ] Create analysis pipeline
  - [ ] ELA analysis integration
  - [ ] PRNU analysis integration
  - [ ] Entropy analysis integration
  - [ ] Combined results aggregation
- [ ] Implement rate limiting
- [ ] Add error handling
- [ ] Create response models
- [ ] Add input validation
- [ ] Implement CORS

### 3. Frontend Development

#### Basic Structure
- [ ] Create HTML template
- [ ] Set up CSS styling
- [ ] Implement responsive design
- [ ] Create file upload interface

#### Visualization Components
- [ ] Implement image preview
- [ ] Create heatmap overlay for tampered areas
- [ ] Add analysis results display
  - [ ] ELA visualization
  - [ ] PRNU correlation map
  - [ ] Entropy pattern visualization
  - [ ] Combined analysis view
- [ ] Implement loading states
- [ ] Add error handling and user feedback

### 4. Integration
- [ ] Connect frontend to backend API
- [ ] Implement proper error handling
- [ ] Add loading indicators
- [ ] Test end-to-end functionality

### 5. Testing
- [x] Create test dataset of manipulated images
- [x] Write unit tests for analysis components
  - [x] ELA analysis tests
  - [x] PRNU analysis tests
  - [x] Entropy analysis tests
    - [x] Natural image detection
    - [x] AI-generated image detection
    - [x] Parameter validation
  - [ ] C2PA metadata analysis tests
    - [ ] Manifest extraction tests
    - [ ] Signature validation tests
    - [ ] Provenance chain tests
    - [ ] Integration tests with other detectors
- [ ] Write integration tests for API endpoints
- [ ] Perform frontend testing
- [ ] Conduct performance testing

### 6. Documentation
- [ ] API documentation
- [ ] Setup instructions
- [ ] Usage guide
- [ ] Analysis methodology explanation
  - [ ] ELA methodology
  - [ ] PRNU methodology
  - [x] Entropy analysis methodology
    - [x] Parameter descriptions
    - [x] Detection thresholds
    - [x] Interpretation guidelines
  - [ ] C2PA analysis methodology
    - [ ] Manifest structure explanation
    - [ ] Validation process
    - [ ] Trust chain verification
    - [ ] Integration with other analysis methods

### 7. Deployment
- [ ] Set up production environment
- [ ] Configure CORS and security settings
- [ ] Set up proper rate limiting for production
- [ ] Create deployment documentation

## Directory Structure
```
image-tamper-detector/
├── data/
│   └── samples/
│       ├── original/     # Original, untampered images
│       └── tampered/     # Known tampered images for testing
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── analysis/
│   │   │   ├── __init__.py
│   │   │   ├── ela.py
│   │   │   ├── entropy.py
│   │   │   ├── prnu.py
│   │   │   └── c2pa.py
│   │   ├── api/
│   │   │   ├── __init__.py
│   │   │   └── endpoints.py
│   │   └── utils/
│   │       ├── __init__.py
│   │       └── helpers.py
│   ├── tests/
│   │   └── ...
│   └── requirements.txt
├── frontend/
│   ├── index.html
│   ├── css/
│   │   └── styles.css
│   └── js/
│       └── main.js
├── README.md
└── project_plan.md
```

## Notes
- All image processing should be done asynchronously to prevent blocking
- Implement proper error handling for various image formats and sizes
- Consider adding image size limits and format restrictions
- Implement proper cleanup of temporary files
- Consider adding caching for frequently analyzed images
- Add proper logging for debugging and monitoring
- Maintain a diverse set of sample images for testing and demonstration:
  - Original images should be unmodified and from reliable sources
  - Tampered images should include various manipulation techniques
  - AI-generated images should include various types (not just text-based)
  - Include metadata about the type of tampering/generation in each sample
- Entropy analysis parameters:
  - radius = 4 (local entropy calculation)
  - tolerance = 0.12 (entropy matching)
  - matching_threshold = 0.35 (AI detection)
  - uniformity_threshold = 0.2 (local patterns) 