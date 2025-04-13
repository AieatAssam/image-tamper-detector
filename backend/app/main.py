"""
Main FastAPI application for image tampering detection.
"""
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.utils import get_openapi
from pathlib import Path
import os
import base64
import cv2
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
import logging

from backend.app.analysis.ela import ELAAnalyzer
from backend.app.analysis.prnu import PRNUAnalyzer
from backend.app.analysis.entropy import EntropyAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def encode_image_to_base64(image_array) -> str:
    """Convert numpy array to base64 string."""
    success, buffer = cv2.imencode('.png', image_array)
    if not success:
        raise ValueError("Failed to encode image")
    return base64.b64encode(buffer).decode('utf-8')

# Response models
class AnalysisDetails(BaseModel):
    """Details of the analysis result."""
    method: str = Field(..., description="The analysis method used")
    description: str = Field(..., description="Description of the analysis method")
    edge_discontinuity: Optional[float] = Field(None, description="Measure of edge consistency in the image")
    texture_variance: Optional[float] = Field(None, description="Variance in texture patterns")
    noise_consistency: Optional[float] = Field(None, description="Consistency of noise patterns")
    compression_artifacts: Optional[float] = Field(None, description="Level of compression artifacts")
    matching_proportion: Optional[float] = Field(None, description="Proportion of matching entropy patterns")

class AnalysisResponse(BaseModel):
    """Response model for individual analysis endpoints."""
    is_tampered: bool = Field(..., description="Whether the image is detected as tampered/AI-generated")
    confidence_score: float = Field(..., ge=0, le=1, description="Confidence score of the detection (0-1)")
    analysis_type: str = Field(..., description="Type of analysis performed (ELA, PRNU, or Entropy)")
    visualization_base64: str = Field(..., description="Base64 encoded visualization image showing detected issues")
    details: Dict[str, Any] = Field(..., description="Detailed analysis results")

    class Config:
        json_schema_extra = {
            "example": {
                "is_tampered": True,
                "confidence_score": 0.85,
                "analysis_type": "ELA",
                "visualization_base64": "base64_encoded_image_string",
                "details": {
                    "edge_discontinuity": 0.65,
                    "texture_variance": 1500.0,
                    "noise_consistency": 22.5,
                    "compression_artifacts": 85.0,
                    "method": "Error Level Analysis",
                    "description": "Analysis of JPEG compression artifacts"
                }
            }
        }

class CombinedAnalysisResponse(BaseModel):
    """Response model for combined analysis endpoint."""
    is_tampered: bool = Field(..., description="Whether any analysis detected tampering")
    confidence_score: float = Field(..., ge=0, le=1, description="Combined confidence score (0-1)")
    ela_result: Optional[Dict[str, Any]] = Field(None, description="Results from ELA analysis")
    prnu_result: Optional[Dict[str, Any]] = Field(None, description="Results from PRNU analysis")
    entropy_result: Optional[Dict[str, Any]] = Field(None, description="Results from entropy analysis")
    ela_visualization_base64: Optional[str] = Field(None, description="Base64 encoded ELA visualization")
    prnu_visualization_base64: Optional[str] = Field(None, description="Base64 encoded PRNU visualization")
    entropy_visualization_base64: Optional[str] = Field(None, description="Base64 encoded entropy visualization")

    class Config:
        json_schema_extra = {
            "example": {
                "is_tampered": True,
                "confidence_score": 0.75,
                "ela_visualization_base64": "base64_encoded_image_string",
                "prnu_visualization_base64": "base64_encoded_image_string",
                "entropy_visualization_base64": "base64_encoded_image_string",
                "ela_result": {
                    "is_tampered": True,
                    "edge_discontinuity": 0.65,
                    "texture_variance": 1500.0,
                    "noise_consistency": 22.5,
                    "compression_artifacts": 85.0
                },
                "prnu_result": {
                    "is_tampered": False,
                    "method": "Photo Response Non-Uniformity Analysis"
                },
                "entropy_result": {
                    "is_tampered": True,
                    "matching_proportion": 0.3,
                    "method": "Entropy Analysis"
                }
            }
        }

# Initialize FastAPI app
app = FastAPI(
    title="Image Tampering Detection API",
    description="""
    API for detecting image tampering and AI-generated content using multiple analysis techniques.
    
    Available methods:
    * **ELA (Error Level Analysis)**: Detects inconsistencies in JPEG compression artifacts
    * **PRNU (Photo Response Non-Uniformity)**: Analyzes camera sensor patterns
    * **Entropy Analysis**: Detects patterns common in AI-generated images
    
    Use the individual endpoints for specific analysis methods or the combined endpoint
    for comprehensive analysis using all methods.
    """,
    version="1.0.0",
    contact={
        "name": "Image Tampering Detection Team",
        "url": "https://github.com/yourusername/image-tamper-detector",
    },
    license_info={
        "name": "MIT",
    }
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize analyzers
ela_analyzer = ELAAnalyzer()
prnu_analyzer = PRNUAnalyzer()
entropy_analyzer = EntropyAnalyzer()

@app.get("/", tags=["Info"])
async def root():
    """
    Get API information and available endpoints.
    
    Returns:
        dict: Basic information about the API and its endpoints
    """
    return {
        "name": "Image Tampering Detection API",
        "version": "1.0.0",
        "endpoints": [
            "/analyze/ela",
            "/analyze/prnu",
            "/analyze/entropy",
            "/analyze/combined"
        ]
    }

@app.post("/analyze/ela", response_model=AnalysisResponse, tags=["Analysis"])
async def analyze_ela(
    file: UploadFile = File(..., description="Image file to analyze (JPEG, PNG)")
):
    """
    Perform Error Level Analysis (ELA) on an uploaded image.
    
    This endpoint analyzes JPEG compression artifacts to detect potential image tampering.
    It works best with JPEG images and can detect:
    * Copy-paste manipulation
    * Digital splicing
    * Object removal
    
    Args:
        file: Image file to analyze (JPEG recommended)
        
    Returns:
        AnalysisResponse: Analysis results including tampering detection, visualization, and confidence score
        
    Raises:
        HTTPException: If file processing or analysis fails
    """
    try:
        contents = await file.read()
        is_tampered, visualization, features = ela_analyzer.detect_tampering(contents)
        
        return {
            "is_tampered": is_tampered,
            "confidence_score": 1 - min(1.0, features.edge_discontinuity),
            "analysis_type": "ELA",
            "visualization_base64": encode_image_to_base64(visualization),
            "details": {
                "edge_discontinuity": float(features.edge_discontinuity),
                "texture_variance": float(features.texture_variance),
                "noise_consistency": float(features.noise_consistency),
                "compression_artifacts": float(features.compression_artifacts),
                "method": "Error Level Analysis",
                "description": "Analysis of JPEG compression artifacts"
            }
        }

    except Exception as e:
        logger.error(f"Error in ELA analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze/prnu", response_model=AnalysisResponse, tags=["Analysis"])
async def analyze_prnu(
    file: UploadFile = File(..., description="Image file to analyze (JPEG, PNG)")
):
    """
    Perform PRNU (Photo Response Non-Uniformity) analysis on an uploaded image.
    
    This endpoint analyzes camera sensor patterns to detect potential image tampering.
    It works by detecting:
    * Inconsistencies in sensor noise patterns
    * Areas from different source cameras
    * Digital manipulation that disrupts sensor patterns
    
    Args:
        file: Image file to analyze
        
    Returns:
        AnalysisResponse: Analysis results including tampering detection, visualization, and confidence score
        
    Raises:
        HTTPException: If file processing or analysis fails
    """
    try:
        contents = await file.read()
        is_tampered, visualization = prnu_analyzer.detect_tampering(contents)
        
        return {
            "is_tampered": is_tampered,
            "confidence_score": 0.8,  # Default confidence
            "analysis_type": "PRNU",
            "visualization_base64": encode_image_to_base64(visualization),
            "details": {
                "method": "Photo Response Non-Uniformity Analysis",
                "description": "Analysis of sensor pattern noise"
            }
        }

    except Exception as e:
        logger.error(f"Error in PRNU analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze/entropy", response_model=AnalysisResponse, tags=["Analysis"])
async def analyze_entropy(
    file: UploadFile = File(..., description="Image file to analyze (JPEG, PNG)")
):
    """
    Perform entropy analysis on an uploaded image to detect AI-generated content.
    
    This endpoint analyzes entropy patterns to detect AI-generated images.
    It can detect:
    * Images generated by AI models (e.g., DALL-E, Stable Diffusion)
    * Unusual entropy patterns not typical in natural photos
    * Certain types of digital manipulation
    
    Args:
        file: Image file to analyze
        
    Returns:
        AnalysisResponse: Analysis results including AI-generation detection, visualization, and confidence score
        
    Raises:
        HTTPException: If file processing or analysis fails
    """
    try:
        contents = await file.read()
        is_ai_generated, visualization, matching_proportion = entropy_analyzer.detect_ai_generated(contents)
        
        return {
            "is_tampered": is_ai_generated,
            "confidence_score": float(1 - matching_proportion),
            "analysis_type": "Entropy",
            "visualization_base64": encode_image_to_base64(visualization),
            "details": {
                "matching_proportion": float(matching_proportion),
                "method": "Entropy Analysis",
                "description": "Detection of AI-generated content patterns"
            }
        }

    except Exception as e:
        logger.error(f"Error in entropy analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze/combined", response_model=CombinedAnalysisResponse, tags=["Analysis"])
async def analyze_combined(
    file: UploadFile = File(..., description="Image file to analyze (JPEG, PNG)")
):
    """
    Perform combined analysis using all available methods.
    
    This endpoint runs all three analysis methods (ELA, PRNU, and Entropy)
    and provides a comprehensive result. This is useful for:
    * Higher confidence detection
    * Cross-validation of results
    * Detecting multiple types of manipulation
    
    The combined confidence score is weighted based on the reliability
    of each method for different types of manipulation.
    
    Args:
        file: Image file to analyze
        
    Returns:
        CombinedAnalysisResponse: Combined analysis results including visualizations from all methods
        
    Raises:
        HTTPException: If file processing or analysis fails
    """
    try:
        contents = await file.read()
        ela_tampered, ela_vis, ela_features = ela_analyzer.detect_tampering(contents)
        prnu_tampered, prnu_vis = prnu_analyzer.detect_tampering(contents)
        entropy_tampered, entropy_vis, matching_proportion = entropy_analyzer.detect_ai_generated(contents)

        # Convert NumPy booleans to Python booleans
        ela_tampered = bool(ela_tampered)
        prnu_tampered = bool(prnu_tampered)
        entropy_tampered = bool(entropy_tampered)

        confidence_scores = [
            1 - min(1.0, ela_features.edge_discontinuity),  # ELA confidence
            0.8 if prnu_tampered else 0.2,  # PRNU confidence
            float(1 - matching_proportion)  # Entropy confidence
        ]
        
        combined_confidence = sum(confidence_scores) / len(confidence_scores)
        is_tampered = any([ela_tampered, prnu_tampered, entropy_tampered])

        return {
            "is_tampered": bool(is_tampered),  # Convert to Python boolean
            "confidence_score": float(combined_confidence),  # Ensure float
            "ela_visualization_base64": encode_image_to_base64(ela_vis),
            "prnu_visualization_base64": encode_image_to_base64(prnu_vis),
            "entropy_visualization_base64": encode_image_to_base64(entropy_vis),
            "ela_result": {
                "is_tampered": ela_tampered,
                "confidence_score": float(1 - min(1.0, ela_features.edge_discontinuity)),
                "analysis_type": "ELA",
                "visualization_base64": encode_image_to_base64(ela_vis),
                "details": {
                    "method": "Error Level Analysis",
                    "description": "Analysis of JPEG compression artifacts",
                    "edge_discontinuity": float(ela_features.edge_discontinuity),
                    "texture_variance": float(ela_features.texture_variance),
                    "noise_consistency": float(ela_features.noise_consistency),
                    "compression_artifacts": float(ela_features.compression_artifacts)
                }
            },
            "prnu_result": {
                "is_tampered": prnu_tampered,
                "confidence_score": float(0.8 if prnu_tampered else 0.2),
                "analysis_type": "PRNU",
                "visualization_base64": encode_image_to_base64(prnu_vis),
                "details": {
                    "method": "Photo Response Non-Uniformity Analysis",
                    "description": "Analysis of camera sensor patterns"
                }
            },
            "entropy_result": {
                "is_tampered": entropy_tampered,
                "confidence_score": float(1 - matching_proportion),
                "analysis_type": "Entropy",
                "visualization_base64": encode_image_to_base64(entropy_vis),
                "details": {
                    "method": "Entropy Analysis",
                    "description": "Analysis of patterns common in AI-generated images",
                    "matching_proportion": float(matching_proportion)
                }
            }
        }

    except Exception as e:
        logger.error(f"Error in combined analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def custom_openapi():
    """Generate custom OpenAPI schema with additional metadata."""
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )

    # Add security schemes if needed
    # openapi_schema["components"]["securitySchemes"] = {...}

    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi
