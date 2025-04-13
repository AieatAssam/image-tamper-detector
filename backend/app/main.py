"""
Main FastAPI application for image tampering detection.
"""
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pathlib import Path
import tempfile
import shutil
import os
from typing import Dict, Any, Optional
from pydantic import BaseModel
import logging

from app.analysis.ela import ELAAnalyzer
from app.analysis.prnu import PRNUAnalyzer
from app.analysis.entropy import EntropyAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Image Tampering Detection API",
    description="API for detecting image tampering and AI-generated content using multiple analysis techniques",
    version="1.0.0"
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

# Response models
class AnalysisResponse(BaseModel):
    is_tampered: bool
    confidence_score: float
    analysis_type: str
    details: Dict[str, Any]

class CombinedAnalysisResponse(BaseModel):
    is_tampered: bool
    confidence_score: float
    ela_result: Optional[Dict[str, Any]]
    prnu_result: Optional[Dict[str, Any]]
    entropy_result: Optional[Dict[str, Any]]

@app.get("/")
async def root():
    """Root endpoint returning API information."""
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

@app.post("/analyze/ela", response_model=AnalysisResponse)
async def analyze_ela(file: UploadFile = File(...)):
    """
    Perform Error Level Analysis (ELA) on an uploaded image.
    """
    try:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_path = Path(temp_file.name)

        # Perform ELA analysis
        is_tampered, _, features = ela_analyzer.detect_tampering(temp_path)
        
        # Create response
        response = {
            "is_tampered": is_tampered,
            "confidence_score": 1 - min(1.0, features.edge_discontinuity),
            "analysis_type": "ELA",
            "details": {
                "edge_discontinuity": float(features.edge_discontinuity),
                "texture_variance": float(features.texture_variance),
                "noise_consistency": float(features.noise_consistency),
                "compression_artifacts": float(features.compression_artifacts)
            }
        }

        # Cleanup
        os.unlink(temp_path)
        return response

    except Exception as e:
        logger.error(f"Error in ELA analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze/prnu", response_model=AnalysisResponse)
async def analyze_prnu(file: UploadFile = File(...)):
    """
    Perform PRNU analysis on an uploaded image.
    """
    try:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_path = Path(temp_file.name)

        # Perform PRNU analysis
        is_tampered, _ = prnu_analyzer.detect_tampering(temp_path)
        
        # Create response
        response = {
            "is_tampered": is_tampered,
            "confidence_score": 0.8,  # Default confidence
            "analysis_type": "PRNU",
            "details": {
                "method": "Photo Response Non-Uniformity Analysis",
                "description": "Analysis of sensor pattern noise"
            }
        }

        # Cleanup
        os.unlink(temp_path)
        return response

    except Exception as e:
        logger.error(f"Error in PRNU analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze/entropy", response_model=AnalysisResponse)
async def analyze_entropy(file: UploadFile = File(...)):
    """
    Perform entropy analysis on an uploaded image to detect AI-generated content.
    """
    try:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_path = Path(temp_file.name)

        # Perform entropy analysis
        is_ai_generated, _, matching_proportion = entropy_analyzer.detect_ai_generated(temp_path)
        
        # Create response
        response = {
            "is_tampered": is_ai_generated,
            "confidence_score": float(1 - matching_proportion),
            "analysis_type": "Entropy",
            "details": {
                "matching_proportion": float(matching_proportion),
                "method": "Entropy Analysis",
                "description": "Detection of AI-generated content patterns"
            }
        }

        # Cleanup
        os.unlink(temp_path)
        return response

    except Exception as e:
        logger.error(f"Error in entropy analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze/combined", response_model=CombinedAnalysisResponse)
async def analyze_combined(file: UploadFile = File(...)):
    """
    Perform combined analysis using all available methods.
    """
    try:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_path = Path(temp_file.name)

        # Perform all analyses
        ela_tampered, _, ela_features = ela_analyzer.detect_tampering(temp_path)
        prnu_tampered, _ = prnu_analyzer.detect_tampering(temp_path)
        entropy_tampered, _, matching_proportion = entropy_analyzer.detect_ai_generated(temp_path)

        # Calculate combined confidence and result
        confidence_scores = [
            1 - min(1.0, ela_features.edge_discontinuity),  # ELA confidence
            0.8 if prnu_tampered else 0.2,  # PRNU confidence
            float(1 - matching_proportion)  # Entropy confidence
        ]
        
        combined_confidence = sum(confidence_scores) / len(confidence_scores)
        is_tampered = any([ela_tampered, prnu_tampered, entropy_tampered])

        # Create response
        response = {
            "is_tampered": is_tampered,
            "confidence_score": combined_confidence,
            "ela_result": {
                "is_tampered": ela_tampered,
                "edge_discontinuity": float(ela_features.edge_discontinuity),
                "texture_variance": float(ela_features.texture_variance),
                "noise_consistency": float(ela_features.noise_consistency),
                "compression_artifacts": float(ela_features.compression_artifacts)
            },
            "prnu_result": {
                "is_tampered": prnu_tampered,
                "method": "Photo Response Non-Uniformity Analysis"
            },
            "entropy_result": {
                "is_tampered": entropy_tampered,
                "matching_proportion": float(matching_proportion),
                "method": "Entropy Analysis"
            }
        }

        # Cleanup
        os.unlink(temp_path)
        return response

    except Exception as e:
        logger.error(f"Error in combined analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
