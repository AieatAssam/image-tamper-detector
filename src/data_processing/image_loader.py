import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Union

class ImageLoader:
    def __init__(self, image_size: Tuple[int, int] = (224, 224)):
        """
        Initialize the ImageLoader with specified image size.
        
        Args:
            image_size (Tuple[int, int]): Target size for loaded images (height, width)
        """
        self.image_size = image_size

    def load_image(self, image_path: Union[str, Path]) -> np.ndarray:
        """
        Load and preprocess a single image.
        
        Args:
            image_path (Union[str, Path]): Path to the image file
            
        Returns:
            np.ndarray: Preprocessed image array
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found at {image_path}")
            
        # Read image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to load image at {image_path}")
            
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize image
        image = cv2.resize(image, self.image_size)
        
        # Normalize pixel values
        image = image.astype(np.float32) / 255.0
        
        return image

    def load_directory(self, directory_path: Union[str, Path]) -> List[np.ndarray]:
        """
        Load all images from a directory.
        
        Args:
            directory_path (Union[str, Path]): Path to the directory containing images
            
        Returns:
            List[np.ndarray]: List of preprocessed images
        """
        directory_path = Path(directory_path)
        if not directory_path.exists():
            raise FileNotFoundError(f"Directory not found at {directory_path}")
            
        image_paths = list(directory_path.glob("*.jpg")) + list(directory_path.glob("*.png"))
        return [self.load_image(path) for path in image_paths] 