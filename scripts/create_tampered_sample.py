"""
Script to create a tampered version of the landscape image by copying and pasting regions.
"""
import cv2
import numpy as np
from pathlib import Path

def create_tampered_image(input_path: str, output_path: str):
    """
    Create a tampered version of the input image by:
    1. Copying a region and pasting it elsewhere
    2. Blending the edges
    3. Saving with different compression
    """
    # Read the original image
    img = cv2.imread(input_path)
    if img is None:
        raise ValueError(f"Could not read image: {input_path}")
    
    # resize image to 1920x1080
    img = cv2.resize(img, (1920, 1080))

    # Get dimensions
    height, width = img.shape[:2]
    
    # Define source region (a mountain peak or cloud section)
    src_x, src_y = int(width * 0.7), int(height * 0.2)  # Source coordinates
    region_w, region_h = int(width * 0.15), int(height * 0.15)  # Region size
    
    # Extract the region
    source_region = img[src_y:src_y+region_h, src_x:src_x+region_w].copy()
    
    # Create a mask for smooth blending
    mask = np.zeros((region_h, region_w), dtype=np.float32)
    cv2.circle(mask, 
               (region_w//2, region_h//2),
               min(region_w, region_h)//2,
               1.0,
               -1)
    
    # Feather the edges
    mask = cv2.GaussianBlur(mask, (21, 21), 11)
    mask = np.dstack([mask] * 3)  # Create 3-channel mask
    
    # Define destination region (different location)
    dst_x, dst_y = int(width * 0.3), int(height * 0.2)
    
    # Create the tampered image
    tampered = img.copy()
    
    # Blend the region into the new location
    blended_region = (source_region * mask + 
                     tampered[dst_y:dst_y+region_h, dst_x:dst_x+region_w] * (1 - mask))
    tampered[dst_y:dst_y+region_h, dst_x:dst_x+region_w] = blended_region
    
    # Save with different compression quality
    cv2.imwrite(output_path, tampered, [cv2.IMWRITE_JPEG_QUALITY, 92])
    
    print(f"Created tampered image: {output_path}")

if __name__ == "__main__":
    # Setup paths
    root_dir = Path(__file__).parent.parent
    original_path = str(root_dir / "data/samples/original/landscape_original.jpg")
    tampered_path = str(root_dir / "data/samples/tampered/landscape_copy_paste.jpg")
    
    # Create tampered version
    create_tampered_image(original_path, tampered_path) 