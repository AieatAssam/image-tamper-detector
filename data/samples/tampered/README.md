# Tampered Image Samples

This directory contains tampered images for testing the image tampering detection system.

## Existing Images

1. `gpt-4o-generated-receipt-01.png`
   - Type: AI-Generated
   - Tool: GPT-4
   - Description: AI-generated receipt

2. `gpt-4o-generated-receipt-02.png`
   - Type: AI-Generated
   - Tool: GPT-4
   - Description: AI-generated receipt

3. `landscape_copy_paste.jpg`
   - Original: Based on landscape_original.jpg
   - Resolution: 8640x5760
   - Manipulation Type: Copy-paste with blending
   - Manipulation Details:
     - Region copied from 70% width, 20% height
     - Pasted to 30% width, 20% height
     - Size: 15% of original dimensions
     - Circular mask with Gaussian blending
   - Technical Details:
     - JPEG quality: 92
     - Feathered edges (Gaussian blur sigma=11)
     - Smooth alpha blending at boundaries
   - Expected ELA Response:
     - High error levels at copied region
     - Visible compression artifacts at blend boundaries
     - Distinct pattern in tampered area



Note: Each tampered image is designed to test different aspects of the ELA detection:
- Copy-paste detection
- AI-generated content
- Local modifications
- Global adjustments
- Text manipulation
- Object removal/addition 