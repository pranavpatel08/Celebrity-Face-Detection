import numpy as np
import pywt
import cv2
import logging

logger = logging.getLogger(__name__)

def w2d(img, mode='haar', level=1):
    """
    Apply 2D Discrete Wavelet Transform to an image.
    
    This function processes an image by:
    1. Converting to grayscale
    2. Applying wavelet decomposition
    3. Zeroing out the approximation coefficients
    4. Reconstructing the image
    
    Args:
        img: Input image (BGR format)
        mode: Wavelet type (default: 'haar')
        level: Decomposition level (default: 1)
    
    Returns:
        Processed image as uint8 array
    """
    try:
        if img is None:
            raise ValueError("Input image is None")
        
        # Validate input image
        if len(img.shape) != 3:
            raise ValueError(f"Expected 3D image array, got shape: {img.shape}")
        
        # Convert to grayscale
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Convert to float32 and normalize
        float_img = np.float32(gray_img) / 255.0
        
        # Perform wavelet decomposition
        coeffs = pywt.wavedec2(float_img, mode, level=level)
        
        # Process coefficients - zero out approximation coefficients
        coeffs_processed = list(coeffs)
        coeffs_processed[0] *= 0
        
        # Reconstruct image
        reconstructed = pywt.waverec2(coeffs_processed, mode)
        
        # Convert back to uint8
        reconstructed = np.clip(reconstructed * 255, 0, 255)
        result = np.uint8(reconstructed)
        
        logger.debug(f"Wavelet transform completed: input shape {img.shape}, output shape {result.shape}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error in wavelet transform: {str(e)}")
        raise

def get_wavelet_features(img, mode='db1', level=5):
    """
    Extract wavelet features from an image.
    
    Args:
        img: Input image
        mode: Wavelet type (default: 'db1' - Daubechies 1)
        level: Decomposition level (default: 5)
    
    Returns:
        Wavelet transformed image
    """
    return w2d(img, mode, level)

def validate_wavelet_mode(mode):
    """
    Validate if the wavelet mode is supported.
    
    Args:
        mode: Wavelet mode string
    
    Returns:
        bool: True if valid, False otherwise
    """
    try:
        # Get list of available wavelets
        available_wavelets = pywt.wavelist()
        return mode in available_wavelets
    except Exception:
        return False

# Available wavelet families for reference
WAVELET_FAMILIES = {
    'haar': 'Haar',
    'db': 'Daubechies',
    'sym': 'Symlets',
    'coif': 'Coiflets',
    'bior': 'Biorthogonal',
    'rbio': 'Reverse biorthogonal',
    'dmey': 'Discrete Meyer'
}

if __name__ == "__main__":
    # Test wavelet transform
    print("Testing wavelet module...")
    print(f"Available wavelet families: {list(WAVELET_FAMILIES.keys())}")
    
    # Create a test image
    test_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    try:
        result = w2d(test_img)
        print(f"Test successful! Output shape: {result.shape}")
    except Exception as e:
        print(f"Test failed: {str(e)}")