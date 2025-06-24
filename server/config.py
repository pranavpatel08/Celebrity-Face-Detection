import os

class Config:
    """Application configuration"""
    
    # Flask settings
    DEBUG = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    PORT = int(os.environ.get('PORT', 5000))
    
    # Model settings
    MODEL_PATH = './artifacts/saved_model.pkl'
    CLASS_DICT_PATH = './artifacts/class_dictionary.json'
    
    # Image processing
    IMG_SIZE = 32
    MAX_IMAGE_SIZE_MB = 10
    MAX_IMAGE_SIZE_BYTES = MAX_IMAGE_SIZE_MB * 1024 * 1024
    
    # Haar cascades
    FACE_CASCADE_PATH = './opencv/haarcascades/haarcascade_frontalface_default.xml'
    EYE_CASCADE_PATH = './opencv/haarcascades/haarcascade_eye.xml'
    
    # Wavelet settings
    WAVELET_MODE = 'db1'
    WAVELET_LEVEL = 5
    
    # Face detection parameters
    SCALE_FACTOR = 1.3
    MIN_NEIGHBORS = 5
    
    # API settings
    ALLOWED_ORIGINS = ['*']  # Configure this for production
    MAX_FACES_PER_IMAGE = 10  # Limit to prevent memory issues
    
    # Logging
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# Create config instance
config = Config()