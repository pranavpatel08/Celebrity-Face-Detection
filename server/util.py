import joblib
import json
import numpy as np
import base64
import cv2
from wavelet import w2d
import os
import logging

logger = logging.getLogger(__name__)

# Global variables for model and mappings
__class_name_to_number = {}
__class_number_to_name = {}
__model = None

# Constants
IMG_SIZE = 32
MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB limit

def classify_image(image_base64_data, file_path=None):
    """
    Classify faces in an image.
    Returns a list of results for each detected face with 2 eyes.
    """
    try:
        # Get cropped faces
        cropped_faces = get_cropped_image_if_2_eyes(file_path, image_base64_data)
        
        if not cropped_faces:
            logger.warning("No faces with 2 eyes detected")
            return []
        
        logger.info(f"Processing {len(cropped_faces)} face(s)")
        
        results = []
        for idx, face_img in enumerate(cropped_faces):
            try:
                # Process single face
                result = classify_single_face(face_img, idx)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing face {idx}: {str(e)}")
                continue
        
        return results
        
    except Exception as e:
        logger.error(f"Error in classify_image: {str(e)}")
        raise

def classify_single_face(img, face_index=0):
    """Process a single face image and return classification results"""
    # Resize raw image
    scaled_raw_img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    
    # Apply wavelet transform
    img_har = w2d(img, 'db1', 5)
    scaled_img_har = cv2.resize(img_har, (IMG_SIZE, IMG_SIZE))
    
    # Combine features
    combined_img = np.vstack((
        scaled_raw_img.reshape(IMG_SIZE * IMG_SIZE * 3, 1),
        scaled_img_har.reshape(IMG_SIZE * IMG_SIZE, 1)
    ))
    
    # Prepare for prediction
    len_image_array = IMG_SIZE * IMG_SIZE * 3 + IMG_SIZE * IMG_SIZE
    final = combined_img.reshape(1, len_image_array).astype(float)
    
    # Get prediction
    predicted_class = __model.predict(final)[0]
    probabilities = __model.predict_proba(final)[0]
    
    # Sort probabilities with class names for better UI display
    class_probabilities = []
    for idx, prob in enumerate(probabilities):
        class_probabilities.append({
            'name': __class_number_to_name[idx],
            'probability': round(prob * 100, 2)
        })
    
    # Sort by probability descending
    class_probabilities.sort(key=lambda x: x['probability'], reverse=True)
    
    return {
        'face_index': face_index,
        'predicted_class': class_number_to_name(predicted_class),
        'confidence': round(max(probabilities) * 100, 2),
        'all_probabilities': class_probabilities,
        'top_3': class_probabilities[:3]  # For quick display
    }

def class_number_to_name(class_num):
    """Convert class number to celebrity name"""
    return __class_number_to_name.get(class_num, "Unknown")

def get_class_names():
    """Get list of all celebrity names"""
    return list(__class_name_to_number.keys())

def is_model_loaded():
    """Check if model is loaded"""
    return __model is not None

def load_saved_artifacts():
    """Load saved model and class mappings"""
    global __class_name_to_number
    global __class_number_to_name
    global __model
    
    try:
        # Check if artifacts directory exists
        artifacts_dir = "./artifacts"
        if not os.path.exists(artifacts_dir):
            raise FileNotFoundError(f"Artifacts directory not found: {artifacts_dir}")
        
        # Load class dictionary
        class_dict_path = os.path.join(artifacts_dir, "class_dictionary.json")
        if not os.path.exists(class_dict_path):
            raise FileNotFoundError(f"Class dictionary not found: {class_dict_path}")
            
        with open(class_dict_path, "r") as f:
            __class_name_to_number = json.load(f)
            __class_number_to_name = {v: k for k, v in __class_name_to_number.items()}
        
        logger.info(f"Loaded {len(__class_name_to_number)} classes")
        
        # Load model
        model_path = os.path.join(artifacts_dir, "saved_model.pkl")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        with open(model_path, 'rb') as f:
            __model = joblib.load(f)
        
        logger.info("Model loaded successfully")
        
    except Exception as e:
        logger.error(f"Error loading artifacts: {str(e)}")
        raise

def get_cv2_image_from_base64_string(b64str):
    """Convert base64 string to cv2 image (supports both pure base64 and data URI format)"""
    try:
        # Handle different input formats
        if b64str.startswith('data:'):
            # Data URI format: data:image/png;base64,iVBORw0KGgo...
            # Extract the base64 part after the comma
            encoded_data = b64str.split(',')[1]
        else:
            # Pure base64 string
            encoded_data = b64str
        
        # Decode base64
        decoded = base64.b64decode(encoded_data)
        
        # Check size limit
        if len(decoded) > MAX_IMAGE_SIZE:
            raise ValueError(f"Image too large. Maximum size: {MAX_IMAGE_SIZE/1024/1024}MB")
        
        # Convert to numpy array and decode image
        nparr = np.frombuffer(decoded, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("Failed to decode image")
        
        return img
        
    except Exception as e:
        logger.error(f"Error decoding base64 image: {str(e)}")
        raise

def get_cropped_image_if_2_eyes(image_path, image_base64_data):
    """
    Detect faces with 2 eyes and return cropped face images.
    Returns list of cropped face images.
    """
    try:
        # Load cascade classifiers
        face_cascade_path = './opencv/haarcascades/haarcascade_frontalface_default.xml'
        eye_cascade_path = './opencv/haarcascades/haarcascade_eye.xml'
        
        if not os.path.exists(face_cascade_path) or not os.path.exists(eye_cascade_path):
            raise FileNotFoundError("Haar cascade files not found")
        
        face_cascade = cv2.CascadeClassifier(face_cascade_path)
        eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
        
        # Get image
        if image_path:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
            img = cv2.imread(image_path)
        else:
            img = get_cv2_image_from_base64_string(image_base64_data)
        
        # Convert to grayscale for detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        logger.info(f"Detected {len(faces)} face(s)")
        
        cropped_faces = []
        for i, (x, y, w, h) in enumerate(faces):
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            
            # Detect eyes in face region
            eyes = eye_cascade.detectMultiScale(roi_gray)
            
            if len(eyes) >= 2:
                logger.info(f"Face {i}: Found {len(eyes)} eyes")
                cropped_faces.append(roi_color)
            else:
                logger.info(f"Face {i}: Only {len(eyes)} eye(s) detected, skipping")
        
        return cropped_faces
        
    except Exception as e:
        logger.error(f"Error in face detection: {str(e)}")
        raise

def get_b64_test_image_for_testing():
    """Load test image for testing purposes"""
    try:
        with open("b64.txt") as f:
            return f.read()
    except Exception as e:
        logger.error(f"Error loading test image: {str(e)}")
        return None

if __name__ == '__main__':
    # Test the module
    logging.basicConfig(level=logging.INFO)
    
    print("Testing util module...")
    load_saved_artifacts()
    
    # Test with sample image if available
    test_image = get_b64_test_image_for_testing()
    if test_image:
        print("Testing classification...")
        results = classify_image(test_image, None)
        print(f"Results: {json.dumps(results, indent=2)}")
    else:
        print("No test image available")