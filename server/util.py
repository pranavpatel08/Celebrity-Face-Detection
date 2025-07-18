import joblib
import json
import numpy as np
import base64
import cv2
from wavelet import w2d

__class_name_to_number = {}
__class_number_to_name = {}
__model = None

def classify_image(image_base64_data, file_path=None):
    imgs = get_cropped_image_if_2_eyes(file_path, image_base64_data)
    
    if not imgs:
        return []
    
    results = []
    for idx, img in enumerate(imgs):
        scalled_raw_img = cv2.resize(img, (32, 32))
        img_har = w2d(img, 'db1', 5)
        scalled_img_har = cv2.resize(img_har, (32, 32))
        combined_img = np.vstack((scalled_raw_img.reshape(32 * 32 * 3, 1), scalled_img_har.reshape(32 * 32, 1)))

        len_image_array = 32*32*3 + 32*32
        final = combined_img.reshape(1,len_image_array).astype(float)
        
        # Get prediction and probabilities
        prediction = __model.predict(final)[0]
        probabilities = np.around(__model.predict_proba(final)*100,2).tolist()[0]
        
        # Create a sorted list of all predictions with probabilities
        all_predictions = []
        for class_num, prob in enumerate(probabilities):
            all_predictions.append({
                'name': class_number_to_name(class_num),
                'probability': prob
            })
        # Sort by probability descending
        all_predictions.sort(key=lambda x: x['probability'], reverse=True)
        
        results.append({
            'face_index': idx,
            'predicted_class': class_number_to_name(prediction),
            'top_predictions': all_predictions[:3],  # Top 3 predictions
            'all_probabilities': probabilities,
            'class_dictionary': __class_name_to_number
        })

    return results

def class_number_to_name(class_num):
    return __class_number_to_name[class_num]

def load_saved_artifacts():
    print("loading saved artifacts...start")
    global __class_name_to_number
    global __class_number_to_name
    global __model

    import os
    print(f"Current directory: {os.getcwd()}")
    print(f"Files in current directory: {os.listdir('.')}")
    
    # Check if artifacts directory exists
    if os.path.exists('./artifacts'):
        print(f"Files in artifacts directory: {os.listdir('./artifacts')}")
    else:
        print("ERROR: artifacts directory not found!")
        raise Exception("artifacts directory not found")

    # Load class dictionary
    class_dict_path = "./artifacts/class_dictionary.json"
    if not os.path.exists(class_dict_path):
        print(f"ERROR: {class_dict_path} not found!")
        raise Exception(f"{class_dict_path} not found")
        
    with open(class_dict_path, "r") as f:
        __class_name_to_number = json.load(f)
        __class_number_to_name = {v:k for k,v in __class_name_to_number.items()}
    print(f"Loaded classes: {list(__class_name_to_number.keys())}")

    # Load model
    model_path = './artifacts/saved_model.pkl'
    if not os.path.exists(model_path):
        print(f"ERROR: {model_path} not found!")
        raise Exception(f"{model_path} not found")
        
    if __model is None:
        with open(model_path, 'rb') as f:
            __model = joblib.load(f)
        print(f"Model loaded successfully. Type: {type(__model)}")
    print("loading saved artifacts...done")

def get_cv2_image_from_base64_string(b64str):
    '''
    credit: https://stackoverflow.com/questions/33754935/read-a-base-64-encoded-image-from-memory-using-opencv-python-library
    '''
    encoded_data = b64str.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def get_cropped_image_if_2_eyes(image_path, image_base64_data):
    import os
    
    # Check if cascade files exist
    face_cascade_path = './opencv/haarcascades/haarcascade_frontalface_default.xml'
    eye_cascade_path = './opencv/haarcascades/haarcascade_eye.xml'
    
    if not os.path.exists(face_cascade_path):
        print(f"ERROR: Face cascade not found at {face_cascade_path}")
        # Try alternative paths
        alt_face_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        alt_eye_path = cv2.data.haarcascades + 'haarcascade_eye.xml'
        print(f"Trying alternative path: {alt_face_path}")
        face_cascade = cv2.CascadeClassifier(alt_face_path)
        eye_cascade = cv2.CascadeClassifier(alt_eye_path)
    else:
        face_cascade = cv2.CascadeClassifier(face_cascade_path)
        eye_cascade = cv2.CascadeClassifier(eye_cascade_path)

    if image_path:
        img = cv2.imread(image_path)
    else:
        img = get_cv2_image_from_base64_string(image_base64_data)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    cropped_faces = []
    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) >= 2:
            cropped_faces.append(roi_color)
    return cropped_faces

def get_b64_test_image_for_keanu():
    with open("b64.txt") as f:
        return f.read()

def test_classify_image():
    load_saved_artifacts()
    results = classify_image(None, "./testing2.jpg")
    return results