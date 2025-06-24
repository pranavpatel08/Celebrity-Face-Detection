from flask import Flask, request, jsonify
import util
import logging
from functools import wraps
import time

app = Flask(__name__)

# Configure maximum request size (16MB)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max request size

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def log_request(f):
    """Decorator to log request details"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        start_time = time.time()
        logger.info(f"Request: {request.method} {request.path}")
        result = f(*args, **kwargs)
        duration = time.time() - start_time
        logger.info(f"Response time: {duration:.3f}s")
        return result
    return decorated_function

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'model_loaded': util.is_model_loaded()})

@app.route('/classify_image', methods=['POST'])
@log_request
def classify_image():
    """
    Classify celebrity faces in an image.
    Expects: image_data (base64 encoded image) in form data
    Returns: JSON with classification results for all detected faces
    """
    try:
        # Validate request
        if 'image_data' not in request.form:
            return jsonify({'error': 'No image_data provided'}), 400
        
        image_data = request.form['image_data']
        
        # Validate base64 data (more flexible validation)
        if not image_data:
            return jsonify({'error': 'Empty image data'}), 400
        
        # Check if it's base64 (either pure or data URI format)
        if not (image_data.startswith('data:image') or 
                all(c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=' for c in image_data[:100])):
            return jsonify({'error': 'Invalid image data format'}), 400
        
        # Process image
        results = util.classify_image(image_data)
        
        # Enhanced response structure
        response_data = {
            'success': True,
            'faces_detected': len(results),
            'results': results
        }
        
        # Add summary if multiple faces detected
        if len(results) > 1:
            response_data['summary'] = f"Detected {len(results)} faces in the image"
        elif len(results) == 0:
            response_data['message'] = "No faces with 2 eyes detected in the image"
        
        response = jsonify(response_data)
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        
        return response
        
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': 'Failed to process image',
            'message': str(e)
        }), 500

@app.route('/classes', methods=['GET'])
def get_classes():
    """Get list of available celebrity classes"""
    try:
        classes = util.get_class_names()
        return jsonify({
            'success': True,
            'classes': classes,
            'count': len(classes)
        })
    except Exception as e:
        logger.error(f"Error getting classes: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == "__main__":
    print("Starting Python Flask Server For Sports Celebrity Image Classification")
    print("Loading model artifacts...")
    
    try:
        util.load_saved_artifacts()
        print("Model loaded successfully!")
        print(f"Available classes: {util.get_class_names()}")
        app.run(port=5000, debug=False)
    except Exception as e:
        print(f"Failed to start server: {str(e)}")
        exit(1)