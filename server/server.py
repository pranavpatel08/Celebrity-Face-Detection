from flask import Flask, request, jsonify
from flask_cors import CORS
import util
import traceback
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Increase maximum allowed payload to 16MB
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

# Load model when the server starts
print("=== Starting Server ===")
print(f"Current working directory: {os.getcwd()}")
print(f"Directory contents: {os.listdir('.')}")

# Load artifacts on startup
try:
    util.load_saved_artifacts()
    print("=== Model loaded successfully ===")
except Exception as e:
    print(f"=== ERROR loading model: {str(e)} ===")
    print(traceback.format_exc())


@app.route('/classify_image', methods=['GET', 'POST'])
def classify_image():
    try:
        image_data = request.form.get('image_data')
        
        if not image_data:
            response = jsonify({'error': 'No image data provided'})
            response.headers.add('Access-Control-Allow-Origin', '*')
            return response, 400
        
        results = util.classify_image(image_data)
        
        if not results:
            response = jsonify({'error': 'No faces detected in the image'})
            response.headers.add('Access-Control-Allow-Origin', '*')
            return response, 400
        
        response = jsonify({
            'success': True,
            'results': results,
            'faces_detected': len(results)
        })
        
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response
        
    except Exception as e:
        print(f"Error in classify_image: {str(e)}")
        print(traceback.format_exc())
        response = jsonify({'error': str(e)})
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response, 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for server status"""
    model_loaded = util.__model is not None
    response = jsonify({
        'status': 'healthy',
        'model_loaded': model_loaded,
        'working_directory': os.getcwd(),
        'artifacts_exists': os.path.exists('./artifacts')
    })
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

@app.route('/', methods=['GET'])
def home():
    """Root endpoint"""
    return jsonify({'message': 'Celebrity Face Recognition API is running!'}), 200

if __name__ == "__main__":
    print("Starting Python Flask Server For Celebrity Image Classification")
    app.run(port=5000, debug=False)  # Set debug=False for production