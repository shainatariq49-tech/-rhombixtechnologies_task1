from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np
import io
import base64
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import colorsys
from collections import Counter

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Load pre-trained MobileNetV2 model (lightweight and accurate)
model = MobileNetV2(weights='imagenet')

def get_dominant_colors(img, num_colors=5):
    """Extract dominant colors from image"""
    img = img.resize((150, 150))
    img_array = np.array(img)
    pixels = img_array.reshape(-1, 3)
    
    # Remove very dark and very bright pixels
    pixels = pixels[(pixels.sum(axis=1) > 30) & (pixels.sum(axis=1) < 735)]
    
    # Get most common colors
    unique_colors = np.unique(pixels, axis=0, return_counts=True)
    color_counts = list(zip(unique_colors[0], unique_colors[1]))
    color_counts.sort(key=lambda x: x[1], reverse=True)
    
    dominant = []
    for color, count in color_counts[:num_colors]:
        r, g, b = color
        h, s, v = colorsys.rgb_to_hsv(r/255, g/255, b/255)
        
        # Determine color name
        if s < 0.1:
            if v > 0.8:
                color_name = "White"
            elif v < 0.2:
                color_name = "Black"
            else:
                color_name = "Gray"
        elif h < 0.05 or h > 0.95:
            color_name = "Red"
        elif 0.05 <= h < 0.15:
            color_name = "Orange"
        elif 0.15 <= h < 0.22:
            color_name = "Yellow"
        elif 0.22 <= h < 0.42:
            color_name = "Green"
        elif 0.42 <= h < 0.58:
            color_name = "Cyan"
        elif 0.58 <= h < 0.75:
            color_name = "Blue"
        else:
            color_name = "Purple"
        
        if color_name not in dominant:
            dominant.append(color_name)
    
    return dominant[:3]

def analyze_brightness(img):
    """Analyze image brightness"""
    img_array = np.array(img.convert('L'))
    avg_brightness = np.mean(img_array)
    
    if avg_brightness > 180:
        return "Bright"
    elif avg_brightness > 100:
        return "Well-lit"
    else:
        return "Dark"

def detect_scene_type(predictions):
    """Determine scene type from predictions"""
    outdoor_keywords = ['beach', 'mountain', 'sky', 'ocean', 'tree', 'forest', 'park', 'street']
    indoor_keywords = ['room', 'kitchen', 'office', 'furniture', 'table', 'chair']
    
    pred_text = ' '.join([pred[1].lower() for pred in predictions])
    
    if any(keyword in pred_text for keyword in outdoor_keywords):
        return "outdoor"
    elif any(keyword in pred_text for keyword in indoor_keywords):
        return "indoor"
    else:
        return "general"

@app.route('/analyze', methods=['POST'])
def analyze_image():
    try:
        # Get image from request
        data = request.json
        image_data = data.get('image')
        
        if not image_data:
            return jsonify({'error': 'No image provided'}), 400
        
        # Decode base64 image
        image_data = image_data.split(',')[1] if ',' in image_data else image_data
        img_bytes = base64.b64decode(image_data)
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        
        # Prepare image for model
        img_resized = img.resize((224, 224))
        img_array = image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        
        # Make prediction
        predictions = model.predict(img_array)
        decoded_predictions = decode_predictions(predictions, top=5)[0]
        
        # Extract main object and confidence
        main_pred = decoded_predictions[0]
        main_object = main_pred[1].replace('_', ' ').title()
        confidence = int(main_pred[2] * 100)
        
        # Get all detected objects
        objects = [pred[1].replace('_', ' ').title() for pred in decoded_predictions[:4]]
        
        # Determine category
        categories = {
            'animal': ['dog', 'cat', 'bird', 'elephant', 'lion', 'tiger', 'bear'],
            'vehicle': ['car', 'truck', 'bus', 'airplane', 'motorcycle', 'bicycle'],
            'food': ['pizza', 'hamburger', 'coffee', 'ice_cream', 'fruit', 'vegetable'],
            'furniture': ['chair', 'table', 'couch', 'bed', 'desk'],
            'electronics': ['laptop', 'phone', 'computer', 'television', 'camera'],
            'nature': ['tree', 'flower', 'mountain', 'ocean', 'beach', 'forest']
        }
        
        category = 'General'
        main_lower = main_pred[1].lower()
        for cat, keywords in categories.items():
            if any(keyword in main_lower for keyword in keywords):
                category = cat.title()
                break
        
        # Get colors
        colors = get_dominant_colors(img)
        
        # Get scene information
        scene_type = detect_scene_type(decoded_predictions)
        brightness = analyze_brightness(img)
        
        scene = f"A {brightness.lower()} {scene_type} scene featuring {main_object.lower()}"
        
        # Generate attributes
        attributes = []
        if confidence > 80:
            attributes.append("High confidence")
        if brightness == "Bright":
            attributes.append("Well illuminated")
        elif brightness == "Dark":
            attributes.append("Low light")
        
        # Add size info
        width, height = img.size
        if width > 1920 or height > 1080:
            attributes.append("High resolution")
        
        attributes.append(f"{scene_type.title()} environment")
        
        # Generate description
        description = f"This image contains {main_object.lower()} as the primary subject. "
        description += f"The image is captured in {brightness.lower()} lighting conditions in an {scene_type} setting. "
        description += f"The dominant colors are {', '.join(colors[:2]).lower()}. "
        description += f"Additional objects detected include {', '.join(objects[1:3]).lower()}."
        
        # Prepare response
        result = {
            'mainObject': main_object,
            'category': category,
            'confidence': confidence,
            'objects': objects,
            'colors': colors,
            'scene': scene,
            'attributes': attributes,
            'description': description
        }
        
        return jsonify(result)
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'model': 'MobileNetV2'})

if __name__ == '__main__':
    print("Starting Image Recognition ML Backend...")
    print("Loading MobileNetV2 model...")
    app.run(debug=True, host='0.0.0.0', port=5000)