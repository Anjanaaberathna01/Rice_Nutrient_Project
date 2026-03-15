from flask import Flask, request, jsonify
from flask_cors import CORS
from rembg import remove
from PIL import Image
import tensorflow as tf
import cv2
import numpy as np
import io

app = Flask(__name__)
CORS(app)

# 1. Models Load කිරීම
BINARY_MODEL = tf.keras.models.load_model('models/rice_binary_model.h5')
MULTILABEL_MODEL = tf.keras.models.load_model('models/rice_multilabel_model.h5')

CLASS_NAMES = ['Healthy', 'Nitrogen_Deficiency', 'Phosphorus_Deficiency', 'Potassium_Deficiency']

SOLUTIONS = {
    'Healthy': 'Your rice plant is healthy. Continue current farming practices.',
    'Nitrogen_Deficiency': 'Apply Urea fertilizer (46% N). Ensure proper water management to improve nutrient uptake.',
    'Phosphorus_Deficiency': 'Apply Triple Super Phosphate (TSP). This will help in root development and early growth.',
    'Potassium_Deficiency': 'Apply Muriate of Potash (MOP). It will improve disease resistance and grain quality.'
}

def get_aspect_ratio(image_bytes):
    """පින්තූරයේ ඇති ප්‍රධාන වස්තුවේ දිග/පළල අනුපාතය ගණනය කරයි."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None: return 0.0
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Background ඉවත් කළ පසු ඉතිරි වන කොටස හඳුනා ගැනීමට
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0

    cnt = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(cnt)
    (x, y), (w, h), angle = rect
    
    if w == 0 or h == 0: return 0.0
    
    # Aspect Ratio = දිගම පැත්ත / කෙටිම පැත්ත
    aspect_ratio = max(w, h) / min(w, h)
    return aspect_ratio

def remove_background(img_bytes):
    output_raw = remove(img_bytes)
    img = Image.open(io.BytesIO(output_raw)).convert("RGB")
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    return img_byte_arr.getvalue()

def preprocess_image(image_bytes, target_size=(224, 224)):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, target_size)
    return img_resized

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    img_bytes = file.read()
    lat = request.form.get('lat', '0.0')
    lng = request.form.get('lng', '0.0')

    try:
        # 1. පසුබිම ඉවත් කිරීම (Rembg භාවිතයෙන්)
        clean_img_bytes = remove_background(img_bytes)
        
        # 2. පින්තූරයේ හැඩය පරීක්ෂා කිරීම (Geometric Filter)
        ratio = get_aspect_ratio(clean_img_bytes)
        print(f"Captured/Cropped Aspect Ratio: {ratio:.2f}")

        # අපි Flutter වලින් කපලා එවන නිසා Ratio එක අනිවාර්යයෙන්ම 3.0 ට වඩා වැඩි විය යුතුයි
        if ratio < 3.0: 
            return jsonify({
                'is_rice': False,
                'prediction': 'None',
                'lat': '0.0', 'lng': '0.0',
                'message': f'Invalid leaf shape (Ratio: {ratio:.2f}). පත්‍රය සිරස් අතට කොටුව තුළ තබා නැවත උත්සාහ කරන්න.'
            })

        img_data = preprocess_image(clean_img_bytes)
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500

    # 3. Binary CNN Filter (ඇත්තටම මේක වී පත්‍රයක්ද?)
    binary_input = img_data / 255.0
    binary_input = np.expand_dims(binary_input, axis=0)
    leaf_pred = BINARY_MODEL.predict(binary_input, verbose=0)
    
    # Confidence එක 0.9 කට වඩා වැඩි නම් වී පත්‍රයක් ලෙස සලකමු
    if leaf_pred[0][0] > 0.9: 
        # 4. Nutrient Analysis (HSV Model)
        img_hsv = cv2.cvtColor(img_data, cv2.COLOR_RGB2HSV)
        hsv_input = img_hsv.astype('float32') / 255.0
        hsv_input = np.expand_dims(hsv_input, axis=0)
        
        nutrient_pred = MULTILABEL_MODEL.predict(hsv_input, verbose=0)
        idx = np.argmax(nutrient_pred)
        predicted_class = CLASS_NAMES[idx]
        
        return jsonify({
            'is_rice': True,
            'prediction': predicted_class,
            'confidence': float(nutrient_pred[0][idx]),
            'solution': SOLUTIONS[predicted_class],
            'lat': lat, 'lng': lng,
            'message': 'Rice leaf verified by shape and CNN.'
        })
    else:
        return jsonify({
            'is_rice': False,
            'prediction': 'None',
            'lat': '0.0', 'lng': '0.0',
            'message': 'Shape verified, but this does not look like a rice leaf (CNN Rejected).'
        })

if __name__ == '__main__':
    # host='0.0.0.0' දැමීමෙන් එකම Network එකේ ඇති පෝන් එකට සම්බන්ධ විය හැක
    app.run(host='0.0.0.0', port=5000, debug=True)