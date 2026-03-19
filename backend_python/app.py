from flask import Flask, request, jsonify
from flask_cors import CORS
from rembg import remove
from PIL import Image
import tensorflow as tf
import cv2
import numpy as np
import io
import os

app = Flask(__name__)
CORS(app)

# 1. Models Load කිරීම (Path නිවැරදිදැයි පරීක්ෂා කරන්න)
BINARY_MODEL = tf.keras.models.load_model('models/rice_binary_model.h5', compile=False)
MULTILABEL_MODEL = tf.keras.models.load_model('models/rice_multilabel_model.h5', compile=False)

CLASS_NAMES = ['Healthy', 'Nitrogen_Deficiency', 'Phosphorus_Deficiency', 'Potassium_Deficiency']

SOLUTIONS = {
    'Healthy': 'ඔබේ වී ශාකය නිරෝගී ය. දැනට පවතින වගා පිළිවෙත් දිගටම කරගෙන යන්න.',
    'Nitrogen_Deficiency': 'යූරියා පොහොර (46% N) යොදන්න. පෝෂක අවශෝෂණය වැඩි දියුණු කිරීම සඳහා නිසි ජල කළමනාකරණය සහතික කරන්න.',
    'Phosphorus_Deficiency': 'ත්‍රිත්ව සුපිරි පොස්පේට් (TSP) යොදන්න. මෙය මුල් වර්ධනයට සහ මුල් අවධියේ වර්ධනයට උපකාරී වේ.',
    'Potassium_Deficiency': 'මියුරියේට් ඔෆ් පොටෑෂ් (MOP) යොදන්න. එය රෝග ප්‍රතිරෝධය සහ ධාන්‍ය ගුණාත්මකභාවය වැඩි දියුණු කරයි.'
}

def get_aspect_ratio(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None: return 0.0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return 0.0
    cnt = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(cnt)
    (x, y), (w, h), angle = rect
    if w == 0 or h == 0: return 0.0
    return max(w, h) / min(w, h)

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
        clean_img_bytes = remove_background(img_bytes)
        ratio = get_aspect_ratio(clean_img_bytes)
        
        # 1. Geometric Filter (Shape Check)
        if ratio < 2.5: # 3.0 ට වඩා මඳක් අඩු කළා සාර්ථකත්වය වැඩි කිරීමට
            return jsonify({
                'status': 'invalid',
                'message': f'වැරදි හැඩයකි (Ratio: {ratio:.2f}). පත්‍රය සිරස් අතට තබා නැවත උත්සාහ කරන්න.'
            })

        img_data = preprocess_image(clean_img_bytes)
        
        # 2. Binary CNN Filter (Rice vs Non-Rice)
        binary_input = img_data / 255.0
        binary_input = np.expand_dims(binary_input, axis=0)
        leaf_pred = BINARY_MODEL.predict(binary_input, verbose=0)
        score = float(leaf_pred[0][0])

        # --- Three-Tier Logic Implementation ---
        if score > 0.85:
            # සාර්ථකයි - NPK Analysis
            img_hsv = cv2.cvtColor(img_data, cv2.COLOR_RGB2HSV)
            hsv_input = img_hsv.astype('float32') / 255.0
            hsv_input = np.expand_dims(hsv_input, axis=0)
            
            nutrient_pred = MULTILABEL_MODEL.predict(hsv_input, verbose=0)
            idx = np.argmax(nutrient_pred)
            predicted_class = CLASS_NAMES[idx]
            
            return jsonify({
                'status': 'success',
                'prediction': predicted_class,
                'confidence': float(nutrient_pred[0][idx]),
                'solution': SOLUTIONS[predicted_class],
                'lat': lat, 'lng': lng,
                'message': 'වී පත්‍රය සාර්ථකව හඳුනා ගන්නා ලදී.'
            })

        elif 0.70 <= score <= 0.85:
            # අපැහැදිලියි (Unclear)
            return jsonify({
                'status': 'unclear',
                'message': f'පින්තූරය අපැහැදිලියි (Score: {score:.2f}). කරුණාකර වඩාත් පැහැදිලි පින්තූරයක් ගන්න.'
            })

        else:
            # ප්‍රතික්ෂේපිතයි (Invalid)
            return jsonify({
                'status': 'invalid',
                'message': f'මෙය වී පත්‍රයක් නොවේ (Score: {score:.2f}). කරුණාකර නිවැරදි පින්තූරයක් ලබා දෙන්න.'
            })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)