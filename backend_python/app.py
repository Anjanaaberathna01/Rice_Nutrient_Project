from flask import Flask, request, jsonify
from flask_cors import CORS
from rembg import remove
from PIL import Image
import tensorflow as tf
import cv2
import numpy as np
import io
import os
import base64

app = Flask(__name__)
CORS(app)

# 1. Models Load කිරීම
# පද්ධතියේ ඇති මාදිලි (Models) නිවැරදි Path වල ඇති බව තහවුරු කරගන්න
RICE_VS_NONRICE_MODEL = tf.keras.models.load_model('models/rice_binary_model.h5', compile=False)
# නව Softmax Binary Model එක (v2)
DISEASE_VS_DEFICIENCY_MODEL = tf.keras.models.load_model('models/rice_binary_disease_and_deficiency_v2.h5', compile=False)

MODEL_MOBILE = tf.keras.models.load_model('models/rice_multilabel_model_v4_final.h5', compile=False)
MODEL_DENSE = tf.keras.models.load_model('models/rice_multilabel_model_densenet121.h5', compile=False)
MODEL_INCEPTION = tf.keras.models.load_model('models/rice_multilabel_model_inceptionv3.h5', compile=False)

CLASS_NAMES = [
    'Bacterial_Leaf_Blight', 'Brown_Spot', 'Healthy', 'Leaf_Blast', 
    'Leaf_Scald', 'Nitrogen_Deficiency', 'Phosphorus_Deficiency', 
    'Potassium_Deficiency', 'Sheath_Blight'
]

# Indices වෙන් කිරීම
DEFICIENCY_INDICES = [2, 5, 6, 7]
DISEASE_INDICES = [0, 1, 3, 4, 8]

# 2. Helper Functions
def generate_heatmap_base64(img_np, model_dense):
    try:
        img_resized = cv2.resize(img_np, (224, 224))
        img_array = np.expand_dims(tf.keras.applications.densenet.preprocess_input(img_resized.copy()), axis=0)
        
        base_model = model_dense.get_layer('densenet121')
        last_conv_layer = base_model.layers[-1]
        feature_extractor = tf.keras.Model(inputs=base_model.input, outputs=last_conv_layer.output)
        
        features = feature_extractor.predict(img_array, verbose=0)[0]
        heatmap = np.mean(features, axis=-1)
        heatmap = np.maximum(heatmap, 0)
        heatmap /= (np.max(heatmap) + 1e-10)
        
        heatmap_resized = cv2.resize(heatmap, (img_np.shape[1], img_np.shape[0]))
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        superimposed_img = cv2.addWeighted(img_np, 0.6, heatmap_color, 0.4, 0)
        
        _, buffer = cv2.imencode('.jpg', cv2.cvtColor(superimposed_img, cv2.COLOR_RGB2BGR))
        return base64.b64encode(buffer).decode('utf-8')
    except: return None

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files: return jsonify({'error': 'No image'}), 400
    
    file = request.files['image']
    img_bytes = file.read()
    
    try:
        # --- Tier 1: Background Removal & Rice vs Non-Rice Check ---
        clean_raw = remove(img_bytes)
        img_pil = Image.open(io.BytesIO(clean_raw)).convert("RGB")
        img_np = np.array(img_pil)
        
        img_224 = cv2.resize(img_np, (224, 224))
        rice_check_input = np.expand_dims(img_224 / 255.0, axis=0)
        is_rice_score = float(RICE_VS_NONRICE_MODEL.predict(rice_check_input, verbose=0)[0][0])

        if is_rice_score < 0.85:
            return jsonify({'status': 'invalid', 'message': 'මෙය වී පත්‍රයක් නොවේ.', 'score': is_rice_score})

        # --- Tier 2: Disease vs Deficiency Binary Filter (Softmax) ---
        # v2 Softmax මොඩලය [Deficiency_Prob, Disease_Prob] ලෙස අගයන් දෙකක් ලබා දේ 
        bin_preds = DISEASE_VS_DEFICIENCY_MODEL.predict(rice_check_input, verbose=0)[0]
        deficiency_prob = float(bin_preds[0])
        disease_prob = float(bin_preds[1])

        # වැඩිම සම්භාවිතාව ඇති පන්තිය තීරණය කිරීම
        is_disease = disease_prob > deficiency_prob
        
        # ප්‍රතිශතයන් සහිත පණිවිඩය සකස් කිරීම
        res_msg = (f"පද්ධතිය {disease_prob:.0%} ක රෝග ලක්ෂණ සහ {deficiency_prob:.0%} ක ඌණතා ලක්ෂණ හඳුනාගන්නා ලදී. "
                   f"ඒ අනුව වැඩි නැඹුරුවක් {'රෝගී තත්ත්වයන්' if is_disease else 'පෝෂක ඌණතාවයන්'} කෙරෙහි දක්වයි.")

        # --- Tier 3: Ensemble Multi-label Analysis ---
        in_m = tf.keras.applications.mobilenet_v2.preprocess_input(img_224.copy().reshape(1,224,224,3))
        in_d = tf.keras.applications.densenet.preprocess_input(img_224.copy().reshape(1,224,224,3))
        in_i = tf.keras.applications.inception_v3.preprocess_input(img_224.copy().reshape(1,224,224,3))

        p_m = MODEL_MOBILE.predict(in_m, verbose=0)[0]
        p_d = MODEL_DENSE.predict(in_d, verbose=0)[0]
        p_i = MODEL_INCEPTION.predict(in_i, verbose=0)[0]

        # Ensemble Weighted Average
        final_preds = (p_m * 0.3) + (p_d * 0.4) + (p_i * 0.3)

        # Smart Filtering Logic
        if is_disease:
            for idx in DEFICIENCY_INDICES: final_preds[idx] = 0
        else:
            for idx in DISEASE_INDICES: final_preds[idx] = 0

        best_idx = np.argmax(final_preds)
        predicted_class = CLASS_NAMES[best_idx]
        
        # Heatmap එක සෑදීම (DenseNet Model එක ඇසුරෙන්)
        heatmap_b64 = generate_heatmap_base64(img_np, MODEL_DENSE)

        return jsonify({
            'status': 'success',
            'prediction': predicted_class,
            'category': 'Disease' if is_disease else 'Deficiency',
            'disease_score': disease_prob,
            'deficiency_score': deficiency_prob,
            'confidence': float(final_preds[best_idx]),
            'heatmap': heatmap_b64,
            'message': res_msg
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)