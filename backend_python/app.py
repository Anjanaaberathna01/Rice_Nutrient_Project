import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from rembg import remove
from PIL import Image
import tensorflow as tf
import cv2
import numpy as np
import io
import base64
import mysql.connector

app = Flask(__name__)
CORS(app)

CSV_PATH = "database/soil_reference.csv"

def get_db_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="rice_project_db"
    )

def save_to_db(prediction, category, confidence, dis_score, def_score, lat, lng):
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        sql = """INSERT INTO leaf_analysis 
                 (prediction, category, confidence, disease_score, deficiency_score, latitude, longitude, status, marker_color) 
                 VALUES (%s, %s, %s, %s, %s, %s, %s, 'Pending', 'ORANGE')"""
        values = (prediction, category, confidence, dis_score, def_score, lat, lng)
        cursor.execute(sql, values)
        conn.commit()
        return cursor.lastrowid
    except Exception as e:
        print("Database Insert Error :", e)
        return None
    finally:
        if cursor: cursor.close()
        if conn: conn.close()

# --- AI Models Load (ඔයාගේ models folder path එක හරියට තියෙන්න ඕනේ) ---
try:
    RICE_VS_NONRICE_MODEL = tf.keras.models.load_model('models/rice_binary_model.h5', compile=False)
    DISEASE_VS_DEFICIENCY_MODEL = tf.keras.models.load_model('models/rice_binary_disease_and_deficiency_v2.h5', compile=False)
    MODEL_MOBILE = tf.keras.models.load_model('models/rice_multilabel_model_v4_final.h5', compile=False)
    MODEL_DENSE = tf.keras.models.load_model('models/rice_multilabel_model_densenet121.h5', compile=False)
    MODEL_INCEPTION = tf.keras.models.load_model('models/rice_multilabel_model_inceptionv3.h5', compile=False)
except Exception as e:
    print("Error loading models:", e)

CLASS_NAMES = ['Bacterial_Leaf_Blight', 'Brown_Spot', 'Healthy', 'Leaf_Blast', 'Leaf_Scald', 'Nitrogen_Deficiency', 'Phosphorus_Deficiency', 'Potassium_Deficiency', 'Sheath_Blight']
DEFICIENCY_INDICES = [2, 5, 6, 7]
DISEASE_INDICES = [0, 1, 3, 4, 8]

def generate_heatmap_base64(img_np, model_dense):
    try:
        img_resized = cv2.resize(img_np, (224,224))
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
    except Exception as e:
        print("Heatmap Error:", e)
        return None

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"status": "error", "message": "Image not found"}), 400
    file = request.files['image']
    img_bytes = file.read()
    lat = request.form.get('lat')
    lng = request.form.get('lng')

    try:
        clean_raw = remove(img_bytes)
        img_pil = Image.open(io.BytesIO(clean_raw)).convert("RGB")
        img_np = np.array(img_pil)
        img_224 = cv2.resize(img_np, (224,224))
        rice_check_input = np.expand_dims(img_224 / 255.0, axis=0)

        is_rice_score = float(RICE_VS_NONRICE_MODEL.predict(rice_check_input, verbose=0)[0][0])
        if is_rice_score < 0.85:
            return jsonify({"status":"invalid", "message":"මෙය වී පත්‍රයක් නොවේ.", "score":is_rice_score})

        bin_preds = DISEASE_VS_DEFICIENCY_MODEL.predict(rice_check_input, verbose=0)[0]
        deficiency_prob = float(bin_preds[0])
        disease_prob = float(bin_preds[1])

        in_m = tf.keras.applications.mobilenet_v2.preprocess_input(img_224.copy().reshape(1,224,224,3))
        in_d = tf.keras.applications.densenet.preprocess_input(img_224.copy().reshape(1,224,224,3))
        in_i = tf.keras.applications.inception_v3.preprocess_input(img_224.copy().reshape(1,224,224,3))

        p_m = MODEL_MOBILE.predict(in_m, verbose=0)[0]
        p_d = MODEL_DENSE.predict(in_d, verbose=0)[0]
        p_i = MODEL_INCEPTION.predict(in_i, verbose=0)[0]

        final_preds = (p_m * 0.3) + (p_d * 0.4) + (p_i * 0.3)
        is_disease = disease_prob > deficiency_prob
        category_label = "Disease" if is_disease else "Deficiency"

        if is_disease:
            for idx in DEFICIENCY_INDICES: final_preds[idx] = 0
        else:
            for idx in DISEASE_INDICES: final_preds[idx] = 0

        best_idx = np.argmax(final_preds)
        predicted_class = CLASS_NAMES[best_idx]
        confidence = float(final_preds[best_idx])

        prediction_id = save_to_db(predicted_class, category_label, confidence, disease_prob, deficiency_prob, lat, lng)

        return jsonify({
            "status":"success",
            "prediction":predicted_class,
            "category":category_label,
            "prediction_id":prediction_id,
            "confidence":confidence,
            "disease_score":disease_prob,
            "deficiency_score":deficiency_prob,
            "heatmap":generate_heatmap_base64(img_np, MODEL_DENSE),
            "message": "AI විශ්ලේෂණය සාර්ථකයි."
        })
    except Exception as e:
        return jsonify({"status":"error", "message":str(e)}), 500

@app.route('/get_latest_location', methods=['GET'])
def get_latest_location():
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM leaf_analysis WHERE latitude IS NOT NULL AND longitude IS NOT NULL ORDER BY id DESC")
        rows = cursor.fetchall()
        for row in rows:
            if row['marker_color'] is None: row['marker_color']="ORANGE"
        cursor.close()
        conn.close()
        return jsonify(rows)
    except Exception as e: return jsonify({"status":"error", "message":str(e)}), 500

@app.route('/validate_and_recommend_fertilizer', methods=['POST'])
def validate_and_recommend_fertilizer():
    try:
        data = request.json
        prediction_id = data.get('prediction_id')
        user_gn = data.get('gn_division', '').strip().lower()
        user_ds = data.get('ds_division', '').strip().lower()
        ai_prediction = data.get('prediction')
        category = data.get('category')

        if category == "Disease":
            return jsonify({
                "status": "success",
                "validation_status": "Pending",
                "map_color": "ORANGE",
                "recommendation": "මෙය රෝගී තත්ත්වයකි. සුදුසු දිලීර නාශකයක් භාවිත කරන්න."
            })

        df = pd.read_csv(CSV_PATH)
        df['gn_division'] = df['gn_division'].astype(str).str.strip().str.lower()
        df['ds_division'] = df['ds_division'].astype(str).str.strip().str.lower()
        soil_record = df[(df['gn_division'] == user_gn) & (df['ds_division'] == user_ds)]

        if soil_record.empty:
            return jsonify({"status": "success", "validation_status": "Pending", "map_color": "ORANGE", "recommendation": "පස් වාර්තා නොමැත."})

        record = soil_record.iloc[0]
        # (මෙතන ඔයාගේ කලින් තිබුණු logic එකම දාන්න)

        return jsonify({"status": "success", "validation_status": "VALIDATED", "map_color": "GREEN", "recommendation": "සාර්ථකයි."})
    except Exception as e:
        return jsonify({"status":"error", "message":str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)