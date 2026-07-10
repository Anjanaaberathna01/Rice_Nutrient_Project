import pandas as pd
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
import mysql.connector

app = Flask(__name__)
CORS(app)

CSV_PATH = "database/soil_reference.csv" # 💡 ඔබේ CSV එක මෙතන තියෙන්න ඕනේ bn

def get_db_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="", 
        database="rice_project_db"
    )

def save_to_db(prediction, category, confidence, dis_score, def_score, lat, lng):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        sql = """INSERT INTO leaf_analysis 
                 (prediction, category, confidence, disease_score, deficiency_score, latitude, longitude, status, marker_color) 
                 VALUES (%s, %s, %s, %s, %s, %s, %s, 'Pending', 'ORANGE')"""
        val = (prediction, category, confidence, dis_score, def_score, lat, lng)
        cursor.execute(sql, val)
        conn.commit()
        inserted_id = cursor.lastrowid # 🔥 Flutter එකට ඕනෙම කරන ID එක ගත්තා bn
        cursor.close()
        conn.close()
        return inserted_id
    except Exception as e:
        print(f"❌ Database Error: {e}")
        return None

# --- AI Models Load ---
RICE_VS_NONRICE_MODEL = tf.keras.models.load_model('models/rice_binary_model.h5', compile=False)
DISEASE_VS_DEFICIENCY_MODEL = tf.keras.models.load_model('models/rice_binary_disease_and_deficiency_v2.h5', compile=False)
MODEL_MOBILE = tf.keras.models.load_model('models/rice_multilabel_model_v4_final.h5', compile=False)
MODEL_DENSE = tf.keras.models.load_model('models/rice_multilabel_model_densenet121.h5', compile=False)
MODEL_INCEPTION = tf.keras.models.load_model('models/rice_multilabel_model_inceptionv3.h5', compile=False)

CLASS_NAMES = ['Bacterial_Leaf_Blight', 'Brown_Spot', 'Healthy', 'Leaf_Blast', 'Leaf_Scald', 'Nitrogen_Deficiency', 'Phosphorus_Deficiency', 'Potassium_Deficiency', 'Sheath_Blight']
DEFICIENCY_INDICES = [2, 5, 6, 7]
DISEASE_INDICES = [0, 1, 3, 4, 8]

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
    lat = request.form.get('lat')
    lng = request.form.get('lng')
    
    try:
        clean_raw = remove(img_bytes)
        img_pil = Image.open(io.BytesIO(clean_raw)).convert("RGB")
        img_np = np.array(img_pil)
        img_224 = cv2.resize(img_np, (224, 224))
        rice_check_input = np.expand_dims(img_224 / 255.0, axis=0)
        
        is_rice_score = float(RICE_VS_NONRICE_MODEL.predict(rice_check_input, verbose=0)[0][0])
        if is_rice_score < 0.85:
            return jsonify({'status': 'invalid', 'message': 'මෙය වී පත්‍රයක් නොවේ.', 'score': is_rice_score})

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
        category_label = 'Disease' if is_disease else 'Deficiency'

        if is_disease:
            for idx in DEFICIENCY_INDICES: final_preds[idx] = 0
        else:
            for idx in DISEASE_INDICES: final_preds[idx] = 0

        uncertainty_threshold = 0.15
        diff = abs(disease_prob - deficiency_prob)
        if diff < uncertainty_threshold:
            res_msg = f"පද්ධතිය රෝග ({disease_prob:.0%}) සහ ඌණතා ({deficiency_prob:.0%}) එකවර හඳුනා ගනී."
        else:
            res_msg = f"පද්ධතිය වැඩි නැඹුරුවක් {'රෝග' if is_disease else 'ඌණතා'} කෙරෙහි දක්වයි."

        best_idx = np.argmax(final_preds)
        predicted_class = CLASS_NAMES[best_idx]
        heatmap_b64 = generate_heatmap_base64(img_np, MODEL_DENSE)

        # 🔥 සේව් කරලා හැදෙන ID එක ගන්නවා bn
        new_id = save_to_db(predicted_class, category_label, float(final_preds[best_idx]), disease_prob, deficiency_prob, lat, lng)

        return jsonify({
            'status': 'success', 'prediction': predicted_class, 'category': category_label,
            'prediction_id': new_id, 'disease_score': disease_prob, 'deficiency_score': deficiency_prob,
            'confidence': float(final_preds[best_idx]), 'heatmap': heatmap_b64, 'message': res_msg
        })
    except Exception as e: return jsonify({'error': str(e)}), 500

@app.route('/get_latest_location', methods=['GET'])
def get_latest_location():
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM leaf_analysis ORDER BY id DESC")
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        return jsonify(rows) # සියලුම හිස්ට්‍රි එක ලස්සනට මැප් එකට යවනවා bn
    except Exception as e: return jsonify({'error': str(e)}), 500

# 🔥 3. වැදගත්ම ත්‍රිත්වකරණ රවුට් එක (Triangulation Logic)
@app.route('/validate_and_recommend_fertilizer', methods=['POST'])
def validate_and_recommend_fertilizer():
    try:
        data = request.json
        prediction_id = data.get('prediction_id')
        user_gn = data.get('gn_division').strip().lower()
        ai_prediction = data.get('prediction')
        category = data.get('category')

        # 🛑 රෝගයක් (Disease) නම් පස් වාර්තා බලන්නේ නැතිව කෙටි උපදෙසක් පමණක් දීම
        if category == 'Disease':
            recommendation_text = f"🛑 [රෝග පාලන උපදෙස්] AI මඟින් පත්‍රයේ පවතින්නේ '{ai_prediction}' රෝගය බව හඳුනාගෙන ඇත. මෙය පසෙහි පෝෂක හිඟයක් නොවන බැවින් පස් වාර්තාවන් මෙයට අදාළ නොවේ. සුදුසු කෘෂිකාර්මික දිලීර/බැක්ටීරියා නාශකයක් නියමිත මාත්‍රාවෙන් භාවිත කරන්න."
            return jsonify({
                'status': 'success',
                'validation_status': 'Pending',
                'map_color': 'ORANGE', # රෝග හැමතිස්සෙම Orange තියනවා bn
                'recommendation': recommendation_text
            })

        # 🌾 ඌණතාවයක් (Deficiency) නම් පමණක් CSV එක කියවීම
        df = pd.read_csv(CSV_PATH)
        soil_record = df[df['gn_division'].str.strip().str.lower() == user_gn]

        if soil_record.empty:
            return jsonify({
                'status': 'success',
                'validation_status': 'Pending',
                'map_color': 'ORANGE',
                'recommendation': f"⚠️ ඔබේ ග්‍රාම නිලධාරී වසම ({data.get('gn_division')}) සඳහා පස් වාර්තාවක් පද්ධතියේ නොමැත. එබැවින් AI අනාවැකිය තහවුරු කළ නොහැක. සාමාන්‍ය කෘෂිකාර්මික උපදෙස් පිළිපදින්න."
            })

        record = soil_record.iloc[0]
        ph = float(record['ph_value'])
        p_val = float(record['phosphorus_value'])
        k_val = float(record['potassium_value'])
        om_val = float(record['organic_matter_status'])

        validation_status = "REJECTED"
        map_color = "RED"
        recommendation_text = ""

        # 🧪 Nitrogen_Deficiency Rules
        if ai_prediction == 'Nitrogen_Deficiency':
            if om_val < 2.0:
                validation_status = "VALIDATED"; map_color = "GREEN"
                recommendation_text = f"✅ [පස් වාර්තාවෙන් තහවුරුයි] ඔබේ වසමේ පසෙහි කාබනික ද්‍රව්‍ය ප්‍රතිශතය ඉතා අඩුයි ({om_val}%). AI අනාවැකිය නිවැරදි බැවින් නයිට්‍රජන් සැපයීමට යූරියා (Urea) පොහොර නිර්දේශ කරනු ලැබේ."
            else:
                recommendation_text = f"❌ [පස් වාර්තාවෙන් ප්‍රතික්ෂේපයි] ඔබේ වසමේ පසෙහි කාබනික ද්‍රව්‍ය සතුටුදායකයි ({om_val}%). ශාකයේ නයිට්‍රජන් හිඟයක් තිබිය නොහැක. AI අනාවැකිය වැරදි වීමට ඉඩ ඇත."

        # 🧪 Phosphorus_Deficiency Rules
        elif ai_prediction == 'Phosphorus_Deficiency':
            if p_val < 10.0 or ph > 7.5:
                validation_status = "VALIDATED"; map_color = "GREEN"
                recommendation_text = f"✅ [පස් වාර්තාවෙන් තහවුරුයි] ඔබේ වසමේ පසෙහි පොස්පරස් අගය {p_val} mg/kg වේ (pH அගය: {ph}). AI අනාවැකිය නිවැරදි බැවින් මූලික පොහොර ලෙස TSP යෙදීම නිර්දේශ කෙරේ."
            else:
                recommendation_text = f"❌ [පස් වාර්තාවෙන් ප්‍රතික්ෂේපයි] ඔබේ වසමේ පසෙහි පොස්පරස් සතුටුදායකයි ({p_val} mg/kg). AI අනාවැකිය වැරදි වීමට ඉඩ ඇති බැවින් TSP යෙදීම අනවශ්‍යයි."

        # 🧪 Potassium_Deficiency Rules
        elif ai_prediction == 'Potassium_Deficiency':
            if k_val < 80.0:
                validation_status = "VALIDATED"; map_color = "GREEN"
                recommendation_text = f"✅ [පස් වාර්තාවෙන් තහවුරුයි] ඔබේ වසමේ පොටෑසියම් මට්ටම ඉතා අඩුයි ({k_val} mg/kg). AI අනාවැකිය නිවැරදි බැවින් පත්‍ර වියළීම වැළැක්වීමට MOP පොහොර නිර්දේශ කරනු ලැබේ."
            else:
                recommendation_text = f"❌ [පස් වාර්තාවෙන් ප්‍රතික්ෂේපයි] ඔබේ වසමේ පොටෑසියම් සතුටුදායකයි ({k_val} mg/kg). AI අනාවැකිය වැරදි වීමට ඉඩ ඇති බැවින් MOP යෙදීම අනවශ්‍යයි."

        # 💡 UPDATE Database Marker Colors
        try:
            if prediction_id:
                conn = get_db_connection()
                cursor = conn.cursor()
                query = "UPDATE leaf_analysis SET status = %s, marker_color = %s WHERE id = %s"
                cursor.execute(query, (validation_status, map_color, prediction_id))
                conn.commit()
                cursor.close()
                conn.close()
        except Exception as db_err:
            print(f"Database update error: {db_err}")

        return jsonify({
            'status': 'success',
            'validation_status': validation_status,
            'map_color': map_color,
            'recommendation': recommendation_text
        })
    except Exception as e: return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)