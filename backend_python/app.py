import pandas as pd # data manipulation
from flask import Flask, request, jsonify # web framework
from flask_cors import CORS # cross-origin resource sharing
from rembg import remove # image background removal
from PIL import Image # image processing
import tensorflow as tf # machine learning
import cv2 # image processing
import numpy as np # numerical operations
import io
import base64
import mysql.connector # database connection
import requests # weather api call සඳහා
from datetime import datetime, timedelta

app = Flask(__name__)
CORS(app)

# - soil reference CSV file path
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

# Load the pre-trained models 
try:
    RICE_VS_NONRICE_MODEL = tf.keras.models.load_model('models/rice_binary_model.h5', compile=False)
    DISEASE_VS_DEFICIENCY_MODEL = tf.keras.models.load_model('models/rice_binary_disease_and_deficiency.h5', compile=False)
    MODEL_EFFICIENT = tf.keras.models.load_model('models/rice_multilabel_model_efficientnetb0.h5', compile=False)
    MODEL_DENSE = tf.keras.models.load_model('models/rice_multilabel_model_densenet121.h5', compile=False)
    MODEL_INCEPTION = tf.keras.models.load_model('models/rice_multilabel_model_inceptionv3.h5', compile=False)
except Exception as e:
    print("Error loading models:", e)

# Define class names and indices for deficiencies and diseases
CLASS_NAMES = ['Bacterial_Leaf_Blight', 'Brown_Spot', 'Healthy', 'Leaf_Blast', 'Leaf_Scald', 'Nitrogen_Deficiency', 'Phosphorus_Deficiency', 'Potassium_Deficiency', 'Sheath_Blight']
DEFICIENCY_INDICES = [2, 5, 6, 7]
DISEASE_INDICES = [0, 1, 3, 4, 8]

# Function to generate heatmap and return it as a base64 string (DenseNet model)
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

# =========================================================
# 💡 OpenWeatherMap API Integration Function
# =========================================================
def get_realtime_weather(lat, lng):
    try:
        api_key = "YOUR_OPENWEATHERMAP_API_KEY"
        # 💡 අපි දැන් පාවිච්චි කරන්නේ 5-day/3-hour Forecast API එක (වඩාත් නිවැරදි දත්ත සඳහා)
        url = f"http://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lng}&appid={api_key}&units=metric"
        
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            list_data = data.get('list', [])
            
            # පසුගිය දින 5ක දත්ත (API එකෙන් දෙන උපරිම දත්ත ප්‍රමාණය) එකතු කිරීම
            total_rainfall = 0.0
            dry_spell_count = 0
            
            for entry in list_data:
                # වර්ෂාපතනය (mm)
                rain = entry.get('rain', {}).get('3h', 0.0)
                total_rainfall += rain
                
                # ආර්ද්‍රතාවය අඩු නම් වියළි දින ලෙස ගණන් කිරීම
                if entry['main']['humidity'] < 45:
                    dry_spell_count += 1
            
            # දින 5ක සාමාන්‍යය පදනම් කරගෙන දින 7ක දත්ත ප්‍රක්ෂේපණය (Projection) කිරීම
            recent_rainfall_7d = total_rainfall * 1.4 
            is_drought = True if dry_spell_count > 15 else False # දින 5ක දත්ත ලැයිස්තුවේ වැඩි ප්‍රමාණයක් වියළි නම්
            
            return recent_rainfall_7d, is_drought
            
        return 0.0, False
    except Exception as e:
        print("Weather API Error:", e)
        return 0.0, False


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

        in_m = tf.keras.applications.efficientnet.preprocess_input(img_224.copy().reshape(1,224,224,3))
        in_d = tf.keras.applications.densenet.preprocess_input(img_224.copy().reshape(1,224,224,3))
        in_i = tf.keras.applications.inception_v3.preprocess_input(img_224.copy().reshape(1,224,224,3))

        p_eff = MODEL_EFFICIENT.predict(in_m, verbose=0)[0]
        p_d = MODEL_DENSE.predict(in_d, verbose=0)[0]
        p_i = MODEL_INCEPTION.predict(in_i, verbose=0)[0]

        final_preds = (p_eff * 0.2) + (p_d * 0.4) + (p_i * 0.4)
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
            "message": "Prediction completed successfully."
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

# =========================================================
# Get GN and DS division from CSV for mobile app dropdowns
# =========================================================
@app.route('/api/get_divisions', methods=['GET'])
def get_divisions():
    try:
        df = pd.read_csv(CSV_PATH)
        
        df['ds_division'] = df['ds_division'].astype(str).str.strip().str.lower()
        df['gn_division'] = df['gn_division'].astype(str).str.strip().str.lower()
        
        # DS divisions list
        ds_list = df['ds_division'].unique().tolist()
        
        # in list of GN in DS division
        gn_dict = {}
        for ds in ds_list:
            gns = df[df['ds_division'] == ds]['gn_division'].unique().tolist()
            gn_dict[ds] = gns
            
        return jsonify({
            "status": "success",
            "ds_divisions": ds_list,
            "gn_divisions": gn_dict
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
    
# =========================================================
# Decision Support System (DSS) - Soil & Weather Validation
# =========================================================

@app.route('/validate_and_recommend_fertilizer', methods=['POST'])
def validate_and_recommend_fertilizer():
    try:
        data = request.json
        prediction_id = data.get('prediction_id')
        user_gn = data.get('gn_division', '').strip().lower()
        user_ds = data.get('ds_division', '').strip().lower()
        ai_prediction = data.get('prediction') # 'Nitrogen_Deficiency'
        category = data.get('category')        # 'Disease' or 'Deficiency'
        lat = data.get('lat')                  # GPS Latitude
        lng = data.get('lng')                  # GPS Longitude

        
        if category == "Disease":
            return jsonify({
                "status": "success",
                "validation_status": "Pending",
                "map_color": "ORANGE",
                "recommendation": f"AI මඟින් පත්‍රයේ පවතින්නේ '{ai_prediction}' රෝගය බව හඳුනාගෙන ඇත. මෙය පසෙහි පෝෂක හිඟයක් නොව දිලීර/බැක්ටීරියා රෝගයක් වන බැවින් පස් වාර්තා සහ කාලගුණ දත්ත මෙයට අදාළ නොවේ. සුදුසු කෘෂිකාර්මික නාශකයක් භාවිත කරන්න."
            })

        # 2. is it deficiency:- check soil reference CSV for the GN and DS division
        df_soil = pd.read_csv(CSV_PATH)
        df_soil['gn_division'] = df_soil['gn_division'].astype(str).str.strip().str.lower()
        df_soil['ds_division'] = df_soil['ds_division'].astype(str).str.strip().str.lower()
        
        soil_record = df_soil[(df_soil['gn_division'] == user_gn) & (df_soil['ds_division'] == user_ds)]

        # Not in soil data is it in pending
        if soil_record.empty:
            return jsonify({
                "status": "success", 
                "validation_status": "Pending", 
                "map_color": "ORANGE", 
                "recommendation": f"ඔබේ වසම ({data.get('gn_division')}) සඳහා පස් වාර්තාවක් පද්ධතියේ නොමැත. එබැවින් AI අනාවැකිය තහවුරු කළ නොහැක."
            })

        # get soil data from the first matching record
        record = soil_record.iloc[0]
        ph = float(record['ph_value'])
        p_val = float(record['phosphorus_value'])
        k_val = float(record['potassium_value'])
        om_val = float(record['organic_matter_status']) 

        # 3. (Real-time Weather via API)
        if lat and lng:
            recent_rainfall_7d, is_drought = get_realtime_weather(float(lat), float(lng))
        else:
            recent_rainfall_7d = 0.0
            is_drought = False

        validation_status = "REJECTED"
        map_color = "RED"
        recommendation_text = ""

        # =========================================================
        # 1.  (Nitrogen Deficiency)
        # =========================================================
        if ai_prediction == 'Nitrogen_Deficiency':
            if om_val < 2.0 and recent_rainfall_7d > 150.0:
                validation_status = "VALIDATED"
                map_color = "GREEN"
                recommendation_text = f"වසමේ පසෙහි කාබනික ද්‍රව්‍ය මට්ටම අඩුයි ({om_val}%) සහ පසුගිය දිනවල ලැබුණු අධික වර්ෂාපතනය (mm {recent_rainfall_7d}) නිසා පෝෂක සේදී යාමකට (Leaching) ලක්වී ඇත. මේ නිසා මෙම ඌනතාවය තදින්ම මතුවී තිබිය හැක."
            elif om_val < 2.0:
                validation_status = "VALIDATED"
                map_color = "GREEN"
                recommendation_text = f"වසමේ පසෙහි කාබනික ද්‍රව්‍ය මට්ටම අඩුයි ({om_val}%). පසෙහි පෝෂක ඌනතාවයක් පවතින බැවින් AI අනාවැකිය විද්‍යාත්මකව තහවුරු වේ."
            elif om_val >= 2.0 and recent_rainfall_7d > 150.0:
                validation_status = "VALIDATED"
                map_color = "GREEN"
                recommendation_text = f"පසෙහි සාමාන්‍ය නයිට්‍රජන් මට්ටමක් තිබුණද, පසුගිය දිනවල ලැබුණු අධික වර්ෂාපතනය (mm {recent_rainfall_7d}) හේතුවෙන් පෝෂක සේදී යාමකට (Leaching) ලක්වී ඇත. මේ නිසා මෙම ඌනතාවය මතුවී තිබිය හැක."
            else:
                validation_status = "REJECTED"
                map_color = "RED"
                recommendation_text = f"වසමේ පසෙහි කාබනික ද්‍රව්‍ය තත්ත්වය සතුටුදායකය ({om_val}%). කාලගුණික බලපෑමක්ද නොමැත. AI අනාවැකිය පරස්පර වේ."

        # =========================================================
        # 2.  (Phosphorus Deficiency) 
        # =========================================================
        elif ai_prediction == 'Phosphorus_Deficiency':
            if p_val < 10.0 and is_drought:
                validation_status = "VALIDATED"
                map_color = "GREEN"
                recommendation_text = f"වසමේ පසෙහි පොස්පරස් මට්ටම අඩුයි ({p_val} mg/kg) සහ වර්තමාන වියළි කාලගුණය/නියඟය නිසා තෙතමනය සිඳී ගොස් මුල් මඟින් පෝෂක උරාගැනීම ඇනහිට ඇත."
            elif p_val < 10.0 or ph > 7.5:
                validation_status = "VALIDATED"
                map_color = "GREEN"
                recommendation_text = f"වසමේ පසෙහි පොස්පරස් මට්ටම අඩුයි හෝ පසෙහි pH අගය වැඩි මට්ටමක පවතී ({p_val} mg/kg). AI අනාවැකිය තහවුරු වේ."
            elif p_val >= 10.0 and is_drought:
                validation_status = "VALIDATED"
                map_color = "GREEN"
                recommendation_text = f"පසෙහි පොස්පරස් තිබුණද, පවතින වියළි කාලගුණය/නියඟය නිසා තෙතමනය සිඳී ගොස් මුල් මඟින් පෝෂක උරාගැනීම ඇනහිට ඇත. මේ නිසා මෙම ඌනතාවය මතුවී තිබිය හැක."
            else:
                validation_status = "REJECTED"
                map_color = "RED"
                recommendation_text = f"වසමේ පසෙහි පොස්පරස් මට්ටම සතුටුදායකය ({p_val} mg/kg)."

        # =========================================================
        # 3.  (Potassium Deficiency) 
        # =========================================================
        elif ai_prediction == 'Potassium_Deficiency':
            if k_val < 175.0 and recent_rainfall_7d > 150.0:
                validation_status = "VALIDATED"
                map_color = "GREEN"
                recommendation_text = f"වසමේ පොටෑසියම් මට්ටම අඩුයි ({k_val} mg/kg) සහ අධික වැසි නිසා වැඩිදුරටත් සේදී ගොස් ඇත."
            elif k_val < 175.0 or recent_rainfall_7d > 150.0:
                validation_status = "VALIDATED"
                map_color = "GREEN"
                recommendation_text = f"වසමේ පොටෑසියම් අගය අඩුයි ({k_val} mg/kg) හෝ අධික වැසි නිසා සේදී ගොස් ඇත. අනාවැකිය තහවුරු වේ."
            elif k_val >= 175.0 and is_drought:
                validation_status = "VALIDATED"
                map_color = "GREEN"
                recommendation_text = f"වසමේ පොටෑසියම් තිබුණද, පවතින වියළි කාලගුණය/නියඟය නිසා තෙතමනය සිඳී ගොස් මුල් මඟින් පෝෂක උරාගැනීම ඇනහිට ඇත."
            else:
                validation_status = "REJECTED"
                map_color = "RED"
                recommendation_text = f"වසමේ පොටෑසියම් මට්ටම සතුටුදායකය ({k_val} mg/kg)."

        # Database Update  (Status and  Map Color )
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
            "status": "success", 
            "validation_status": validation_status, 
            "map_color": map_color, 
            "recommendation": recommendation_text
        })

    except Exception as e:
        print("Validation Error:", e)
        return jsonify({"status":"error", "message":str(e)}), 500
    

# =========================================================
#  Soil Reference CSV CRUD (For Admin Dashboard)
# =========================================================

@app.route('/api/soil', methods=['GET'])
def get_soil_data():
    try:
        df = pd.read_csv(CSV_PATH)
        df['id'] = df.index 
        return jsonify(df.to_dict(orient='records'))
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/soil', methods=['POST'])
def add_soil_data():
    try:
        data = request.json
        df = pd.read_csv(CSV_PATH)
        new_row = pd.DataFrame([data])
        for col in df.columns:
            if col not in new_row.columns:
                new_row[col] = 0
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(CSV_PATH, index=False)
        return jsonify({"status": "success", "message": "Soil record added successfully!"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/soil/<int:row_id>', methods=['PUT'])
def update_soil_data(row_id):
    try:
        data = request.json
        df = pd.read_csv(CSV_PATH)
        if 0 <= row_id < len(df):
            for key, value in data.items():
                if key in df.columns:
                    df.at[row_id, key] = value
            df.to_csv(CSV_PATH, index=False)
            return jsonify({"status": "success", "message": "Soil record updated successfully!"})
        return jsonify({"status": "error", "message": "Row not found"}), 404
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/soil/<int:row_id>', methods=['DELETE'])
def delete_soil_data(row_id):
    try:
        df = pd.read_csv(CSV_PATH)
        if 0 <= row_id < len(df):
            df = df.drop(row_id).reset_index(drop=True)
            df.to_csv(CSV_PATH, index=False)
            return jsonify({"status": "success", "message": "Soil record deleted successfully!"})
        return jsonify({"status": "error", "message": "Row not found"}), 404
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)