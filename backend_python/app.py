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


try:
    RICE_VS_NONRICE_MODEL = tf.keras.models.load_model('models/rice_binary_model.h5', compile=False)
    DISEASE_VS_DEFICIENCY_MODEL = tf.keras.models.load_model('models/rice_binary_disease_and_deficiency_v2.h5', compile=False)
    MODEL_MOBILE = tf.keras.models.load_model('models/rice_multilabel_model_v4_final.h5', compile=False)
    MODEL_DENSE = tf.keras.models.load_model('models/rice_multilabel_model_densenet121.h5', compile=False)
    MODEL_INCEPTION = tf.keras.models.load_model('models/rice_multilabel_model_inceptionv3.h5', compile=False)
except Exception as e:
    print("Error loading models:", e)

# Define class names and indices for deficiencies and diseases
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

        # Check if the image is of a rice leaf using the RICE_VS_NONRICE_MODEL

        is_rice_score = float(RICE_VS_NONRICE_MODEL.predict(rice_check_input, verbose=0)[0][0])
        if is_rice_score < 0.85:
            return jsonify({"status":"invalid", "message":"මෙය වී පත්‍රයක් නොවේ.", "score":is_rice_score})
        
        # If it is a rice leaf, proceed to classify it as disease or deficiency

        bin_preds = DISEASE_VS_DEFICIENCY_MODEL.predict(rice_check_input, verbose=0)[0]
        deficiency_prob = float(bin_preds[0])
        disease_prob = float(bin_preds[1])

        # Preprocess the image for each model and get predictions

        in_m = tf.keras.applications.mobilenet_v2.preprocess_input(img_224.copy().reshape(1,224,224,3))
        in_d = tf.keras.applications.densenet.preprocess_input(img_224.copy().reshape(1,224,224,3))
        in_i = tf.keras.applications.inception_v3.preprocess_input(img_224.copy().reshape(1,224,224,3))

        # Get predictions from each model
        p_m = MODEL_MOBILE.predict(in_m, verbose=0)[0]
        p_d = MODEL_DENSE.predict(in_d, verbose=0)[0]
        p_i = MODEL_INCEPTION.predict(in_i, verbose=0)[0]

        # Combine predictions with weights
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

        # Save the prediction to the database
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

@app.route('/validate_and_recommend_fertilizer', methods=['POST'])
def validate_and_recommend_fertilizer():
    try:
        data = request.json
        prediction_id = data.get('prediction_id')
        user_gn = data.get('gn_division', '').strip().lower()
        user_ds = data.get('ds_division', '').strip().lower()
        ai_prediction = data.get('prediction')
        category = data.get('category')

        # is disease, return recommendation directly without checking CSV
        if category == "Disease":
            return jsonify({
                "status": "success",
                "validation_status": "Pending",
                "map_color": "ORANGE",
                "recommendation": f"AI මඟින් පත්‍රයේ පවතින්නේ '{ai_prediction}' රෝගය බව හඳුනාගෙන ඇත. මෙය පසෙහි පෝෂක හිඟයක් නොවන බැවින් පස් වාර්තාවන් මෙයට අදාළ නොවේ. සුදුසු කෘෂිකාර්මික දිලීර/බැක්ටීරියා නාශකයක් භාවිත කරන්න."
            })

        # if deficiency, check the CSV for soil data
        df = pd.read_csv(CSV_PATH)
        df['gn_division'] = df['gn_division'].astype(str).str.strip().str.lower()
        df['ds_division'] = df['ds_division'].astype(str).str.strip().str.lower()
        
        soil_record = df[(df['gn_division'] == user_gn) & (df['ds_division'] == user_ds)]

        if soil_record.empty:
            return jsonify({
                "status": "success", 
                "validation_status": "Pending", 
                "map_color": "ORANGE", 
                "recommendation": f"ඔබේ වසම ({data.get('gn_division')}) සඳහා පස් වාර්තාවක් පද්ධතියේ නොමැත. එබැවින් AI අනාවැකිය තහවුරු කළ නොහැක."
            })

        #get the first matching record
        record = soil_record.iloc[0]
        ph = float(record['ph_value'])
        p_val = float(record['phosphorus_value'])
        k_val = float(record['potassium_value'])
        om_val = float(record['organic_matter_status'])

        validation_status = "REJECTED"
        map_color = "RED"
        recommendation_text = ""

        if ai_prediction == 'Nitrogen_Deficiency':
            if om_val < 2.0:
                validation_status = "VALIDATED"
                map_color = "GREEN"
                recommendation_text = f"[VALIDATED] වසමේ පසෙහි කාබනික ද්‍රව්‍ය ඉතා අඩුයි ({om_val}%). AI අනාවැකිය නිවැරදි බැවින් යූරියා (Urea) පොහොර නිර්දේශ කෙරේ."
            else:
                recommendation_text = f"[REJECTED] වසමේ පසෙහි කාබනික ද්‍රව්‍ය සතුටුදායකයි ({om_val}%). AI අනාවැකිය සහ ඔබගේ ග්‍රාමසේවා වසමට අදාළ පස් වාර්තාවන් පරස්පරතාවයක් ඇත."

        elif ai_prediction == 'Phosphorus_Deficiency':
            if p_val < 10.0 or ph > 7.5:
                validation_status = "VALIDATED"
                map_color = "GREEN"
                recommendation_text = f"[VALIDATED] වසමේ පසෙහි පොස්පරස් අගය {p_val} mg/kg වේ (pH අගය: {ph}). AI අනාවැකිය නිවැරදි බැවින් TSP යෙදීම නිර්දේශ කෙරේ."
            else:
                recommendation_text = f"[REJECTED] වසමේ පසෙහි පොස්පරස් සතුටුදායකයි ({p_val} mg/kg). AI අනාවැකිය සහ ඔබගේ ග්‍රාමසේවා වසමට අදාළ පස් වාර්තාවන් පරස්පරතාවයක් ඇත."

        elif ai_prediction == 'Potassium_Deficiency':
            if k_val < 80.0:
                validation_status = "VALIDATED"
                map_color = "GREEN"
                recommendation_text = f"[VALIDATED] වසමේ පොටෑසියම් මට්ටම ඉතා අඩුයි ({k_val} mg/kg). AI අනාවැකිය නිවැරදි බැවින් MOP පොහොර නිර්දේශ කෙරේ."
            else:
                recommendation_text = f"[REJECTED] වසමේ පොටෑසියම් සතුටුදායකයි ({k_val} mg/kg). AI අනාවැකිය සහ ඔබගේ ග්‍රාමසේවා වසමට අදාළ පස් වාර්තාවන් පරස්පරතාවයක් ඇත."

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
# 💡 CSV File CRUD Operations for React Admin Dashboard
# =========================================================

@app.route('/api/soil', methods=['GET'])
def get_soil_data():
    try:
        df = pd.read_csv(CSV_PATH)
        df['id'] = df.index # React එකට අඳුරගන්න index එක id එක විදිහට යවනවා
        return jsonify(df.to_dict(orient='records'))
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/soil', methods=['POST'])
def add_soil_data():
    try:
        data = request.json
        df = pd.read_csv(CSV_PATH)
        new_row = pd.DataFrame([data])
        # CSV එකේ තියෙන අනිත් තීරුවලට හිස් නම් 0 දානවා
        for col in df.columns:
            if col not in new_row.columns:
                new_row[col] = 0
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(CSV_PATH, index=False)
        return jsonify({"status": "success", "message": "Data added successfully!"})
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
            return jsonify({"status": "success", "message": "Data updated successfully!"})
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
            return jsonify({"status": "success", "message": "Data deleted successfully!"})
        return jsonify({"status": "error", "message": "Row not found"}), 404
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)