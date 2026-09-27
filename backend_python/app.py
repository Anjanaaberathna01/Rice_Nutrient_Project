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
#  OpenWeatherMap API Integration Function
# =========================================================
def get_realtime_weather(lat, lng):
    try:
        api_key = "e1d3fecabdd809b3bd85c53a5ffeea7d"
        url = f"http://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lng}&appid={api_key}&units=metric"
        
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            list_data = data.get('list', [])
            
            total_rainfall = 0.0
            dry_spell_count = 0

            # Get the most recent weather entry for temperature, humidity, and wind speed
            current_entry = list_data[0] if list_data else {}
            temperature = current_entry.get('main', {}).get('temp', 28.0)
            relative_humidity = current_entry.get('main', {}).get('humidity', 75.0)
            wind_speed = current_entry.get('wind', {}).get('speed', 0.0) * 3.6  
            

        # Calculate the average temperature during the night
            night_temperatures = []

            for entry in list_data:
                try:
                    dt_txt = entry.get('dt_txt', '')
                    hour = int(dt_txt.split(' ')[1].split(':')[0])
                    temp = entry.get('main', {}).get('temp')

                    if temp is not None and (hour >= 18 or hour <= 6):
                        night_temperatures.append(float(temp))
                except (ValueError, IndexError, TypeError):
                    continue

            # Loop එකෙන් පිටතදී සාමාන්‍ය අගය ගණනය කිරීම (Bug 1 Fixed)
            if night_temperatures:
                night_temperature = sum(night_temperatures) / len(night_temperatures)
            else:
                night_temperature = None


        # Calculate total rainfall and dry spell count over the forecast period

            for entry in list_data:
                rain = entry.get('rain', {}).get('3h', 0.0)
                total_rainfall += rain
                if entry.get('main', {}).get('humidity', 50) < 45:
                    dry_spell_count += 1

            recent_rainfall_7d = total_rainfall * 1.4 
            is_drought = True if dry_spell_count > 15 else False 
            
            return recent_rainfall_7d, is_drought, temperature, relative_humidity, wind_speed, night_temperature
            
        return 0.0, False, 28.0, 75.0, 0.0, None
    except Exception as e:
        print("Weather API Error:", e)
        return 0.0, False, 28.0, 75.0, 0.0, None


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
        ai_prediction = data.get('prediction') # example 'Nitrogen_Deficiency'
        category = data.get('category')        # 'Disease' or 'Deficiency'
        lat = data.get('lat')                  # GPS Latitude
        lng = data.get('lng')                  # GPS Longitude

        # error handling for missing lat/lng
        if not lat or not lng:
            raise ValueError("GPS coordinates (latitude and longitude) are missing. Location is required for weather triangulation.")

        # To get soil data from CSV based on GN and DS division
        df_soil = pd.read_csv(CSV_PATH)
        df_soil['gn_division'] = df_soil['gn_division'].astype(str).str.strip().str.lower()
        df_soil['ds_division'] = df_soil['ds_division'].astype(str).str.strip().str.lower()
        
        soil_record = df_soil[(df_soil['gn_division'] == user_gn) & (df_soil['ds_division'] == user_ds)]

        # is soil data are empty
        if soil_record.empty:
            return jsonify({
                "status": "success", 
                "validation_status": "Pending", 
                "map_color": "ORANGE", 
                "recommendation": f"ඔබේ වසම ({data.get('gn_division')}) සඳහා පස් වාර්තාවක් පද්ධතියේ නොමැත. එබැවින් AI අනාවැකිය තහවුරු කළ නොහැක."
            })

        record = soil_record.iloc[0]
        ph = float(record['ph_value'])
        p_val = float(record['phosphorus_value'])
        k_val = float(record['potassium_value'])
        om_val = float(record['organic_matter_status'])
        ec_val = float(record['ec_value'])

        # To get weather data
        recent_rainfall_7d, is_drought, temperature, relative_humidity, wind_speed, night_temperature = get_realtime_weather(float(lat), float(lng))

        nitrogen_status = "HIGH" if om_val < 2.0 else "NORMAL"

        validation_status = "REJECTED"
        map_color = "RED"
        recommendation_text = ""

        # =========================================================
        # 1. BACTERIAL LEAF BLIGHT (Disease Triangulation)
        # =========================================================
        if ai_prediction == 'Bacterial_Leaf_Blight':
            favourable_conditions = 0
            supporting_factors = []

            if 25.0 <= temperature <= 34.0:
                favourable_conditions += 1
                supporting_factors.append(f"උෂ්ණත්වය 25–34°C අතර ({temperature:.1f}°C)")

            if relative_humidity >= 80.0:
                favourable_conditions += 1
                supporting_factors.append(f"අධික සාපේක්ෂ ආර්ද්‍රතාව ({relative_humidity:.1f}%)")

            if recent_rainfall_7d > 100.0:
                favourable_conditions += 1
                supporting_factors.append(f"පසුගිය දින 7 තුළ වර්ෂාපතනය ({recent_rainfall_7d:.1f} mm)")

            if wind_speed >= 20.0:
                favourable_conditions += 1
                supporting_factors.append(f"ප්‍රබල සුළං තත්ත්වය ({wind_speed:.1f} km/h)")

            if nitrogen_status == "HIGH":
                favourable_conditions += 1
                supporting_factors.append("අධික නයිට්‍රජන් / යූරියා භාවිතය")

            if favourable_conditions >= 3:
                validation_status = "VALIDATED"
                map_color = "GREEN"
                factors = ", ".join(supporting_factors)
                recommendation_text = (
                    f"AI මඟින් 'බැක්ටීරියා කොළ අංගමාරය' (Bacterial Leaf Blight) හඳුනාගෙන ඇත. "
                    f"DOA විසින් සඳහන් කර ඇති රෝගයට හිතකර තත්ත්වයන්ගෙන් කිහිපයක් පවතී. "
                    f"එබැවින් AI අනාවැකිය තහවුරු වේ. DOA උපදෙස්: ක්ෂේත්‍රයේ ජල බැසයාම තහවුරු කරන්න, "
                    f"ජල සැපයුම නවත්වා කුඹුර වියළීමට ඉඩ දෙන්න, යූරියා භාවිතය සීමා කර පොටෑසියම් යොදන්න."
                )
            elif favourable_conditions >= 1:
                validation_status = "PARTIALLY_VALIDATED"
                map_color = "ORANGE"
                factors = ", ".join(supporting_factors)
                recommendation_text = (
                    f"AI මඟින් 'බැක්ටීරියා කොළ අංගමාරය' හඳුනාගෙන ඇත. හිතකර තත්ත්වයන් සමහරක් පවතී. "
                    f"මෙය අර්ධ වශයෙන් තහවුරු වූ අනාවැකියකි. ජල කළමනාකරණය සහ නයිට්‍රජන් භාවිතය සැලකිලිමත් වන්න."
                )
            else:
                validation_status = "LOW_CONFIDENCE"
                map_color = "RED"
                recommendation_text = (
                    f"AI මඟින් 'බැක්ටීරියා කොළ අංගමාරය' පෙන්වුවද, පාරිසරික තත්ත්වයන් රෝගයට හිතකර නොවේ. "
                    f"මෙය අඩු විශ්වාසනීය අනාවැකියකි. පත්‍ර තවදුරටත් නිරීක්ෂණය කරන්න."
                )

        # =========================================================
        # RICE BLAST DISEASE
        # =========================================================
        elif ai_prediction == 'Leaf_Blast':

            evidence_score = 0
            supporting_factors = []


            # =====================================================
            # 1. NIGHT TEMPERATURE
            # DOA favourable condition: 17-20°C
            # =====================================================

            if night_temperature is not None:

                if 17.0 <= night_temperature <= 20.0:

                    evidence_score += 2

                    supporting_factors.append(
                        f"රාත්‍රී උෂ්ණත්වය Rice Blast සඳහා "
                        f"හිතකර 17–20°C පරාසයේ පවතී "
                        f"({night_temperature:.1f}°C)"
                    )

                else:

                    supporting_factors.append(
                        f"රාත්‍රී උෂ්ණත්වය 17–20°C "
                        f"හිතකර පරාසයෙන් පිටත පවතී "
                        f"({night_temperature:.1f}°C)"
                    )

            else:

                supporting_factors.append(
                    "රාත්‍රී උෂ්ණත්ව දත්ත ලබාගත නොහැකි විය"
                )


            # =====================================================
            # 2. HIGH RELATIVE HUMIDITY
            # System operational threshold = 80%
            # =====================================================

            if relative_humidity >= 80.0:

                evidence_score += 1

                supporting_factors.append(
                    f"සාපේක්ෂ ආර්ද්‍රතාවය ඉහළ මට්ටමක පවතී "
                    f"({relative_humidity:.1f}%)"
                )

            else:

                supporting_factors.append(
                    f"සාපේක්ෂ ආර්ද්‍රතාවය "
                    f"{relative_humidity:.1f}% කි"
                )


            # =====================================================
            # 3. RAINFALL
            # Context ONLY
            # =====================================================

            if recent_rainfall_7d > 0:

                supporting_factors.append(
                    f"පසුගිය forecast කාලය තුළ "
                    f"වර්ෂාපතන තත්ත්වයක් පවතී "
                    f"({recent_rainfall_7d:.1f} mm)"
                )


            # =====================================================
            # 4. NITROGEN FERTILIZER
            # IMPORTANT:
            # NOT USED FOR VALIDATION SCORE
            #
            # We only warn the farmer about N fertilizer usage.
            # =====================================================

            nitrogen_warning = (
                "Rice Blast සඳහා නයිට්‍රජන් / යූරියා පොහොර "
                "අධික ලෙස යෙදීමෙන් වළකින්න. "
                "යෙදූ නයිට්‍රජන් පොහොර ප්‍රමාණය "
                "DOA නිර්දේශිත මාත්‍රාව සමඟ සසඳා බලන්න."
            )


            # =====================================================
            # 5. FINAL VALIDATION
            # Maximum score = 3
            #
            # 3 = VALIDATED
            # 1-2 = PARTIALLY_VALIDATED
            # 0 = REJECTED
            # =====================================================

            if evidence_score >= 3:

                validation_status = "VALIDATED"
                map_color = "GREEN"

                recommendation_text = (

                    f"AI මඟින් Rice Blast රෝගය හඳුනාගෙන ඇත. "

                    f"පවතින පාරිසරික දත්ත අනුව රෝගයට "
                    f"හිතකර තත්ත්වයන් පවතී: "

                    f"{', '.join(supporting_factors)}. "

                    f"එබැවින් AI අනාවැකියට ශක්තිමත් "
                    f"පාරිසරික සහායක් පවතී. "

                    f"{nitrogen_warning}"
                )


            elif evidence_score >= 1:

                validation_status = "PARTIALLY_VALIDATED"
                map_color = "ORANGE"

                recommendation_text = (

                    f"AI මඟින් Rice Blast රෝගය හඳුනාගෙන ඇත. "

                    f"නමුත් පවතින පාරිසරික දත්ත මඟින් "
                    f"සීමිත supporting evidence පමණක් පවතී: "

                    f"{', '.join(supporting_factors)}. "

                    f"එබැවින් අනාවැකිය සම්පූර්ණයෙන් "
                    f"තහවුරු කළ නොහැක. "

                    f"{nitrogen_warning}"
                )


            else:

                validation_status = "REJECTED"
                map_color = "RED"

                recommendation_text = (

                    f"AI මඟින් Rice Blast රෝගය හඳුනාගෙන ඇත. "

                    f"නමුත් පවතින පාරිසරික දත්ත මඟින් "
                    f"Rice Blast සඳහා ප්‍රමාණවත් "
                    f"supporting evidence හමු වී නොමැත. "

                    f"{', '.join(supporting_factors)}. "

                    f"එබැවින් රෝග ලක්ෂණ නැවත පරීක්ෂා කරන්න. "

                    f"{nitrogen_warning}"
                )

        # =========================================================
        # 5. BROWN SPOT DISEASE
        # =========================================================
        elif ai_prediction == 'Brown_Spot':

            evidence_score = 0
            supporting_factors = []

            # Temperature
            if 16.0 <= temperature <= 36.0:

                evidence_score += 1

                supporting_factors.append(
                    f"පවතින උෂ්ණත්වය Brown Spot සඳහා හිතකර පරාසයේ පවතී "
                    f"({temperature:.1f}°C)"
                )

            # Humidity (ඔබගේ පද්ධතියේ 'relative_humidity' ලෙස භාවිතා වේ)
            if 86.0 <= relative_humidity <= 100.0:

                evidence_score += 1

                supporting_factors.append(
                    f"සාපේක්ෂ ආර්ද්‍රතාවය ඉහළ මට්ටමක පවතී "
                    f"({relative_humidity:.1f}%)"
                )

            # Drought (ඔබගේ පද්ධතියේ 'is_drought' ලෙස භාවිතා වේ)
            if is_drought:

                evidence_score += 1

                supporting_factors.append(
                    "වියළි / නියං තත්ත්වයක් පවතී"
                )

            # ---------------------------------------------------------
            # 2. SECONDARY SOIL EVIDENCE
            # ---------------------------------------------------------
            # DOA states that Brown Spot is favoured by soils with
            # low levels of required nutrients or problem soils,
            # including high salinity and iron toxicity.
            #
            # Therefore soil information is used as supporting
            # evidence rather than the primary disease criterion.
            # ---------------------------------------------------------

            soil_stress_factors = []

            # Low organic matter as an indicator of poor soil quality (ඔබගේ පද්ධතියේ 'om_val' ලෙස භාවිතා වේ)
            if om_val < 1.0:

                soil_stress_factors.append(
                    f"පසේ කාබනික ද්‍රව්‍ය මට්ටම අඩුය ({om_val:.1f}%)"
                )

            # High EC / salinity condition
            # (මතක තබාගන්න: මෙය ක්‍රියාත්මක වීමට ඔබේ CSV එකෙන් 'ec_val' ලබාගන්නා කේතයක් මුලින් තිබිය යුතුය)
            if ec_val > 0.4:

                soil_stress_factors.append(
                    f"පසේ EC අගය ඉහළය ({ec_val:.1f} dS/m)"
                )

            if soil_stress_factors:

                evidence_score += 1

                supporting_factors.extend(soil_stress_factors)

            # ---------------------------------------------------------
            # 3. FINAL VALIDATION
            # ---------------------------------------------------------

            if evidence_score >= 3:

                validation_status = "VALIDATED"
                map_color = "GREEN"

                recommendation_text = (
                    f"AI මඟින් Brown Spot රෝගය හඳුනාගෙන ඇත. "
                    f"පවතින පාරිසරික හා පාංශු දත්ත මඟින් අනාවැකියට "
                    f"ශක්තිමත් සහායක් ලැබේ: "
                    f"{', '.join(supporting_factors)}. "
                    f"DOA නිර්දේශ අනුව නිසි පෝෂණ කළමනාකරණය, "
                    f"වල් පැළෑටි පාලනය සහ සුදුසු වගා කළමනාකරණ ක්‍රම "
                    f"අනුගමනය කරන්න."
                )

            elif evidence_score >= 1:

                validation_status = "PARTIALLY_VALIDATED"
                map_color = "ORANGE"

                recommendation_text = (
                    f"AI මඟින් Brown Spot රෝගය හඳුනාගෙන ඇත. "
                    f"කෙසේ වෙතත්, පවතින පාරිසරික හා පාංශු දත්ත මඟින් "
                    f"සීමිත supporting evidence පමණක් පවතී: "
                    f"{', '.join(supporting_factors)}. "
                    f"එබැවින් අනාවැකිය සම්පූර්ණයෙන් තහවුරු කළ නොහැක. "
                    f"වගාවේ රෝග ලක්ෂණ සහ පවතින පාරිසරික තත්ත්වයන් "
                    f"නැවත පරීක්ෂා කරන්න."
                )

            else:

                validation_status = "REJECTED"
                map_color = "RED"

                recommendation_text = (
                    f"AI මඟින් Brown Spot රෝගය හඳුනාගෙන ඇත. "
                    f"නමුත් පවතින පාරිසරික හා පාංශු දත්ත මඟින් "
                    f"Brown Spot සඳහා ප්‍රමාණවත් supporting evidence "
                    f"හමු වී නොමැත. "
                    f"එබැවින් රෝග ලක්ෂණ සහ පාරිසරික තත්ත්වයන් "
                    f"නැවත පරීක්ෂා කිරීම නිර්දේශ කෙරේ."
                )

        # =========================================================
        # 6. SHEATH BLIGHT DISEASE
        # =========================================================
        
        elif ai_prediction == 'Sheath_Blight':

            evidence_score = 0
            supporting_factors = []

            if 30.0 <=temperature <= 40.0:

                evidence_score += 1

                supporting_factors.append(
                    f"උෂ්ණත්වය ඉහළ මට්ටමක පවතී ({temperature:.1f}°C)"
                )

            if relative_humidity > 90.0:

                evidence_score += 1

                supporting_factors.append(
                    f"සාපේක්ෂ ආර්ද්‍රතාවය ඉහළ මට්ටමක පවතී "
                    f"({relative_humidity:.1f}%)"
                )

            # ---------------------------------------------------------
            # 2. RAINFALL - CONTEXT ONLY
            # ---------------------------------------------------------
            # Rainfall is displayed as environmental context.
            # It is NOT given a validation score because the DOA
            # Sheath Blight reference does not specify a rainfall
            # threshold.
            # ---------------------------------------------------------

            if recent_rainfall_7d > 0.0:

                supporting_factors.append(
                    f"පසුගිය දින 7 තුළ සැලකිය යුතු වර්ෂාපතනයක් ලැබී ඇත "
                    f"({recent_rainfall_7d:.1f} mm)"
                )

            # ---------------------------------------------------------
            # 3. DROUGHT
            # ---------------------------------------------------------
            # Drought is not listed by DOA as a favourable condition
            # for Sheath Blight.
            # Therefore, no evidence score is added.
            # ---------------------------------------------------------

            # ---------------------------------------------------------
            # 4. FINAL VALIDATION
            # ---------------------------------------------------------

            if evidence_score >= 2:

                validation_status = "VALIDATED"
                map_color = "GREEN"

                recommendation_text = (
                    f"AI මඟින් Sheath Blight රෝගය හඳුනාගෙන ඇත. "
                    f"පවතින පාරිසරික දත්ත මඟින් අනාවැකියට "
                    f"ශක්තිමත් සහායක් ලැබේ: "
                    f"{', '.join(supporting_factors)}. "
                    f"DOA නිර්දේශ අනුව නයිට්‍රජන් පොහොර නිර්දේශිත "
                    f"මාත්‍රාවට අනුව පමණක් යොදන්න. "
                    f"වල් පැළෑටි පාලනය සහ සුදුසු වගා ඝනත්වයක් "
                    f"පවත්වා ගැනීම නිර්දේශ කෙරේ."
                )

            elif evidence_score == 1:

                validation_status = "PARTIALLY_VALIDATED"
                map_color = "ORANGE"

                recommendation_text = (
                    f"AI මඟින් Sheath Blight රෝගය හඳුනාගෙන ඇත. "
                    f"නමුත් පවතින පාරිසරික දත්ත මඟින් සීමිත "
                    f"supporting evidence පමණක් පවතී: "
                    f"{', '.join(supporting_factors)}. "
                    f"එබැවින් අනාවැකිය සම්පූර්ණයෙන් තහවුරු කළ නොහැක. "
                    f"නයිට්‍රජන් පොහොර යෙදූ ප්‍රමාණය "
                    f"සලකා බලන්න."
                )

            else:

                validation_status = "REJECTED"
                map_color = "RED"

                recommendation_text = (
                    f"AI මඟින් Sheath Blight රෝගය හඳුනාගෙන ඇත. "
                    f"නමුත් පවතින පාරිසරික දත්ත මඟින් Sheath Blight සඳහා "
                    f"ප්‍රමාණවත් supporting evidence හමු වී නොමැත. "
                    f"එබැවින් රෝග ලක්ෂණ සහ වගාවේ පාරිසරික තත්ත්වයන් "
                    f"නැවත පරීක්ෂා කිරීම නිර්දේශ කෙරේ."
                    f"නයිට්‍රජන් පොහොර යෙදූ ප්‍රමාණය"
                    f"සලකා බලන්න."
                )
            
        # =========================================================
        # 2. NITROGEN DEFICIENCY
        # =========================================================
        elif ai_prediction == 'Nitrogen_Deficiency':
            evidence_score = 0
            supporting_factors = []

            # 1. Primary: Organic matter status
            if om_val < 2.0:
                evidence_score += 2
                supporting_factors.append(f"කාබනික ද්‍රව්‍ය මට්ටම අඩුය ({om_val:.1f}%)")

            # 2. Secondary: Excessive rainfall
            if recent_rainfall_7d > 150.0:
                evidence_score += 1
                supporting_factors.append(f"පසුගිය දින 7 තුළ අධික වර්ෂාපතනයක් ලැබී ඇත ({recent_rainfall_7d:.1f} mm)")

            # DECISION & MAPPING
            if evidence_score >= 2:
                validation_status = "VALIDATED"
                map_color = "GREEN"
                recommendation_text = (
                f"AI මඟින් නයිට්‍රජන් ඌනතාවය හඳුනාගෙන ඇත. "
                f"පවතින පාංශු හා පාරිසරික දත්ත AI අනාවැකියට ශක්තිමත් සහායක් ලබා දෙයි: "
                f"{', '.join(supporting_factors)}. "
                f"එබැවින් නයිට්‍රජන් ඌනතාවය සඳහා ශක්තිමත් supporting evidence පවතී. "
                f"නයිට්‍රජන් පොහොර යෙදීමේදී DOA නිර්දේශිත මාත්‍රාව "
                f"සහ/හෝ Paddy Leaf Colour Chart (LCC) මත පදනම් වූ කළමනාකරණ ක්‍රම අනුගමනය කරන්න."
                )

            elif evidence_score == 1:
                validation_status = "PARTIALLY_VALIDATED"
                map_color = "ORANGE"
                recommendation_text = (
                f"AI මඟින් නයිට්‍රජන් ඌනතාවය හඳුනාගෙන ඇත. "
                f"කෙසේ වෙතත්, පවතින දත්ත මඟින් සීමිත supporting evidence පමණක් පවතී: "
                f"{', '.join(supporting_factors)}. "
                f"එබැවින් නයිට්‍රජන් ඌනතාවය සම්පූර්ණයෙන් තහවුරු කළ නොහැක. "
                f"පසේ තත්ත්වය සහ LCC අගය සලකා බලා නයිට්‍රජන් පොහොර යෙදීම තීරණය කරන්න."
                )
            else:
                validation_status = "REJECTED"
                map_color = "RED"
                recommendation_text = (
                f"AI මඟින් නයිට්‍රජන් ඌනතාවය හඳුනාගෙන ඇත. "
                f"නමුත් පවතින පාංශු හා පාරිසරික දත්ත මඟින් AI අනාවැකියට "
                f"ප්‍රමාණවත් supporting evidence හමු වී නොමැත. "
                f"පසේ කාබනික ද්‍රව්‍ය මට්ටම {om_val:.1f}% ලෙස පවතී. "
                f"එබැවින් නයිට්‍රජන් පොහොර යෙදීමට පෙර පසේ තත්ත්වය සහ LCC අගය "
                f"නැවත පරීක්ෂා කිරීම නිර්දේශ කෙරේ."
                )

        # =========================================================
        # 3. PHOSPHORUS DEFICIENCY
        # =========================================================
        elif ai_prediction == 'Phosphorus_Deficiency':

            evidence_score = 0
            supporting_factors = []
    
            if p_val < 5.0:

                evidence_score += 2

                supporting_factors.append(
                    f"පසේ පොස්පරස් මට්ටම අඩුය ({p_val:.1f} mg/kg)"
                )

            elif p_val <= 10.0:

                evidence_score += 0

                supporting_factors.append(
                    f"පසේ පොස්පරස් මට්ටම මධ්‍යම පරාසයේ පවතී "
                    f"({p_val:.1f} mg/kg)"
                )

            else:

                evidence_score -= 2

                supporting_factors.append(
                    f"පසේ පොස්පරස් මට්ටම ඉහළය ({p_val:.1f} mg/kg)"
                )

            # ---------------------------------------------------------
            # 2. SECONDARY EVIDENCE: Soil pH & Drought
            # ---------------------------------------------------------
            # These are environmental/contextual factors.
            # They do NOT independently confirm P deficiency.
            # ---------------------------------------------------------

            # පසේ ප්‍රශස්ත pH පරාසය 5.5 - 6.5 වේ[cite: 3]
            if ph < 5.5 or ph > 6.5:

                supporting_factors.append(
                    f"පසේ pH අගය ප්‍රශස්ත සීමාවෙන් පිට පවතී ({ph:.1f})"
                )

            if is_drought:

                supporting_factors.append(
                    "වියළි / නියං තත්ත්වයක් පවතී"
                )

            # ---------------------------------------------------------
            # 3. FINAL VALIDATION
            # ---------------------------------------------------------

            if p_val < 5.0:

                validation_status = "VALIDATED"
                map_color = "GREEN"

                recommendation_text = (
                    f"AI මඟින් පොස්පරස් ඌනතාවය හඳුනාගෙන ඇත. "
                    f"පාංශු දත්ත මඟින් අනාවැකියට ශක්තිමත් සහායක් ලැබේ. "
                    f"පසේ පොස්පරස් මට්ටම {p_val:.1f} mg/kg වන අතර "
                    f"එය DOA critical values අනුව අඩු මට්ටමකි. "
                    f"{', '.join(supporting_factors)}. "
                    f"පාංශු පරීක්ෂණ ප්‍රතිඵල අනුව DOA නිර්දේශිත "
                    f"පොස්පරස් පොහොර කළමනාකරණය අනුගමනය කරන්න."
                )

            elif p_val <= 10.0:

                validation_status = "PARTIALLY_VALIDATED"
                map_color = "ORANGE"

                recommendation_text = (
                    f"AI මඟින් පොස්පරස් ඌනතාවය හඳුනාගෙන ඇත. "
                    f"නමුත් පසේ පොස්පරස් මට්ටම {p_val:.1f} mg/kg වන අතර "
                    f"එය DOA critical values අනුව මධ්‍යම පරාසයේ පවතී. "
                    f"එබැවින් පාංශු දත්ත මඟින් පොස්පරස් ඌනතාවය "
                    f"සම්පූර්ණයෙන් තහවුරු නොවේ. "
                    f"{', '.join(supporting_factors)}. "
                    f"අනෙකුත් පාරිසරික සාධක සලකා බලන්න."
                )

            else:

                validation_status = "REJECTED"
                map_color = "RED"

                recommendation_text = (
                    f"AI මඟින් පොස්පරස් ඌනතාවය හඳුනාගෙන ඇත. "
                    f"නමුත් පසේ පොස්පරස් මට්ටම {p_val:.1f} mg/kg වන අතර "
                    f"එය DOA critical values අනුව ඉහළ මට්ටමකි. "
                    f"එබැවින් පවතින පාංශු දත්ත AI අනාවැකියට "
                    f"සහාය නොදක්වයි. "
                    f"පොස්පරස් පොහොර යෙදීමට පෙර පාංශු තත්ත්වය "
                    f"නැවත සලකා බලන්න."
                )
       # =========================================================
        # 4. POTASSIUM DEFICIENCY
        # =========================================================
        elif ai_prediction == 'Potassium_Deficiency':
            evidence_score = 0
            supporting_factors = []

            # DOA Critical Values:
            if k_val < 40.0:
                # < 40 mg/kg = Low
                evidence_score += 2
                supporting_factors.append(f"පසේ පොටෑසියම් මට්ටම ඉතා අඩුය ({k_val:.1f} mg/kg)")
            
            elif k_val <= 80.0:
                # 40 - 80 mg/kg = Medium
                evidence_score += 1
                supporting_factors.append(f"පසේ පොටෑසියම් මට්ටම මධ්‍යස්ථ/අවදානම් මට්ටමකය ({k_val:.1f} mg/kg)")
            
            else:
                # > 80 mg/kg = High[cite: 3]
                evidence_score += 0 # ලකුණු එකතු නොවේ, මන්ද ඌනතාවයක් නොමැති බැවිනි

            # ---------------------------------------------------------
            # 2. SECONDARY EVIDENCE: Heavy Rainfall & Drought
            # ---------------------------------------------------------
            if recent_rainfall_7d > 150.0:
                evidence_score += 1
                supporting_factors.append(f"පසුගිය දින 7 තුළ අධික වර්ෂාපතනයක් ලැබී ඇත ({recent_rainfall_7d:.1f} mm)")

            if is_drought:
                evidence_score += 1
                supporting_factors.append("වියළි / නියං තත්ත්වයක් පවතී")

            # ---------------------------------------------------------
            # 3. FINAL VALIDATION (Based on total evidence_score)
            # ---------------------------------------------------------
            if evidence_score >= 2:
                validation_status = "VALIDATED"
                map_color = "GREEN"
                recommendation_text = (
                    f"AI මඟින් පොටෑසියම් ඌනතාවය හඳුනාගෙන ඇත. "
                    f"පාංශු හා පාරිසරික දත්ත මඟින් අනාවැකියට ශක්තිමත් සහායක් ලැබේ: {', '.join(supporting_factors)}. "
                    f"පාංශු පරීක්ෂණ ප්‍රතිඵල අනුව DOA නිර්දේශිත පොටෑසියම් (MOP) පොහොර මාත්‍රාව අනුගමනය කරන්න."
                )
            
            elif evidence_score == 1:
                validation_status = "PARTIALLY_VALIDATED"
                map_color = "ORANGE"
                recommendation_text = (
                    f"AI මඟින් පොටෑසියම් ඌනතාවය හඳුනාගෙන ඇත. "
                    f"Supporting evidence: {', '.join(supporting_factors)}. "
                    f"අනාවැකියට අර්ධ වශයෙන් සහායක් පවතී. පසේ පොටෑසියම් මට්ටම සහ අනෙකුත් පාරිසරික සාධක සලකා බලන්න."
                )
            
            else:
                validation_status = "REJECTED"
                map_color = "RED"
                recommendation_text = (
                    f"AI මඟින් පොටෑසියම් ඌනතාවය හඳුනාගෙන ඇත. "
                    f"නමුත් පසේ පොටෑසියම් මට්ටම ඉහළ මට්ටමක පවතින බැවින් ({k_val:.1f} mg/kg) පවතින පාංශු දත්ත AI අනාවැකියට සහාය නොදක්වයි. "
                    f"පොටෑසියම් පොහොර යෙදීමට පෙර පාංශු තත්ත්වය නැවත සලකා බලන්න."
                )
                
        # Database Update (Status and Map Color)
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