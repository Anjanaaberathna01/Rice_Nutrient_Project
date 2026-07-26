import requests
import json
import os

# 💡 ඔයාගේ Flask Server එක රන් වෙන IP එක සහ Port එක
BASE_URL = "http://127.0.0.1:5000" 

def test_full_system(image_path, ds_division, gn_division):
    if not os.path.exists(image_path):
        print(f"❌ Error: Image '{image_path}' not found!")
        return

    print("=" * 60)
    print(f"🚀 STARTING END-TO-END TEST FOR: {image_path}")
    print("=" * 60)

    # ---------------------------------------------------------
    # STEP 1: පින්තූරය AI එකට යැවීම (/predict)
    # ---------------------------------------------------------
    print("\n[STEP 1] Sending image to AI for analysis...")
    predict_url = f"{BASE_URL}/predict"
    
    with open(image_path, 'rb') as img_file:
        files = {'image': img_file}
        # නිකමට වගේ Lat, Lng දෙකකුත් යවමු
        data = {'lat': '7.8731', 'lng': '80.7718'} 
        
        try:
            response1 = requests.post(predict_url, files=files, data=data)
            res1_data = response1.json()
        except Exception as e:
            print(f"❌ Backend Connection Error: {e}")
            return

    if res1_data.get('status') != 'success':
        print(f"❌ AI Prediction Failed: {res1_data.get('message')}")
        return

    # AI රිසල්ට් එක පින්ට් කිරීම (Heatmap එකේ Base64 කෝඩ් එක දිග වැඩි නිසා අයින් කරලා පින්ට් කරමු)
    display_res1 = res1_data.copy()
    display_res1.pop('heatmap', None)
    print("✅ AI Prediction Success:")
    print(json.dumps(display_res1, indent=4, ensure_ascii=False))

    # අවශ්‍ය දත්ත ටික වෙන් කරගැනීම
    prediction_id = res1_data.get('prediction_id')
    ai_prediction = res1_data.get('prediction')
    ai_category = res1_data.get('category')

    # ---------------------------------------------------------
    # STEP 2: පොහොර නිර්දේශය ලබාගැනීම (/validate_and_recommend_fertilizer)
    # ---------------------------------------------------------
    print("\n[STEP 2] Validating with Soil Data (Triangulation)...")
    validate_url = f"{BASE_URL}/validate_and_recommend_fertilizer"
    
    payload = {
        "prediction_id": prediction_id,
        "ds_division": ds_division,
        "gn_division": gn_division,
        "prediction": ai_prediction,
        "category": ai_category
    }
    
    headers = {"Content-Type": "application/json"}
    
    try:
        response2 = requests.post(validate_url, json=payload, headers=headers)
        res2_data = response2.json()
    except Exception as e:
        print(f"❌ Validation Error: {e}")
        return

    print("✅ Validation Success:")
    print(json.dumps(res2_data, indent=4, ensure_ascii=False))
    
    print("\n" + "=" * 60)
    print(f"🎯 FINAL RECOMMENDATION: {res2_data.get('recommendation')}")
    print(f"🎨 MAP COLOR WILL BE: {res2_data.get('map_color')}")
    print("=" * 60)

# --- මෙතනින් ටෙස්ට් එක රන් කරන්න ---
# 'anuradhapura' සහ 'srwasthipura' කියන්නේ ඔයාගේ අලුත් CSV එකේ තියෙන දත්ත.
test_full_system('74.JPG', 'anuradhapura', 'srwasthipura')