import requests
import json
import os

# Base URL of the backend server
BASE_URL = "http://127.0.0.1:5000" 

def test_full_system(image_path, ds_division, gn_division):
    if not os.path.exists(image_path):
        print(f"Error: Image '{image_path}' not found!")
        return

    print("=" * 60)
    print(f"STARTING END-TO-END TEST FOR: {image_path}")
    print("=" * 60)

    # ---------------------------------------------------------
    # STEP 1: Image prediction using the AI model (/predict)
    # ---------------------------------------------------------
    print("\n[STEP 1] Sending image to AI for analysis...")
    predict_url = f"{BASE_URL}/predict"
    
    with open(image_path, 'rb') as img_file:
        files = {'image': img_file}
        data = {'lat': '8.3114', 'lng': '80.4037'} 
        
        try:
            response1 = requests.post(predict_url, files=files, data=data)
            res1_data = response1.json()
        except Exception as e:
            print(f"Backend Connection Error: {e}")
            return

    if res1_data.get('status') == 'invalid':
        print(f" Validation Failed: {res1_data.get('message')} (Score: {res1_data.get('score')})")
        return

    if res1_data.get('status') != 'success':
        print(f" AI Prediction Failed: {res1_data.get('message')}")
        return

    display_res1 = res1_data.copy()
    display_res1.pop('heatmap', None)
    print("AI Prediction Success:")
    print(json.dumps(display_res1, indent=4, ensure_ascii=False))

    prediction_id = res1_data.get('prediction_id')
    ai_prediction = res1_data.get('prediction')
    ai_category = res1_data.get('category')

    # ---------------------------------------------------------
    # STEP 2: Validate with Soil Data and Weather (/validate_and_recommend_fertilizer)
    # ---------------------------------------------------------
    print("\n[STEP 2] Validating with Soil Data & Real-time Weather (Triangulation)...")
    validate_url = f"{BASE_URL}/validate_and_recommend_fertilizer"
    
    payload = {
        "prediction_id": prediction_id,
        "ds_division": ds_division,
        "gn_division": gn_division,
        "prediction": ai_prediction,
        "category": ai_category,
        "lat": '8.3114',
        "lng": '80.4037'
    }
    
    headers = {"Content-Type": "application/json"}
    
    try:
        response2 = requests.post(validate_url, json=payload, headers=headers)
        res2_data = response2.json()
    except Exception as e:
        print(f"Validation Error: {e}")
        return

    print("Validation Success:")
    print(json.dumps(res2_data, indent=4, ensure_ascii=False))
    
    print("\n" + "=" * 60)
    print(f"FINAL RECOMMENDATION: {res2_data.get('recommendation')}")
    print(f"VALIDATION STATUS: {res2_data.get('validation_status')}")
    print(f"MAP COLOR WILL BE: {res2_data.get('map_color')}")
    print("=" * 60)

if __name__ == '__main__':
    test_full_system('7.jpg', 'galenbindunuwewa', 'kumbukwewa')