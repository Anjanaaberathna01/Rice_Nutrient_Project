import tensorflow as tf
import cv2
import numpy as np
import os

# 1. පර්යේෂණයේදී පුහුණු කළ Models දෙක Load කිරීම
binary_model_path = 'models/rice_binary_model.h5'
multilabel_model_path = 'models/rice_multilabel_model.h5'

if not os.path.exists(binary_model_path) or not os.path.exists(multilabel_model_path):
    print("Error: Models not found in 'models' folder!")
else:
    binary_model = tf.keras.models.load_model(binary_model_path)
    multilabel_model = tf.keras.models.load_model(multilabel_model_path)
    print("Models loaded successfully.")

# 2. පෝෂක ඌනතා හඳුනාගැනීම සඳහා වූ Labels
class_names = ['Healthy', 'Nitrogen_Deficiency', 'Phosphorus_Deficiency', 'Potassium_Deficiency']

def predict_rice_health(image_path):
    # පින්තූරය කියවා වර්ණ සැකසීම (BGR to RGB)
    img = cv2.imread(image_path)
    if img is None:
        return "Error: Could not read image."
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224)) # 116 පියවරේ සඳහන් size එක
    
    # --- පළමු පියවර: පින්තූරය වී පත්‍රයක්දැයි බැලීම (Binary) ---
    binary_input = img_resized / 255.0
    binary_input = np.expand_dims(binary_input, axis=0)
    
    leaf_pred = binary_model.predict(binary_input, verbose=0)
    
    # පින්තූරය වී පත්‍රයක් නම් (ප්‍රස්තාර අනුව මෙහි නිරවද්‍යතාවය 99.9% කි)
    if leaf_pred[0][0] > 0.5:
        print(f"Leaf Detected: Rice Leaf (Confidence: {leaf_pred[0][0]:.4f})")
        
        # --- දෙවන පියවර: පෝෂක ඌනතාවය හඳුනාගැනීම (Multi-label) ---
        # පර්යේෂණයේ 116 පියවර අනුව පින්තූරය HSV වලට හැරවීම
        img_hsv = cv2.cvtColor(img_resized, cv2.COLOR_RGB2HSV)
        hsv_input = img_hsv.astype('float32') / 255.0
        hsv_input = np.expand_dims(hsv_input, axis=0)
        
        nutrient_pred = multilabel_model.predict(hsv_input, verbose=0)
        predicted_index = np.argmax(nutrient_pred)
        predicted_class = class_names[predicted_index]
        confidence = nutrient_pred[0][predicted_index]
        
        return f"Nutrient Status: {predicted_class} (Confidence: {confidence:.4f})"
    else:
        return "Result: NOT a Rice Leaf. Please provide a clear rice leaf image."

# --- ප්‍රායෝගිකව පරීක්ෂා කිරීම ---
# පරීක්ෂා කිරීමට අවශ්‍ය පින්තූරයේ path එක මෙතනට ලබා දෙන්න
test_image_path = 'dataset/multi_label_classifier/Nitrogen_Deficiency/untitled-1.jpg'
if os.path.exists(test_image_path):
    result = predict_rice_health(test_image_path)
    print(f"\nAnalysis Result: {result}")
else:
    print(f"Please check the test image path: {test_image_path}")