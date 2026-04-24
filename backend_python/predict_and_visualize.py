import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from rembg import remove
from PIL import Image
import io

# 1. පරාමිතීන් සහ මාදිලි Load කිරීම
IMG_SIZE = (224, 224)
CLASS_NAMES = [
    'Bacterial_Leaf_Blight', 'Brown_Spot', 'Healthy', 'Leaf_Blast', 
    'Leaf_Scald', 'Nitrogen_Deficiency', 'Phosphorus_Deficiency', 
    'Potassium_Deficiency', 'Sheath_Blight'
]

# Indices වෙන් කිරීම
DEFICIENCY_INDICES = [2, 5, 6, 7] 
DISEASE_INDICES = [0, 1, 3, 4, 8]  

print("🔄 Models Loading... Please wait.")
# Softmax Binary Model Load (v2)
binary_model = tf.keras.models.load_model('models/rice_binary_disease_and_deficiency_v2.h5', compile=False)

# Ensemble Models Load 
model_mobile = tf.keras.models.load_model('models/rice_multilabel_model_v4_final.h5', compile=False)
model_dense = tf.keras.models.load_model('models/rice_multilabel_model_densenet121.h5', compile=False)
model_inception = tf.keras.models.load_model('models/rice_multilabel_model_inceptionv3.h5', compile=False)
print("✅ All Models Loaded Successfully!")

def generate_heatmap_ensemble(img_array, model):
    try:
        base_model = model.get_layer('densenet121')
        last_conv_layer = base_model.layers[-1]
        feature_extractor = tf.keras.Model(inputs=base_model.input, outputs=last_conv_layer.output)
        features = feature_extractor.predict(img_array, verbose=0)[0]
        heatmap = np.mean(features, axis=-1)
        heatmap = np.maximum(heatmap, 0)
        heatmap /= (np.max(heatmap) + 1e-10)
        return heatmap
    except: return None

def predict_ensemble(image_path):
    if not os.path.exists(image_path):
        print(f"Photo not found: {image_path}")
        return

    # --- 1. Background Removal ---
    print(f"Preprocessing: {image_path}")
    with open(image_path, 'rb') as i:
        output_data = remove(i.read())
        img_cleaned = Image.open(io.BytesIO(output_data)).convert("RGB")
        img_np = np.array(img_cleaned)

    # --- 2. Preprocessing ---
    img_resized = cv2.resize(img_np, IMG_SIZE)
    img_array = np.expand_dims(img_resized, axis=0)
    binary_input = img_array / 255.0

    # --- 3. Binary Softmax Filtering ---
    bin_preds = binary_model.predict(binary_input, verbose=0)[0]
    deficiency_prob = float(bin_preds[0])
    disease_prob = float(bin_preds[1])

    is_disease = disease_prob > deficiency_prob
    
    # --- Soft-Thresholding Logic ---
    uncertainty_threshold = 0.15 # 15% ක වෙනස
    diff = abs(disease_prob - deficiency_prob)
    is_mixed = diff < uncertainty_threshold

    # --- 4. Ensemble Predictions ---
    in_m = tf.keras.applications.mobilenet_v2.preprocess_input(img_array.copy())
    in_d = tf.keras.applications.densenet.preprocess_input(img_array.copy())
    in_i = tf.keras.applications.inception_v3.preprocess_input(img_array.copy())

    p_m = model_mobile.predict(in_m, verbose=0)[0]
    p_d = model_dense.predict(in_d, verbose=0)[0]
    p_i = model_inception.predict(in_i, verbose=0)[0]

    final_preds = (p_m * 0.3) + (p_d * 0.4) + (p_i * 0.3)

    # --- 5. Smart Soft-Filtering Logic ---
    if not is_mixed:
        if is_disease:
            for idx in DEFICIENCY_INDICES: final_preds[idx] = 0
        else:
            for idx in DISEASE_INDICES: final_preds[idx] = 0
    else:
        print("⚠️ මිශ්‍ර ලක්ෂණ පවතින බැවින් සියලුම පන්ති පරීක්ෂා කරයි.")

    # --- 6. Visualization ---
    heatmap = generate_heatmap_ensemble(in_d, model_dense)
    plt.figure(figsize=(20, 6))
    
    plt.subplot(1, 4, 1)
    plt.imshow(cv2.imread(image_path)[:,:,::-1])
    plt.title(f"Original\n({'MIXED' if is_mixed else 'DISEASE' if is_disease else 'DEFICIENCY'})")
    plt.axis('off')
    
    plt.subplot(1, 4, 2)
    plt.imshow(img_np)
    plt.title(f"Binary Output\nDis: {disease_prob:.0%} | Def: {deficiency_prob:.0%}")
    plt.axis('off')
    
    if heatmap is not None:
        heatmap_resized = cv2.resize(heatmap, (img_np.shape[1], img_np.shape[0]))
        heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        superimposed_img = cv2.addWeighted(img_np, 0.6, heatmap_color, 0.4, 0)
        plt.subplot(1, 4, 3); plt.imshow(superimposed_img)
        plt.title("Ensemble Attention Map"); plt.axis('off')

    plt.subplot(1, 4, 4)
    top_indices = np.argsort(final_preds)[-3:][::-1]
    top_labels = [CLASS_NAMES[i] for i in top_indices]
    top_scores = [final_preds[i] for i in top_indices]
    # වර්ණය තීරණය කිරීම (මිශ්‍ර නම් දොඩම් පැහැය)
    bar_color = 'orange' if is_mixed else ('red' if is_disease else 'green')
    plt.barh(top_labels, top_scores, color=bar_color)
    plt.title(f"Top Results\n(Conf: {final_preds[top_indices[0]]:.2%})")
    plt.xlim(0, 1); plt.gca().invert_yaxis()
    
    plt.tight_layout(); plt.show()

    # --- 7. වාර්තාව ---
    print("\n" + "="*65)
    category_info = "MIXED/UNCERTAIN" if is_mixed else ("DISEASE" if is_disease else "DEFICIENCY")
    print(f"FINAL CATEGORY: {category_info} (Diff: {diff:.1%})")
    print("="*65)
    print(f"{'CLASS NAME':<25} | {'V4':<7} | {'DENSE':<7} | {'INCEP':<7} | {'FINAL'}")
    print("-" * 65)
    for i in np.argsort(final_preds)[::-1]:
        if final_preds[i] > 0.001:
            print(f"{CLASS_NAMES[i]:<25} | {p_m[i]:.2f} | {p_d[i]:.2f} | {p_i[i]:.2f} | {final_preds[i]:.2%}")
    print("="*65)

# පරීක්ෂා කිරීමට රූපයේ නම මෙතැනට ඇතුළත් කරන්න
predict_ensemble('42.jpeg')