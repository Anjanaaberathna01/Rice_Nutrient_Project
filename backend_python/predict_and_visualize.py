import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# 1. මොඩලය සහ පරාමිතීන් (Load Model & Parameters)
MODEL_PATH = 'models/rice_multilabel_model_v2.h5'
IMG_SIZE = (224, 224)

# පන්ති නාමයන් (Metadata CSV එකේ තිබුණු අනුපිළිවෙළටම)
class_names = [
    'Bacterial_Leaf_Blight', 'Brown_Spot', 'Healthy', 'Leaf_Blast', 
    'Leaf_Scald', 'Nitrogen_Deficiency', 'Phosphorus_Deficiency', 
    'Potassium_Deficiency', 'Sheath_Blight'
]

# මොඩලය පූරණය කිරීම
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Model එක load කිරීමේදී ගැටලුවක් ආවා: {e}")

def get_gradcam_heatmap(img_array, model, last_conv_layer_name):
    """Grad-CAM Heatmap එක ජනනය කරන ශ්‍රිතය"""
    # MobileNetV2 base model එක හඳුනාගැනීම
    base_model = model.layers[0]
    
    # 1. Conv layer එක දක්වා කොටස සහිත මොඩලයක් හැදීම
    last_conv_layer = base_model.get_layer(last_conv_layer_name)
    last_conv_layer_model = tf.keras.Model(base_model.inputs, last_conv_layer.output)
    
    # 2. ඉතිරි Layers (Top layers) සහිත මොඩලයක් හැදීම
    classifier_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input
    for layer in model.layers[1:]:
        x = layer(x)
    classifier_model = tf.keras.Model(classifier_input, x)
    
    # 3. Gradients මැනීම
    with tf.GradientTape() as tape:
        last_conv_layer_output = last_conv_layer_model(img_array)
        tape.watch(last_conv_layer_output)
        preds = classifier_model(last_conv_layer_output)
        top_pred_index = tf.argmax(preds[0])
        top_class_channel = preds[:, top_pred_index]

    # Gradients ගණනය කිරීම
    grads = tape.gradient(top_class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    # Normalization (0-1 අතරට ගැනීම)
    heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap)
    return heatmap.numpy()

def predict_with_heatmap(image_path):
    """පින්තූරය පරීක්ෂා කර Heatmap එක පෙන්වන ශ්‍රිතය"""
    # පින්තූරය කියවීම
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ පින්තූරය සොයාගත නොහැක: {image_path}")
        return
        
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, IMG_SIZE)
    img_array = preprocess_input(np.expand_dims(img_resized, axis=0))

    # 1. Predictions ලබා ගැනීම
    preds = model.predict(img_array)[0]
    
    # 2. Heatmap සාදා ගැනීම
    try:
        # MobileNetV2 හි අවසාන වැදගත් layer එක 'out_relu' වේ
        heatmap = get_gradcam_heatmap(img_array, model, 'out_relu')
        
        # Heatmap එක මුල් පින්තූරයේ ප්‍රමාණයට සැකසීම
        heatmap_resized = cv2.resize(heatmap, (img_rgb.shape[1], img_rgb.shape[0]))
        heatmap_resized = np.uint8(255 * heatmap_resized)
        heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
        
        # මුල් පින්තූරය මත Heatmap එක ඇඳීම (Overlay)
        superimposed_img = cv2.addWeighted(img_rgb, 0.6, heatmap_colored, 0.4, 0)
        
        # 3. ප්‍රතිඵල Visualise කිරීම
        plt.figure(figsize=(14, 6))
        
        plt.subplot(1, 2, 1)
        plt.imshow(img_rgb)
        plt.title("Original Rice Leaf")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(superimposed_img)
        plt.title("AI Focus Area (Heatmap)")
        plt.axis('off')
        
        print("\n--- Model Analysis Results ---")
        found_any = False
        for i, prob in enumerate(preds):
            if prob > 0.1:  # 10% ට වැඩි ප්‍රතිඵල පමණක් පෙන්වන්න
                print(f"🚩 {class_names[i]}: {prob*100:.2f}%")
                found_any = True
        
        if not found_any:
            print("❔ No significant disease detected.")
            
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"❌ Error during Heatmap generation: {e}")

# --- මෙතනට ඔයාගේ පින්තූරයේ නම දෙන්න ---
# උදාහරණ: predict_with_heatmap('test_image.jpg')
predict_with_heatmap('16.jpg')