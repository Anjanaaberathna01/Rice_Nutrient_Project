import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.applications import MobileNetV2 # type: ignore
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input # type: ignore
from tensorflow.keras import layers, models, optimizers # type: ignore
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix, classification_report

# 1. පරාමිතීන් (Parameters)
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
METADATA_CSV = 'multi_label_metadata.csv'

# 2. Data Augmentation (N, P, K වලට විශේෂ අවධානයක් දීමට Brightness/Zoom වැඩි කළා)
datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.3,
    brightness_range=[0.7, 1.3], 
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

df = pd.read_csv(METADATA_CSV)
classes = sorted(df['label'].unique())

train_gen = datagen.flow_from_dataframe(
    df, directory='dataset/multi_label_classifier', x_col='image_path', y_col='label',
    target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical',
    subset='training', shuffle=True
)

val_gen = datagen.flow_from_dataframe(
    df, directory='dataset/multi_label_classifier', x_col='image_path', y_col='label',
    target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical',
    subset='validation', shuffle=False
)

# 3. Class Weights (N,P,K පන්තිවලට මොඩලය තුළ වැඩි බරක් ලබා දීම)
labels = train_gen.classes
weights = class_weight.compute_class_weight('balanced', classes=np.unique(labels), y=labels)
class_weights = dict(enumerate(weights))

# 4. Model Building with Attention (Squeeze-and-Excitation Layer)
base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.trainable = False 

inputs = layers.Input(shape=(224, 224, 3))
x = base_model(inputs)
x = layers.GlobalAveragePooling2D()(x)

# --- Attention Block: මෙය පත්‍රයේ සියුම් වර්ණ සහ දාරවල ලක්ෂණ වැඩි දියුණු කරයි ---
se = layers.Dense(x.shape[-1] // 16, activation='relu')(x)
se = layers.Dense(x.shape[-1], activation='sigmoid')(se)
x = layers.Multiply()([x, se])

x = layers.BatchNormalization()(x)
x = layers.Dense(512, activation='relu')(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(len(classes), activation='softmax')(x)

model = models.Model(inputs, outputs)

# 5. Training Phase 1: Warming up Top Layers
model.compile(optimizer=optimizers.Adam(learning_rate=1e-3), 
              loss='categorical_crossentropy', metrics=['accuracy'])

print("\n--- Phase 1: Training Top Layers ---")
history1 = model.fit(train_gen, validation_data=val_gen, epochs=10, class_weight=class_weights)

# 6. Training Phase 2: Fine-Tuning (පද්ධතියේ ගැඹුර වැඩි කිරීම)
base_model.trainable = True
for layer in base_model.layers[:-60]: layer.trainable = False

# Learning Rate එක ස්වයංක්‍රීයව අඩු කරන Callback එක (Precision සඳහා)
lr_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-7)

model.compile(optimizer=optimizers.Adam(learning_rate=1e-5), 
              loss='categorical_crossentropy', metrics=['accuracy'])

print("\n--- Phase 2: Fine-Tuning for N,P,K Precision ---")
history2 = model.fit(train_gen, validation_data=val_gen, epochs=20, 
                     class_weight=class_weights, callbacks=[lr_callback])

# 7. Saving the Model
model_path = 'models/rice_multilabel_model_v3.h5'
model.save(model_path)
print(f"✅ Advanced Model Saved as '{model_path}'")

# 8. Visualizing Results (For Thesis/Report)

# Learning Curves
def plot_history(h1, h2):
    acc = h1.history['accuracy'] + h2.history['accuracy']
    val_acc = h1.history['val_accuracy'] + h2.history['val_accuracy']
    loss = h1.history['loss'] + h2.history['loss']
    val_loss = h1.history['val_loss'] + h2.history['val_loss']
    
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Train Accuracy')
    plt.plot(val_acc, label='Val Accuracy')
    plt.legend(); plt.title('Accuracy')
    
    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Train Loss')
    plt.plot(val_loss, label='Val Loss')
    plt.legend(); plt.title('Loss')
    plt.show()

# Confusion Matrix
def show_confusion_matrix(model, val_gen):
    val_gen.reset()
    y_pred = np.argmax(model.predict(val_gen), axis=1)
    y_true = val_gen.classes
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted'); plt.ylabel('Actual'); plt.title('Confusion Matrix')
    plt.show()
    print(classification_report(y_true, y_pred, target_names=classes))

plot_history(history1, history2)
show_confusion_matrix(model, val_gen)