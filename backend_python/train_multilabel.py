import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.applications import MobileNetV2 # type: ignore
from tensorflow.keras import layers, models # type: ignore
import pandas as pd
import cv2 # type: ignore
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import os
from sklearn.utils import class_weight

# 1. පින්තූර RGB සිට HSV වලට හැරවීමේ ශ්‍රිතය
def convert_to_hsv(image):
    image_hsv = cv2.cvtColor(image.astype('uint8'), cv2.COLOR_RGB2HSV)
    return image_hsv.astype('float32') / 255.0 # Normalization

# 2. දත්ත සහ පරාමිතීන්
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
df = pd.read_csv('multi_label_metadata.csv')

# 3. Data Augmentation
datagen = ImageDataGenerator(
    rescale=None, # preprocessing_function එකෙන් අපි මේක කරනවා
    preprocessing_function=convert_to_hsv,
    validation_split=0.2
)

# 4. Generators සකස් කිරීම
train_gen = datagen.flow_from_dataframe(
    df,
    directory='dataset/multi_label_classifier',
    x_col='image_path',
    y_col='label',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_gen = datagen.flow_from_dataframe(
    df,
    directory='dataset/multi_label_classifier',
    x_col='image_path',
    y_col='label',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# 5. Class Weights ගණනය කිරීම (Imbalance එක විසඳීමට)
labels = train_gen.classes
weights = class_weight.compute_class_weight('balanced', classes=np.unique(labels), y=labels)
class_weights = dict(enumerate(weights))

# 6. Model එක ගොඩනැගීම (MobileNetV2)
base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(4, activation='softmax') # Healthy, N, P, K classes 4 සඳහා
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 7. Training ආරම්භ කිරීම
print("Multi-label Training Started...")
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10,
    class_weight=class_weights
)

# 8. Model එක සේව් කිරීම
if not os.path.exists('models'): os.makedirs('models')
model.save('models/rice_multilabel_model.h5')
print("Multi-label Model Saved Successfully!")

# --- EVALUATION & GRAPH GENERATION FOR MULTI-LABEL ---

# 1. පියවර: Accuracy සහ Loss Graphs ඇඳීම
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(1, len(acc) + 1)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.title('Multi-label Model Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.title('Multi-label Model Loss')
plt.legend()

# Graphs සේව් කිරීම
if not os.path.exists('reports'): os.makedirs('reports')
plt.savefig('reports/multilabel_learning_curves.png')
plt.show()

# 2. පියවර: Confusion Matrix එක සෑදීම
print("\nGenerating Multi-class Confusion Matrix...")
val_gen.reset()
Y_pred = model.predict(val_gen)
y_pred = np.argmax(Y_pred, axis=1) # වැඩිම සම්භාවිතාව ඇති Class එක තෝරා ගැනීම

# Class names ලබා ගැනීම
class_labels = list(val_gen.class_indices.keys())

cm = confusion_matrix(val_gen.classes, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
            xticklabels=class_labels, 
            yticklabels=class_labels)
plt.title('Confusion Matrix - NPK Diagnosis')
plt.ylabel('Actual Category')
plt.xlabel('Predicted Category')

# Matrix එක සේව් කිරීම
plt.savefig('reports/multilabel_confusion_matrix.png')
plt.show()

# 3. පියවර: පර්යේෂණ වාර්තාවට අවශ්‍ය නිරවද්‍යතා දත්ත (Precision/Recall)
print("\nMulti-label Classification Report:")
print(classification_report(val_gen.classes, y_pred, target_names=class_labels))