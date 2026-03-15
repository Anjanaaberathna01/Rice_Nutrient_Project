import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.applications import MobileNetV2 # type: ignore
from tensorflow.keras import layers, models, callbacks # type: ignore
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils import class_weight
import numpy as np
import os

# 1. Image size and Batch size 
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# 2. Data Augmentation (පුහුණු දත්ත සඳහා පමණි)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.3,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

# Validation සඳහා පිරිසිදු Generator එකක් (Augmentation නැතිව)
val_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# 3. Dataset එක Load කිරීම
train_generator = train_datagen.flow_from_directory(
    'dataset/binary_classifier',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training'
)

val_generator = val_datagen.flow_from_directory(
    'dataset/binary_classifier',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation',
    shuffle=False # Confusion Matrix එක සඳහා අත්‍යවශ්‍යයි
)

# --- අලුතින් එකතු කළ කොටස: Class Weights ---
# Non-Rice (21,000) සහ Rice (15,000) අතර සමබරතාවය රැකීමට
labels = train_generator.classes
weights = class_weight.compute_class_weight('balanced', classes=np.unique(labels), y=labels)
class_weights = {i: weights[i] for i in range(len(weights))}
print(f"Class Weights: {class_weights}")

# 4. MobileNetV2 Model එක සෑදීම
base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid') 
])

# 5. Compile කිරීම
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# --- අලුතින් එකතු කළ කොටස: Callbacks ---
# Overfitting වැළැක්වීමට හොඳම weights පමණක් ලබා ගනිමු
early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# 6. Model එක පුහුණු කිරීම
print("Training started...")
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=15,
    class_weight=class_weights,
    callbacks=[early_stop]
)

# --- Fine-Tuning ආරම්භය ---
print("Fine-tuning started...")
base_model.trainable = True
for layer in base_model.layers[:100]:
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), 
              loss='binary_crossentropy', 
              metrics=['accuracy'])

history_fine = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10,
    class_weight=class_weights,
    callbacks=[early_stop]
)

# 7. Model එක සේව් කිරීම
if not os.path.exists('models'): os.makedirs('models')
model.save('models/rice_binary_model.h5')
print("Model saved successfully!")

# --- ඇගයීම් කොටස (Evaluation Section) ---

# Reports සඳහා Folder එකක් සෑදීම
if not os.path.exists('reports'):
    os.makedirs('reports')

# A. Accuracy සහ Loss Graphs ඇඳීම
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(1, len(acc) + 1)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.title('Binary Model Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.title('Binary Model Loss')
plt.legend()

plt.savefig('reports/binary_learning_curves.png')
plt.show()

# B. Confusion Matrix එක සෑදීම
print("\nGenerating Confusion Matrix...")
val_generator.reset()
Y_pred = model.predict(val_generator)
y_pred = np.where(Y_pred > 0.9, 1, 0)

cm = confusion_matrix(val_generator.classes, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Non-Rice', 'Rice'], 
            yticklabels=['Non-Rice', 'Rice'])
plt.title('Confusion Matrix - Binary Classifier')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')

plt.savefig('reports/binary_confusion_matrix.png')
plt.show()

# C. Precision, Recall වාර්තාව
print("\nClassification Report:")
print(classification_report(val_generator.classes, y_pred, target_names=['Non-Rice', 'Rice']))