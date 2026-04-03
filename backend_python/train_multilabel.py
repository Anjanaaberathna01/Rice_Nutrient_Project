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

# 1. සැකසුම් (Settings)
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
METADATA_CSV = 'multi_label_metadata.csv'
DATA_DIR = 'dataset/cleaned_multi_label_dataset' # ඔයාගේ අලුත් ෆෝල්ඩරය
REPORTS_DIR = 'reports'

if not os.path.exists(REPORTS_DIR):
    os.makedirs(REPORTS_DIR)

# 2. CSV දත්ත පිරිසැකසුම
df = pd.read_csv(METADATA_CSV)
df['image_path'] = df['image_path'].str.replace('\\', '/', regex=False)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

classes = sorted(df['label'].unique())
print(f"✅ පන්ති 9 හඳුනාගත්තා: {classes}")

# 3. Data Generators (Cleaned නිසා Augmentation සරල කර ඇත)
datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=15,
    horizontal_flip=True,
    validation_split=0.2 # 20% ක් පරීක්ෂණ සඳහා
)

train_gen = datagen.flow_from_dataframe(
    df, directory=DATA_DIR, x_col='image_path', y_col='label',
    target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical',
    subset='training', shuffle=True
)

val_gen = datagen.flow_from_dataframe(
    df, directory=DATA_DIR, x_col='image_path', y_col='label',
    target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical',
    subset='validation', shuffle=False
)

# 4. Class Weights (N,P,K ඌණතාවයන්ට වැඩි බරක් දීමට)
labels = train_gen.classes
weights = class_weight.compute_class_weight('balanced', classes=np.unique(labels), y=labels)
class_weights = dict(enumerate(weights))

# 5. Model Building (Attention Mechanism සමඟ)
base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.trainable = False 

inputs = layers.Input(shape=(224, 224, 3))
x = base_model(inputs)
x = layers.GlobalAveragePooling2D()(x)

# Squeeze-and-Excitation (SE) Block - වර්ණ ලක්ෂණ වලට වැඩි අවධානයක් දීමට
se = layers.Dense(x.shape[-1] // 16, activation='relu')(x)
se = layers.Dense(x.shape[-1], activation='sigmoid')(se)
x = layers.Multiply()([x, se])

x = layers.BatchNormalization()(x)
x = layers.Dense(512, activation='relu')(x)
x = layers.Dropout(0.4)(x)
outputs = layers.Dense(len(classes), activation='softmax')(x)

model = models.Model(inputs, outputs)

# 6. Phase 1: Warming up
model.compile(optimizer=optimizers.Adam(learning_rate=1e-3), 
              loss='categorical_crossentropy', metrics=['accuracy'])

print("\n🚀 Phase 1 ආරම්භ වේ (Top Layers Training)...")
h1 = model.fit(train_gen, validation_data=val_gen, epochs=10, class_weight=class_weights)

# 7. Phase 2: Fine-Tuning
base_model.trainable = True
for layer in base_model.layers[:-50]: layer.trainable = False # අන්තිම ලේයර් 50 පමණක් Open කිරීම

lr_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-7)

model.compile(optimizer=optimizers.Adam(learning_rate=1e-5), 
              loss='categorical_crossentropy', metrics=['accuracy'])

print("\n🔥 Phase 2 ආරම්භ වේ (Fine-tuning for High Accuracy)...")
h2 = model.fit(train_gen, validation_data=val_gen, epochs=20, 
              class_weight=class_weights, callbacks=[lr_callback])

# 8. Saving Model
model.save('models/rice_multilabel_model_v4_final.h5')

# 9. වාර්තා (Reports) සකස් කිරීම
def save_reports(h1, h2, model, val_gen):
    # Accuracy & Loss Curves
    acc = h1.history['accuracy'] + h2.history['accuracy']
    val_acc = h1.history['val_accuracy'] + h2.history['val_accuracy']
    
    plt.figure(figsize=(10, 5))
    plt.plot(acc, label='Training Accuracy'); plt.plot(val_acc, label='Validation Accuracy')
    plt.title('V4 Accuracy Curve'); plt.legend(); plt.savefig(f'{REPORTS_DIR}/v4_accuracy.png'); plt.close()

    # Confusion Matrix
    val_gen.reset()
    preds = np.argmax(model.predict(val_gen), axis=1)
    cm = confusion_matrix(val_gen.classes, preds)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix - V4 (Cleaned)'); plt.savefig(f'{REPORTS_DIR}/v4_matrix.png'); plt.close()

    # Text Report
    with open(f'{REPORTS_DIR}/v4_report.txt', 'w') as f:
        f.write(classification_report(val_gen.classes, preds, target_names=classes))

    print(f"✅ සියලුම වාර්තා {REPORTS_DIR} ෆෝල්ඩරයේ සේව් කරන ලදී.")

save_reports(h1, h2, model, val_gen)