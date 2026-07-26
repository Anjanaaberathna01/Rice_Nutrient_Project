import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.applications import DenseNet121 # type: ignore
from tensorflow.keras.applications.densenet import preprocess_input # type: ignore
from tensorflow.keras import layers, models, optimizers # type: ignore
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix, classification_report

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
METADATA_CSV = 'multi_label_metadata.csv'
DATA_DIR = 'dataset/cleaned_multi_label_dataset'
REPORTS_DIR = 'reports/densenet'

if not os.path.exists(REPORTS_DIR):
    os.makedirs(REPORTS_DIR)

# 2. get CSV 
df = pd.read_csv(METADATA_CSV)
df['image_path'] = df['image_path'].str.replace('\\', '/', regex=False)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
classes = sorted(df['label'].unique())

# 3. Data Generators 
datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=15,
    horizontal_flip=True,
    validation_split=0.2
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

# 4. Class Weights
labels = train_gen.classes
weights = class_weight.compute_class_weight('balanced', classes=np.unique(labels), y=labels)
class_weights = dict(enumerate(weights))

# 5. Model Building (DenseNet121)
base_model = DenseNet121(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.trainable = False 

inputs = layers.Input(shape=(224, 224, 3))
x = base_model(inputs)
x = layers.GlobalAveragePooling2D()(x)

# SE Block
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

print("\n DenseNet121: Phase 1 Starting...")
h1 = model.fit(train_gen, validation_data=val_gen, epochs=10, class_weight=class_weights)

# 7. Phase 2: Fine-Tuning
base_model.trainable = True
for layer in base_model.layers[:-50]: layer.trainable = False 

lr_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-7)

model.compile(optimizer=optimizers.Adam(learning_rate=1e-5), 
              loss='categorical_crossentropy', metrics=['accuracy'])

print("\n🔥 DenseNet121: Phase 2 Fine-Tuning...")
h2 = model.fit(train_gen, validation_data=val_gen, epochs=20, 
              class_weight=class_weights, callbacks=[lr_callback])

# 8. Saving Model
model.save('models/rice_multilabel_model_densenet121.h5')

# 9. Reports
def save_reports(h1, h2, model, val_gen):
    acc = h1.history['accuracy'] + h2.history['accuracy']
    val_acc = h1.history['val_accuracy'] + h2.history['val_accuracy']
    
    plt.figure(figsize=(10, 5))
    plt.plot(acc, label='Train Acc'); plt.plot(val_acc, label='Val Acc')
    plt.title('DenseNet121 Accuracy'); plt.legend(); plt.savefig(f'{REPORTS_DIR}/densenet_accuracy.png'); plt.close()

    val_gen.reset()
    preds = np.argmax(model.predict(val_gen), axis=1)
    cm = confusion_matrix(val_gen.classes, preds)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix - DenseNet121'); plt.savefig(f'{REPORTS_DIR}/densenet_matrix.png'); plt.close()

    with open(f'{REPORTS_DIR}/densenet_report.txt', 'w') as f:
        f.write(classification_report(val_gen.classes, preds, target_names=classes))
    print(f"DenseNet121 Reports saved in {REPORTS_DIR}")

save_reports(h1, h2, model, val_gen)