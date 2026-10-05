import tensorflow as tf
from tensorflow.keras.applications import DenseNet121 # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization, Multiply, Input # type: ignore
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau # type: ignore
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import os
from sklearn.utils import class_weight

if not os.path.exists('reports'): os.makedirs('reports')
if not os.path.exists('models'): os.makedirs('models')

dataset_path = 'dataset/binary_disease_and_deficiency_dataset' 

# ==========================================
# 1.  Data Generators.
# ==========================================

# Training Augmentation 
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2 
)

# Validation  Augmentation  (Rescale only)
val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2 
)

train_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(224, 224),
    batch_size=16,
    class_mode='categorical',
    subset='training'
)

validation_generator = val_datagen.flow_from_directory(
    dataset_path,
    target_size=(224, 224),
    batch_size=16,
    class_mode='categorical',
    subset='validation',
    shuffle=False 
)

# ==========================================
# 2. Class Weights 
# ==========================================
labels = train_generator.classes
weights = class_weight.compute_class_weight('balanced', classes=np.unique(labels), y=labels)
class_weights = dict(enumerate(weights))
print(f"Class Weights Applied: {class_weights}")

# ==========================================
# 3. Model Architecture
# ==========================================
base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

inputs = Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = GlobalAveragePooling2D()(x)

# Squeeze-and-Excitation (SE) Block
se = Dense(x.shape[-1] // 16, activation='relu')(x)
se = Dense(x.shape[-1], activation='sigmoid')(se)
x = Multiply()([x, se])

x = BatchNormalization()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(2, activation='softmax')(x)

model = Model(inputs=inputs, outputs=predictions)

# ==========================================
# 4. Phase 1: Initial Training
# ==========================================
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), 
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model_save_path = 'models/rice_binary_disease_and_deficiency.h5'
checkpoint = ModelCheckpoint(model_save_path, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
early_stop = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)

print("\n Starting Initial Training (Phase 1)...")
model.fit(
    train_generator, 
    epochs=10, 
    validation_data=validation_generator, 
    class_weight=class_weights, 
    callbacks=[checkpoint, early_stop]
)

# ==========================================
# 5. Phase 2: Fine-Tuning
# ==========================================
print("\n Starting Fine-Tuning (Phase 2)...")
base_model.trainable = True
for layer in base_model.layers[:-50]: # Freeze the first 50 layers of DenseNet121
    layer.trainable = False

lr_callback = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-7, verbose=1)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), 
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(
    train_generator,
    epochs=15,
    validation_data=validation_generator,
    class_weight=class_weights,
    callbacks=[checkpoint, early_stop, lr_callback]
)

# ==========================================
# 6. Evaluation & Reports
# ==========================================
def save_matrix_and_report():
    validation_generator.reset()
    Y_pred = model.predict(validation_generator)
    y_pred = np.argmax(Y_pred, axis=1) 
    
    cm = confusion_matrix(validation_generator.classes, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Deficiency', 'Disease'], 
                yticklabels=['Deficiency', 'Disease'])
    plt.title('Confusion Matrix (DenseNet121 + SE Block)', fontsize=14)
    plt.ylabel('True Target (Actual)')
    plt.xlabel('Predicted Output')
    plt.tight_layout()
    plt.savefig('reports/rice_binary_disease_and_deficiency_confusion_matrix.png', dpi=300)
    plt.close()
    
    report = classification_report(validation_generator.classes, y_pred, target_names=['Deficiency', 'Disease'])
    with open('reports/rice_binary_disease_and_deficiency_report.txt', 'w') as f: 
        f.write("Binary Model Classification Report\n")
        f.write("====================================\n\n")
        f.write(report)
    print("Reports and Matrix saved successfully in the 'reports' folder.")

save_matrix_and_report()
model.save(model_save_path)
print("Training Completely Finished!")