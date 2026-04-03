import tensorflow as tf # type: ignore
from tensorflow.keras.applications import DenseNet121 # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization # type: ignore
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau # type: ignore
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import os

# 1. පෝල්ඩර් සකස් කිරීම
if not os.path.exists('reports'): os.makedirs('reports')
if not os.path.exists('models'): os.makedirs('models')

dataset_path = 'dataset/binary_disease_and_deficiency_dataset' 

# 2. Data Augmentation (Categorical Mode)
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

train_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(224, 224),
    batch_size=16,
    class_mode='categorical', # පන්ති දෙකම වෙන වෙනම හඳුනා ගැනීමට
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(224, 224),
    batch_size=16,
    class_mode='categorical',
    subset='validation',
    shuffle=False 
)

# 3. Model Architecture (Softmax for Probabilities)
base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False # මුලින් Base Model එක Freeze කරමු

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(2, activation='softmax')(x) # නියුරෝන 2 සහ Softmax

model = Model(inputs=base_model.input, outputs=predictions)

# 4. Compilation (Categorical Crossentropy)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), 
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 5. Callbacks
model_save_path = 'models/rice_binary_disease_and_deficiency_v2.h5'
checkpoint = ModelCheckpoint(model_save_path, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
early_stop = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)

# 6. පියවර 01: Initial Training
print("Starting Initial Training...")
model.fit(train_generator, epochs=10, validation_data=validation_generator, callbacks=[checkpoint, early_stop])

# --- 7. පියවර 02: Fine-Tuning (Accuracy 90% කරා ගෙන යාමට) ---
print("Starting Fine-Tuning...")
base_model.trainable = True
# මුල් ලේයර්ස් 100 ක් Freeze කර ඉතිරි ඒවා පුහුණු කිරීම
for layer in base_model.layers[:100]:
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), 
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(
    train_generator,
    epochs=15,
    validation_data=validation_generator,
    callbacks=[checkpoint, early_stop]
)

# 8. Evaluation (Confusion Matrix නිවැරදි කිරීම)
def save_matrix_and_report():
    validation_generator.reset()
    Y_pred = model.predict(validation_generator)
    y_pred = np.argmax(Y_pred, axis=1) # වැඩිම සම්භාවිතාව ඇති පන්තිය තෝරා ගැනීම
    
    cm = confusion_matrix(validation_generator.classes, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
                xticklabels=['Deficiency', 'Disease'], 
                yticklabels=['Deficiency', 'Disease'])
    plt.title('Confusion Matrix (Softmax Model)')
    plt.savefig('reports/rice_binary_disease_and_deficiency_v2_confusion_matrix.png')
    
    report = classification_report(validation_generator.classes, y_pred, target_names=['Deficiency', 'Disease'])
    with open('reports/rice_binary_disease_and_deficiency_v2_report.txt', 'w') as f: f.write(report)

save_matrix_and_report()
model.save(model_save_path)