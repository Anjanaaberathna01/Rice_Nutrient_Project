import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

# මොඩල් 3 ට අදාළ නිවැරදි Preprocessing Functions වෙන වෙනම ඉම්පෝට් කිරීම
from tensorflow.keras.applications.efficientnet import preprocess_input as preprocess_efficient # type: ignore
from tensorflow.keras.applications.densenet import preprocess_input as preprocess_dense # type: ignore
from tensorflow.keras.applications.inception_v3 import preprocess_input as preprocess_inception # type: ignore

# 1. Paths සකස් කිරීම
test_data_dir = 'dataset/cleaned_test_data'
report_dir = 'reports/ensemble'

# 'reports' ෆෝල්ඩරය නැත්නම් එය ස්වයංක්‍රීයව සෑදීම
os.makedirs(report_dir, exist_ok=True)

# 2. පුහුණු කළ මොඩල් 3 ලෝඩ් කිරීම
print("Loading trained models...")
m_efficient = load_model('models/rice_multilabel_model_efficientnetb0.h5')
m_dense = load_model('models/rice_multilabel_model_densenet121.h5')
m_inception = load_model('models/rice_multilabel_model_inceptionv3.h5')

# 3. එක් එක් මොඩල් එකට අදාළ නිවැරදි Data Generators වෙන වෙනම සෑදීම (Shuffle=False අනිවාර්යයි)
print("Loading test datasets with correct model-specific preprocessing...")

gen_efficient = ImageDataGenerator(preprocessing_function=preprocess_efficient).flow_from_directory(
    test_data_dir, target_size=(224, 224), batch_size=32, class_mode='categorical', shuffle=False
)

gen_dense = ImageDataGenerator(preprocessing_function=preprocess_dense).flow_from_directory(
    test_data_dir, target_size=(224, 224), batch_size=32, class_mode='categorical', shuffle=False
)

gen_inception = ImageDataGenerator(preprocessing_function=preprocess_inception).flow_from_directory(
    test_data_dir, target_size=(224, 224), batch_size=32, class_mode='categorical', shuffle=False
)

class_labels = list(gen_dense.class_indices.keys())
true_classes = gen_dense.classes

# 4. මොඩල් තුනෙන් අදාළ Generators හරහා වෙන වෙනම අනාවැකි ලබා ගැනීම
print("Predicting with EfficientNetB0...")
pred_eff = m_efficient.predict(gen_efficient, verbose=1)

print("Predicting with DenseNet121...")
pred_d = m_dense.predict(gen_dense, verbose=1)

print("Predicting with InceptionV3...")
pred_i = m_inception.predict(gen_inception, verbose=1)

# ==========================================
# 5. ස්ථාවර බර (Fixed Weights) යෙදීම සහ Ensemble Logic එක ක්‍රියාත්මක කිරීම
# ==========================================
w_dense = 0.4
w_inception = 0.4
w_efficient = 0.2

print(f"\nApplying Fixed Weights: DenseNet121={w_dense}, InceptionV3={w_inception}, EfficientNetB0={w_efficient}")

# මොඩල් තුනේම දායකත්වයෙන් Ensemble Prediction සිදු කිරීම
ensemble_preds = (pred_d * w_dense) + (pred_i * w_inception) + (pred_eff * w_efficient)
predicted_classes = np.argmax(ensemble_preds, axis=1)

# සමස්ත Accuracy එක මැනීම
current_acc = accuracy_score(true_classes, predicted_classes)
print(f" Ensemble Accuracy with Fixed Weights: {current_acc*100:.2f}%")

# ==========================================
# 6. Classification Report එක Save කිරීම
# ==========================================
report_text = classification_report(true_classes, predicted_classes, target_names=class_labels)
report_path = os.path.join(report_dir, 'classification_report.txt')

with open(report_path, 'w') as f:
    f.write("Ensemble Model Classification Report (Fixed Weights)\n")
    f.write("==================================================\n\n")
    f.write(report_text)
print(f" Classification Report saved to: {report_path}")

# ==========================================
# 7. Confusion Matrix එක පින්තූරයක් ලෙස Save කිරීම
# ==========================================
cm = confusion_matrix(true_classes, predicted_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.title('Ensemble Model - Confusion Matrix (Fixed Weights)', fontsize=14)
plt.ylabel('True Target (Actual)', fontsize=12)
plt.xlabel('Predicted Output', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

cm_path = os.path.join(report_dir, 'confusion_matrix.png')
plt.savefig(cm_path, dpi=300)
plt.close()
print(f" Confusion Matrix saved to: {cm_path}")

# ==========================================
# 8. Multi-Class ROC Curve එක පින්තූරයක් ලෙස Save කිරීම
# ==========================================
y_test_bin = label_binarize(true_classes, classes=range(len(class_labels)))
n_classes = y_test_bin.shape[1]

plt.figure(figsize=(12, 8))
for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], ensemble_preds[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f'{class_labels[i]} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('Multi-Class ROC Curve (Ensemble Engine)', fontsize=14)
plt.legend(loc="lower right", fontsize='small')
plt.tight_layout()

roc_path = os.path.join(report_dir, 'roc_curve.png')
plt.savefig(roc_path, dpi=300)
plt.close()
print(f" ROC Curve saved to: {roc_path}")

print("\n All tests completed successfully! Please check the 'reports/ensemble' folder.")