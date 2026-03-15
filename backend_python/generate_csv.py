import os
import pandas as pd

# 1. පින්තූර තියෙන folder එක (backend_python ඇතුළේ dataset තියෙනවා නම්)
dataset_path = 'dataset/multi_label_classifier'

data = []

# සම්පූර්ණ path එක පරීක්ෂා කිරීම
abs_path = os.path.abspath(dataset_path)
print(f"Checking path: {abs_path}")

if os.path.exists(dataset_path):
    # folder එක ඇතුළේ තියෙන sub-folders හඳුනා ගැනීම
    actual_folders = [f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))]
    print(f"Folders found: {actual_folders}")

    for label in actual_folders:
        folder_path = os.path.join(dataset_path, label)
        files = os.listdir(folder_path)
        
        count = 0
        for img_name in files:
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                data.append({
                    'image_path': os.path.join(label, img_name),
                    'label': label
                })
                count += 1
        print(f"Processed {label}: {count} images found.")
else:
    print(f"!!! ERROR: Path does not exist at {abs_path}")

# 2. CSV එක සෑදීම
if len(data) > 0:
    df = pd.DataFrame(data)
    df.to_csv('multi_label_metadata.csv', index=False)
    print(f"\nSUCCESS! Created CSV with {len(df)} images.")
    print(df['label'].value_counts())
else:
    print("\n!!! STILL NO IMAGES. Please check if your 'dataset' folder is INSIDE 'backend_python'.")