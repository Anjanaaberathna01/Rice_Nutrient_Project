import os
import pandas as pd

DATASET_PATH = 'dataset/cleaned_multi_label_dataset'

# set up an empty list to hold the data
data = []

if not os.path.exists(DATASET_PATH):
    print(f" Error: '{DATASET_PATH}' folder not found. Please check the path.")
else:
    print(f" Scanning dataset in: {DATASET_PATH}...")

    # 2. Iterate through each label folder in the dataset
    for label in os.listdir(DATASET_PATH):
        label_path = os.path.join(DATASET_PATH, label)
        
        if os.path.isdir(label_path):
            count = 0
            for img_name in os.listdir(label_path):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    data.append({
                        # set the image path relative to the dataset folder and the corresponding label
                        'image_path': os.path.join(label, img_name),
                        'label': label
                    })
                    count += 1
            print(f" Found {count} images in class: {label}")

    # the data list is converted into a DataFrame and saved as a CSV file
    if data:
        # set up a DataFrame and save it to CSV
        df = pd.DataFrame(data)
        df.to_csv('database/multi_label_metadata.csv', index=False)
        print("-" * 30)
        print(f" Success! 'multi_label_metadata.csv' saved successfully.")
        print(f" Total images mapped: {len(df)}")
    else:
        print(" No images found! Please check your dataset folders.")