import os
import pandas as pd

# 1. පින්තූර ගබඩා කර ඇති ප්‍රධාන ෆෝල්ඩරයේ මාවත (Path) ලබා දීම
DATASET_PATH = 'dataset/multi_label_classifier'

# දත්ත තාවකාලිකව ගබඩා කර ගැනීමට ලැයිස්තුවක් (List) සෑදීම
data = []

# 2. නියමිත ෆෝල්ඩරය පරිගණකයේ පවතින්නේදැයි පරීක්ෂා කිරීම (Scanning)
if not os.path.exists(DATASET_PATH):
    # ෆෝල්ඩරය නොමැති නම් දෝෂ පණිවිඩයක් පෙන්වීම
    print(f" Error: '{DATASET_PATH}' folder not found. Please check the path.")
else:
    print(f" Scanning dataset in: {DATASET_PATH}...")
    
    # පෝෂක ඌණතා සහ රෝග නාමයන් සහිත සියලුම සබ්-ෆෝල්ඩර පරීක්ෂා කිරීම
    for label in os.listdir(DATASET_PATH):
        label_path = os.path.join(DATASET_PATH, label)
        
        # මෙය ෆෝල්ඩරයක් දැයි තහවුරු කර ගැනීම
        if os.path.isdir(label_path):
            count = 0
            # ෆෝල්ඩරය ඇතුළේ ඇති සියලුම ගොනු එකින් එක පරීක්ෂා කිරීම
            for img_name in os.listdir(label_path):
                # ගොනුව පින්තූරයක් (.png, .jpg, .jpeg) දැයි පරීක්ෂා කිරීම
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    # පින්තූරයේ මාවත සහ එහි කාණ්ඩය (Label) දත්ත ලැයිස්තුවට ඇතුළත් කිරීම
                    data.append({
                        'image_path': os.path.join(label, img_name),
                        'label': label
                    })
                    count += 1
            # එම කාණ්ඩයේ සොයාගත් පින්තූර සංඛ්‍යාව පෙන්වීම
            print(f" Found {count} images in class: {label}")

    # 3. එකතු කරගත් දත්ත භාවිතයෙන් CSV ගොනුව සෑදීම සහ සේව් කිරීම
    if data:
        # Pandas DataFrame එකක් සාදා ගැනීම
        df = pd.DataFrame(data)
        # මෙය ප්‍රධාන ප්‍රොජෙක්ට් ෆෝල්ඩරය තුළ CSV එකක් ලෙස සේව් වේ
        df.to_csv('multi_label_metadata.csv', index=False)
        print("-" * 30)
        print(f" Success! 'multi_label_metadata.csv' saved successfully.")
        print(f" Total images mapped: {len(df)}")
    else:
        # පින්තූර කිසිවක් හමු නොවූයේ නම් අනතුරු ඇඟවීමක් පෙන්වීම
        print(" No images found! Please check your dataset folders.")