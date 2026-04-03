from rembg import remove
import PIL.Image
import os

input_dir = 'dataset/multi_label_classifier' # ඔයාගේ දැනට තියෙන පින්තූර
output_dir = 'dataset/cleaned_multi_label_dataset'       # පිරිසිදු පින්තූර සේව් වන තැන

os.makedirs(output_dir, exist_ok=True)

for root, dirs, files in os.walk(input_dir):
    for file in files:
        if file.lower().endswith(('.jpg', '.png', '.jpeg')):
            input_path = os.path.join(root, file)
            # sub-folder structure එක පවත්වා ගැනීම
            rel_path = os.path.relpath(root, input_dir)
            target_folder = os.path.join(output_dir, rel_path)
            os.makedirs(target_folder, exist_ok=True)
            
            # පින්තූරය Clean කිරීම
            input_image = PIL.Image.open(input_path)
            output_image = remove(input_image)
            
            # RGB වලට හැරවීම (Rembg output එක RGBA නිසා)
            output_image = output_image.convert("RGB") 
            output_image.save(os.path.join(target_folder, file))
            print(f"✅ Processed: {file}")