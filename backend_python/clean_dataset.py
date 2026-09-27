from rembg import remove
import PIL.Image
import os

input_dir = 'dataset/test_data' 
output_dir = 'dataset/cleaned_test_data' 

os.makedirs(output_dir, exist_ok=True)

for root, dirs, files in os.walk(input_dir):
    for file in files:
        if file.lower().endswith(('.jpg', '.png', '.jpeg')):
            input_path = os.path.join(root, file)
            rel_path = os.path.relpath(root, input_dir)
            target_folder = os.path.join(output_dir, rel_path)
            os.makedirs(target_folder, exist_ok=True)
            
            # clean the image and save it to the target folder
            input_image = PIL.Image.open(input_path)
            output_image = remove(input_image)
            
            # Convert to RGB and save the cleaned image
            output_image = output_image.convert("RGB") 
            output_image.save(os.path.join(target_folder, file))
            print(f"Processed: {file}")