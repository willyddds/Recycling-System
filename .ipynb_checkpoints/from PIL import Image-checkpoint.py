img_dir = 'C:/Users/willy/OneDrive/桌面/專題/trashnet-master/data/dataset-resized/dataset-resized/metal'

import os
from PIL import Image

def rotate_and_save_images(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img = Image.open(os.path.join(input_folder, filename))
            for angle in [90, 180, 270, 360]:
                rotated_img = img.rotate(angle)
                new_filename = f"{os.path.splitext(filename)[0]}_{angle}.jpg"
                rotated_img.save(os.path.join(output_folder, new_filename))

# 使用方式
input_folder = 'C:/Users/willy/OneDrive/桌面/專題/trashnet-master/data/dataset-resized/dataset-resized/paper'
output_folder = 'C:/Users/willy/OneDrive/桌面/paper'
rotate_and_save_images(input_folder, output_folder)

