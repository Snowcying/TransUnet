from PIL import Image
import numpy as np
import tqdm
import os
import shutil
from PIL import Image
import cv2
import numpy as np

def transform_low_pixel(input_image_path):
    # image_path = 'in.png'  # 替换为你的图片路径
    img = Image.open(input_image_path).convert('L')
    image_array = np.array(img)
    # 定义阈值
    threshold = 10
    # 将低于阈值的像素赋值为0
    transformed_array = np.where(image_array < threshold, 0, image_array)

    return crop_image(transformed_array)

    # return  transformed_array


def crop_image(image_array):
    # 获取图像的行和列
    rows, cols = image_array.shape

    # 初始化边界
    top = rows
    bottom = 0
    left = cols
    right = 0

    # 遍历图像以找到物体的边界
    for row in range(rows):
        for col in range(cols):
            # 如果找到非零像素，则更新边界
            if image_array[row, col] != 0:
                if row < top:
                    top = row
                if row > bottom:
                    bottom = row
                if col < left:
                    left = col
                if col > right:
                    right = col
    # 裁剪图像
    # cropped_image = image_array[top:bottom + 1, left:right + 1]
    # return cropped_image
    return top,bottom+1,left,right+1


source_folder='./x_mask'
target_folder='./x_0pixel'


for filename in os.listdir(source_folder):
    # 检查文件是否为PNG图片并且以_mask结尾
    if filename.endswith('_mask.png'):
        # 构建不带_mask的原始文件名
        base_filename = filename.replace('_mask.png', '.png')
        # 构建完整路径
        mask_path = os.path.join(source_folder, filename)
        image_path = os.path.join(source_folder, base_filename)

        # 检查原始图片是否存在
        if os.path.exists(image_path):
            img = Image.open(image_path).convert('L')
            mask=Image.open(mask_path).convert('L')
            image_array = np.array(img)
            mask_array=np.array(mask)
            # 定义阈值
            threshold = 10
            # 将低于阈值的像素赋值为0
            transformed_array = np.where(image_array < threshold, 0, image_array)
            top,bottom,left,right=crop_image(transformed_array)

            cropped_image = image_array[top:bottom, left:right]
            cropped_mask=mask_array[top:bottom,left:right]

            Image.fromarray(cropped_image).save(os.path.join(target_folder,base_filename))
            Image.fromarray(cropped_mask).save(os.path.join(target_folder,filename))
#

# # 加载图像
# image_path = 'in/out.png'  # 替换为您的图像路径
# cropped_image_array=transform_low_pixel(image_path)
# cropped_image = Image.fromarray(cropped_image_array)
# # 保存裁剪后的图像
# cropped_image.save('cropped_image.png')

