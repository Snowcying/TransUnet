import os
import shutil
from PIL import Image
import cv2
import numpy as np
import tqdm

# 定义原始文件夹和目标文件夹路径
source_folder = './x'
target_folder = './x_mask'

# 确保目标文件夹存在，如果不存在则创建
if not os.path.exists(target_folder):
    os.makedirs(target_folder)

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
            # 打开_mask图片并检查像素值
            mask_image = Image.open(mask_path)

            label = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

            # 将图像转换为灰度图像
            # gray_label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
            # 获取图像像素值的种类
            unique_values = np.unique(label)
            # mask_array = np.array(mask_image)
            # 检查_mask图片的最小像素值是否大于0
            if len(unique_values)>1:
                # 如果是，复制这一组图片到目标文件夹
                destination_mask_path = os.path.join(target_folder, filename)
                destination_image_path = os.path.join(target_folder, base_filename)
                # 复制文件
                shutil.copy2(image_path, destination_image_path)
                shutil.copy2(mask_path, destination_mask_path)
                print(f"Copied {base_filename} and {filename} to destination folder.")
#

# # 遍历原始文件夹中的文件
# for filename in os.listdir(source_folder):
#     if filename.endswith('.png'):
#         # 获取文件名和扩展名
#         name, extension = os.path.splitext(filename)
#
#         # 构建对应的mask文件名
#         mask_filename = f'{name}_mask{extension}'
#
#         # 构建完整的文件路径
#         image_path = os.path.join(source_folder, filename)
#         mask_path = os.path.join(source_folder, mask_filename)
#
#         label = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
#         unique_values = np.unique(label)
#
#         # 检查mask文件是否存在并且像素值最小值大于0
#         if os.path.exists(mask_path) and \
#                 (len(unique_values) > 1):
#             # 将图片复制到目标文件夹
#             shutil.copy(image_path, target_folder)
#             shutil.copy(mask_path, target_folder)