import os
import numpy as np
import cv2
from tqdm import tqdm

folder_path = os.path.dirname(os.path.abspath(__file__))
pic_path=folder_path+'/x_0pixel'
# pic_path=folder_path+'/x'

# out_folder="/npz"
out_folder="/npz_cropped"

image_files = []
label_files = []
# 遍历文件夹中的所有文件
for filename in os.listdir(pic_path):
    # 检查文件是否为PNG图片并且以_mask结尾
    if filename.endswith('_mask.png'):
        # 构建不带_mask的原始文件名
        base_filename = filename.replace('_mask.png', '.png')
        # 构建完整路径
        label_path = os.path.join(pic_path, filename)
        image_path = os.path.join(pic_path, base_filename)

        # 检查原始图片是否存在
        if os.path.exists(image_path):
            # 将文件路径添加到对应的列表中
            label_files.append(label_path)
            image_files.append(image_path)
# image_files = sorted([f for f in os.listdir(pic_path) if f.endswith(".png") and not f.endswith("_mask.png")])
# label_files = sorted([f for f in os.listdir(pic_path) if f.endswith("_mask.png")])
if not os.path.exists(folder_path+out_folder):
    os.makedirs(folder_path+out_folder)

i=0
for image_file, label_file in tqdm(zip(image_files, label_files), total=len(image_files)):
    # print(image_file,label_file)
    image_path = os.path.join(pic_path, image_file)
    label_path = os.path.join(pic_path, label_file)
    
    image = cv2.imread(image_path,cv2.IMREAD_UNCHANGED)
    label = cv2.imread(label_path,cv2.IMREAD_UNCHANGED)

    # 将图像转换为灰度图像
    # gray_label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
    # 获取图像像素值的种类
    unique_values = np.unique(label)
    # print("图像像素值的种类：", unique_values)
    if len(unique_values)>1:
        name=folder_path+out_folder+"/data_"+str(i)+".npz"
        i=i+1
        np.savez(name, image=image, label=label)
print(f"有{i}个npz文件")
# npz_files = [file for file in os.listdir(folder_path+'/npz') if file.endswith('.npz')]
# output_file=folder_path+'/npz/train.txt'
# # print(npz_files)
# def save_filenames_to_txt(file_list, output_file):
#     if not os.path.exists(output_file):
#         open(output_file, 'w').close()
#     with open(output_file, 'w') as f:
#         for file_name in tqdm(file_list,total=len(file_list)):
#             base_name = file_name[:-4] if file_name.endswith('.npz') else file_name
#             f.write(base_name + '\n')
#     print(f"文件名已保存到 {output_file}")
# save_filenames_to_txt(npz_files, output_file)

