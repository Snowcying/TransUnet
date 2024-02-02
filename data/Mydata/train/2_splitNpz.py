import os
import numpy as np
import cv2
from tqdm import tqdm
import random
import shutil
folder_path = os.path.dirname(os.path.abspath(__file__))+'/npz'
train_path=folder_path+"/train"
test_path=folder_path+"/test"

npz_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".npz")])
if not os.path.exists(train_path):
    os.makedirs(train_path)
if not os.path.exists(test_path):
    os.makedirs(test_path)

print(len(npz_files))






def move_files(file_list, path1, path2):
    # 计算移动的文件数量
    num_files = len(file_list)
    num_files_to_move = int(0.8 * num_files)
    
    # 随机选择要移动的文件
    files_to_move = random.sample(file_list, num_files_to_move)
    
    # 移动文件到 path1
    for file_name in files_to_move:
        source_file = os.path.join(folder_path, file_name)
        target_file = os.path.join(path1, file_name)
        shutil.move(source_file, target_file)
    
    # 移动剩余的文件到 path2
    for file_name in file_list:
        if file_name not in files_to_move:
            source_file = os.path.join(folder_path, file_name)
            target_file = os.path.join(path2, file_name)
            shutil.move(source_file, target_file)
    
    # 保存 path1 中的文件名到 txt 文件
    save_filenames_to_txt(files_to_move, folder_path, 'train.txt')
    
    # 保存 path2 中的文件名到 txt 文件
    remaining_files = [file_name for file_name in file_list if file_name not in files_to_move]
    save_filenames_to_txt(remaining_files, folder_path, 'test.txt')

def save_filenames_to_txt(file_list, output_folder, output_file):
    # 检查输出文件夹是否存在，如果不存在则创建
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    output_path = os.path.join(output_folder, output_file)

    with open(output_path, 'w') as f:
        for file_name in file_list:
            # 去除文件名的后缀
            base_name = file_name[:-4] if file_name.endswith('.npz') else file_name
            f.write(base_name + '\n')

    print(f"文件名已保存到 {output_path}")

# 示例用法
file_list = ['file1.npz', 'file2.npz', 'file3.npz', 'file4.npz', 'file5.npz']
path1 = train_path
path2 = test_path

move_files(npz_files, path1, path2)


