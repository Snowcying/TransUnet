import numpy as np
import cv2
from tqdm import tqdm
import random
import shutil
import os

npz_folder='/npz'
folder_path = os.path.dirname(os.path.abspath(__file__))+npz_folder
npz_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".npz")])
print(npz_files)


def save_filenames_to_txt(file_list, output_file):
    output_file=os.path.dirname(os.path.abspath(__file__))+'/'+output_file
    if not os.path.exists(output_file):
        open(output_file, 'w').close()
    
    with open(output_file, 'a') as f:
        for file_name in file_list:
            # 去除文件名的后缀
            base_name = file_name[:-4] if file_name.endswith('.npz') else file_name
            f.write(base_name + '\n')
    
    print(f"文件名已保存到 {output_file}")

# file_list = ['file1.npz', 'file2.npz', 'file3']
output_file = 'train.txt'
save_filenames_to_txt(npz_files, output_file)