import os
import h5py
from PIL import Image
import numpy as np
# 设置文件夹路径
folder_path = './x'  # 替换为你的文件夹路径
picSize1=480
picSize2=480

# 获取所有文件列表
files = os.listdir(folder_path)

# 创建一个字典来存储按name分组的文件
files_dict = {}

# 遍历文件，将它们按name分组
for file in files:
    # 检查文件是否以.png结尾
    if file.endswith('.png'):
        # 分割文件名以获取name和number
        name, number = file.split('_')[0], file.split('_')[1].split('.')[0]
        # 将文件添加到对应的name列表中
        if name not in files_dict:
            files_dict[name] = []
        files_dict[name].append((int(number), file))

# 对每个name的文件列表按number排序
for name in files_dict:
    files_dict[name].sort(key=lambda x: x[0])

# 创建HDF5文件并写入数据
for name, sorted_files in files_dict.items():
    # 创建HDF5文件
    with h5py.File(f'./h5/{name}.npy.h5', 'w') as h5f:
        # 创建两个dataset，一个用于图像，一个用于掩码
        images = h5f.create_dataset('image', (len(sorted_files)/2, picSize1, picSize2), dtype='uint8')
        labels = h5f.create_dataset('label', (len(sorted_files)/2, picSize1, picSize2), dtype='uint8')
        # 遍历排序后的文件列表，读取图像并写入HDF5文件
        i=0
        j=0
        for k,(number, file) in enumerate(sorted_files):
            # 读取图像
            if 'mask' in file:
                img = Image.open(os.path.join(folder_path, file))
                labels[i, :, :] = np.array(img)
                i=i+1
            else:
                img = Image.open(os.path.join(folder_path, file))
                images[j, :, :] = np.array(img)
                j=j+1

# 提示完成
print("HDF5 files created successfully.")
