import random

import numpy as np
import matplotlib.pyplot as plt
import os
# current_path = os.path.abspath(__file__)
# print(current_path)
# 加载NPZ文件
# data = np.load('/home/cxy/paper/TransUnet/data/Mydata/train/npz/data_0.npz')
data = np.load('/home/cxy/paper/TransUnet/data/Mydata/train/npz_cropped/data_'+str(random.randint(0,100))+'.npz')
image_array = data['image']
label_array = data['label']
print(image_array.shape)
print(label_array.shape)

# 创建一个2x1的子图布局
fig, axs = plt.subplots(2, 1)

# 在第一个子图中显示原始图像
axs[0].imshow(image_array)
axs[0].set_title('Original Image')

# 在第二个子图中显示标签
axs[1].imshow(label_array)
axs[1].set_title('Label')

# 调整子图之间的间距
plt.tight_layout()

# 显示图像
plt.show()