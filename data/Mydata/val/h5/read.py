import random
import matplotlib.pyplot as plt

import h5py
import numpy as np

file_path = './P00225518.npy.h5'

file = h5py.File(file_path,'r')

image = file['image'][:]
label = file['label'][:]


index=random.randint(0,300)
print(index)
img1=image[index]
label1=label[index]
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
# 在第一个子图上显示第一个数组
axes[0].imshow(img1, cmap='gray')
axes[0].set_title('Image 1')
axes[0].axis('off')
# 在第二个子图上显示第二个数组
axes[1].imshow(label1, cmap='gray')
axes[1].set_title('Image 2')
axes[1].axis('off')
# 显示画布
plt.tight_layout()
plt.show()
