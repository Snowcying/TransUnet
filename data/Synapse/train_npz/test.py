import numpy as np
import matplotlib.pyplot as plt
import os
# current_path = os.path.abspath(__file__)
# print(current_path)
# 加载NPZ文件
data = np.load('/home/cxy/paper/TransUnet/data/Synapse/train_npz/case0040_slice148.npz')
image_array = data['image']
label_array = data['label']

print(np.unique(image_array))
print(np.unique(label_array))
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