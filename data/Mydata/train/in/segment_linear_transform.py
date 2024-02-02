import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# 加载图像并转换为灰度图
image_path = 'in.png'  # 替换为你的图片路径
image = Image.open(image_path).convert('L')
image_array = np.array(image)

# 定义阈值
threshold = 10

# 将低于阈值的像素赋值为0
transformed_array = np.where(image_array < threshold, 0, image_array)

# 将变换后的数组转换为图像
transformed_image = Image.fromarray(transformed_array.astype(np.uint8))

# 使用matplotlib显示原图和变换后的图像
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(image, cmap='gray')
axes[0].axis('off')
axes[0].set_title('Original Image')
axes[1].imshow(transformed_image, cmap='gray')
axes[1].axis('off')
axes[1].set_title('Thresholded Image')
plt.tight_layout()

transformed_image.save('out.png')
plt.show()
