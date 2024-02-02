import matplotlib.pyplot as plt
from PIL import Image

# 指定文件路径
image_path = './P00225518_233_mask.png'  # 替换为你的图像文件路径

# 使用Pillow打开图像
image = Image.open(image_path).convert('L')  # 转换为灰度图

# 使用matplotlib显示图像
plt.imshow(image, cmap='gray')
plt.axis('off')  # 隐藏坐标轴
plt.show()
