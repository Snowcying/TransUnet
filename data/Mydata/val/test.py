import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# 指定文件路径
# i=233
# image_path = './P00225518_'+str(i)+'_mask.png'  # 替换为你的图像文件路径

for i in range(0,300):
    image_path = './x/P00225518_' + str(i) + '_mask.png'  # 替换为你的图像文件路径
    image = Image.open(image_path).convert('L')
    img = np.array(image)
    unique=np.unique(img)
    indice=img!=3
    img[indice]=0
    if 3 in unique:
        # print(img)
        plt.imshow(img,cmap='gray')
        plt.axis('off')
        plt.show()
# # 使用Pillow打开图像
# image = Image.open(image_path).convert('L')  # 转换为灰度图
#
# img=np.array(image)
# print(np.unique(img))
# # 使用matplotlib显示图像
# plt.imshow(image, cmap='gray')
# plt.axis('off')  # 隐藏坐标轴
# plt.show()
