import cv2
import numpy as np
import os
import h5py
from matplotlib import pyplot as plt

# image = cv2.imread("/home/cxy/paper/TransUnet/data/Mydata/train/x/P00176954_0.png")



# def get_cv2_png_shape(png_file):
#     # 使用cv2.imread加载PNG图像
#     image = cv2.imread(png_file, cv2.IMREAD_UNCHANGED)
    
#     # 获取图像形状
#     height, width, channels = image.shape
    
#     return width, height, channels

# # 示例用法
# png_file = '/home/cxy/paper/TransUnet/data/Mydata/train/x/P00176954_0_mask.png'
# width, height, channels = get_cv2_png_shape(png_file)
# print("宽度:", width)
# print("高度:", height)
# print("通道数:", channels)
folder_path = os.path.dirname(os.path.abspath(__file__))
image_path='/home/cxy/paper/TransUnet/data/Mydata/train/x/P00176954_0.png'
label_path='/home/cxy/paper/TransUnet/data/Mydata/train/x/P00176954_0_mask.png'
image = cv2.imread(image_path,cv2.IMREAD_UNCHANGED)
label = cv2.imread(label_path,cv2.IMREAD_UNCHANGED)


# gray_label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
unique_values = np.unique(label)
print(unique_values)
# np.savez(folder_path+'/1.npz', image=image, label=label)



def get_npz_array_shape(npz_file):
    # 加载npz文件
    data = np.load(npz_file)
    
    # 获取数组的shape
    shapes = []
    for array_name in data.files:
        array = data[array_name]
        shapes.append(array.shape)
    
    return shapes

def plot_image(array):
    # 绘制图像
    plt.imshow(array, cmap='gray')
    plt.axis('off')  # 关闭坐标轴显示
    plt.show()

def get_array_h5(h5_file):
    
    data = h5py.File(h5_file)
    image, label = data['image'][:], data['label'][:]
    pic=image[0]
    plot_image(pic)
    print(image)
    

# 示例用法
# npz_file = '/home/cxy/paper/TransUnet/data/Mydata/train/1.npz'
# array_shapes = get_npz_array_shape(npz_file)
# print(array_shapes)


h5_file= '/home/cxy/paper/TransUnet/data/Synapse/test_vol_h5/case0001.npy.h5'
get_array_h5(h5_file)