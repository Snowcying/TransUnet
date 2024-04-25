import random

import nibabel as nib
import numpy as np
from matplotlib import pyplot as plt
import SimpleITK as sitk
# 指定文件路径
img_path = 'TU_Synapse224/TU_pretrain_R50-ViT-B_16_skip3_epo31_bs4_224/P00225518_img.nii.gz'  # 替换为你的NIFTI文件路径
gt_path= 'TU_Synapse224/TU_pretrain_R50-ViT-B_16_skip3_epo31_bs4_224/P00225518_gt.nii.gz'
pred_path= 'TU_Synapse224/TU_pretrain_R50-ViT-B_16_skip3_epo31_bs4_224/P00225518_pred.nii.gz'

# 使用nibabel读取.nii.gz文件
nii = nib.load(img_path)


img = sitk.ReadImage(img_path)

data = sitk.GetArrayFromImage(img)
gt = sitk.GetArrayFromImage(sitk.ReadImage(gt_path))
pred = sitk.GetArrayFromImage(sitk.ReadImage(pred_path))
# data=np.transpose(data,(1,2,0))

selected_indices = np.random.choice(300, 9, replace=False)

# 根据选定的索引获取选定的二维数组
selected_arrays = data[selected_indices]
selected_gt = gt[selected_indices]
selected_pred = pred[selected_indices]

all=[data,gt,pred]

# 创建一个3x3的子图布局
fig, axs = plt.subplots(3, 3)

# 使用循环在每个子图中绘制选定的二维数组
for i in range(3):
    index=random.randint(0,300)
    for j in range(3):
        print(index)
        axs[i,j].imshow(all[j][index],cmap='gray')
        axs[i, j].axis('off')
    #     axs[i, j].imshow(selected_arrays[i * 3 + j], cmap='gray')  # 使用imshow函数显示数组
    #     axs[i, j].axis('off')  # 关闭坐标轴

# 调整子图之间的间距
plt.subplots_adjust(wspace=0.1, hspace=0.1)

# 显示图形
plt.show()

# print(img.shape)

# # 显示一些基本信息
# print(nii.get_sform())  # 获取空间变换矩阵
# print(nii.get_affine())  # 获取线性变换矩阵
# print(nii.header)  # 获取文件头信息
#
# # 如果需要，可以提取图像数据
# data = nii.get_fdata()
# print(data.shape)  # 显示数据形状
#
# # 显示图像（这里需要根据数据类型转换为合适的格式）
# i=random.randint(0,300)
# plt.imshow(data[i], cmap='gray')
# plt.axis('off')
# plt.show()
