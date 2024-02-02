import h5py
import numpy as np

file_path = './case0001.npy.h5'

file = h5py.File(file_path,'r')

image = file['image'][:]
label = file['label']

# print(image[:])