# -*- coding: utf-8 -*-
"""
Created on Mon Jul 5 16:58:36 2021

@author: sophi
"""
#%%Attempts to import the data in different formats 
import pandas as pd
from matplotlib import pyplot as plt
import scipy as sp
import numpy as np
import glob
#%% Import text file data 
practise_data_1 = pd.read_csv('P3HT 58k 11 5um 0V_190208_Z Height_Forward_001.txt', sep = '\t')
#plt.plot(practise_data)
#practise_data 
practise_data_2 = pd.read_csv('P3HT 58k 11 5um 0V_190208_Z Height_Forward_001.xyz', sep = '\t')
print(type(practise_data_2))
print(practise_data_2)
print(practise_data_2.columns[2])
#practise_data_2.colums = ['x','y','z']
#plt.plot(x,y,z)
#%%Import tiff file with PIL Image
from PIL import Image
import numpy as np
im = Image.open('P3HT 58k 11 5um 0V_190208_Z Height_Forward_001.tiff')
im.show()
imarray = np.array(im)
print(imarray)
Final_Image = Image.fromarray(imarray)
Final_Image.show()
#%%Import with gdal
import gdal
image = gdal.Open('P3HT 58k 11 5um 0V_190208_Z Height_Forward_001.tiff')
array = image.ReadAsArray()
print(array)
height_data = image.geotiff_metadata()
print(height_data)
#%% import with tiff
from libtiff import TIFF
tiff = TIFF.open('P3HT 58k 11 5um 0V_190208_Z Height_Forward_001.tiff')
image = tiff.read_image()
#%% import with tifffile
from tifffile import tifffile as tf
image = tf.imread('P3HT 58k 11 5um 0V_190208_Z Height_Forward_001.tiff')
print(image)
metadata = tf.geotiff_metadata()