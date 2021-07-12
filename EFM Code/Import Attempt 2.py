# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 11:52:40 2021

@author: sophi
"""

#%%Import tiff file with PIL Image
from PIL import Image
import numpy as np

im = Image.open('P3HT 58k 11 5um 0V_190208_Z Height_Forward_001.tiff')
#%%im.show()
imarray = np.array(im)
print(imarray)
Final_Image = Image.fromarray(imarray)
#Final_Image.show()
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
#%%
from tifffile import tifffile as tf
image = tf.imread('P3HT 58k 11 5um 0V_190208_Z Height_Forward_001.tiff')
print(image)
metadata = tf.geotiff_metadata()
