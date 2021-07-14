# -*- coding: utf-8 -*-
"""
Created on Tue Jul 6 11:52:40 2021

@author: sophi
"""
#Testing ideas

#%%
import gwyfile
#%%
import main 
p3ht = main.molecule(iz = 'P3HT 58k 11 5um 0V_190208_Z Height_Forward_001.tiff', ia = 'P3HT 58k 11 5um 0V_190208_EFM Amplitude_Forward_001.tiff', vz = 'P3HT 58k 11 5um EFM 2V_190208_Z Height_Forward_003.tiff', va = 'P3HT 58k 11 5um EFM 2V_190208_EFM Amplitude_Forward_003.tiff')
#p3ht.read()
#p3ht.bearray()
iz = p3ht.iz()
ia = p3ht.ia()
vz = p3ht.vz()
va = p3ht.va()

#%%Alpha Blending
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
im1 = Image.open(vz)
im1.show()
im1 = np.array(im1)
im1 = Image.fromarray(im1)
im2 = Image.open(va)
imarray = np.array(im2)
im2 = 255-imarray
im2 = Image.fromarray(im2)
im2.show()
print(type(im1))
print(type(im2))
#cm = plt.get_cmap()
#cm = plt.get_cmap()

image_blend = Image.blend(im1, im2, 0.5)
plt.imshow(image_blend)

#%%Plotting line graphs
x = list(range(1,257))
y1 = iz[155]
main.plt.plot(x,y1) # plot pixel 155 row
main.plt.show()
y2 = iz[:,155]
main.plt.plot(x,y2) # plot pixel 155 column
main.plt.show()
#%% Normalise to between 0 and 1
#Its also possible to give a Gaussian Distrubution if that's helpful 
#https://machinelearningmastery.com/how-to-manually-scale-image-pixel-data-for-deep-learning/
iz = p3ht.iz()
print('shape:',iz.shape)
print('Data Type: %s' % iz.dtype)
print('Min: %.3f, Max: %.3f' % (iz.min(), iz.max()))
iz = iz.astype('float32')
iz /= 255.0
print('Min: %.3f, Max: %.3f' % (iz.min(), iz.max()))
print(iz)
#%% Invert image 
vai = main.np.invert(va)
imagevai = main.Image.fromarray(vai)
imagevai.show()
#%% Error calculations
err4 = main.np.sum((va.astype("float") - vz.astype("float")) ** 2)
err4 /= float(va.shape[0] * vz.shape[1])
print(err4)
err5 = main.np.sum((vai.astype("float") - vz.astype("float")) ** 2)
err5 /= float(vai.shape[0] * vz.shape[1])
print(err5)
#%% Creating histograms
main.plt.hist(va)
main.plt.show()
main.plt.hist(vz)
main.plt.show()
main.plt.hist(vai)
main.plt.show()
#%%Attempt Mean Squared didn't work
err = main.np.sum((iz.astype("float") - vz.astype("float")) ** 2)
err /= float(iz.shape[0] * vz.shape[1])
print(err)
err2 = main.np.sum((ia.astype("float") - va.astype("float")) ** 2)
err2 /= float(ia.shape[0] * va.shape[1])
print(err2)
err3 = main.np.sum((ia.astype("float") - iz.astype("float")) ** 2)
err3 /= float(ia.shape[0] * iz.shape[1])
print(err3)
err4 = main.np.sum((va.astype("float") - vz.astype("float")) ** 2)
err4 /= float(va.shape[0] * vz.shape[1])
print(err4)
#%% currently is array of the average value for each array
print(iz)
print(len(iz))
def average(a,n):
    sum=0
    for i in range(n):
        sum += a[i]
    return sum/n
print(average(iz, len(iz)))

#%% attempt to find the hidden height data attributed but currently just gives the pixel value as tupple
import gdal
image = gdal.Open('P3HT 58k 11 5um 0V_190208_Z Height_Forward_001.tiff')
array = image.ReadAsArray()
geotransform = image.GetGeoTransform()
band = image.GetRasterBand(1)
scanline = band.ReadRaster(xoff=0, yoff=0,
                        xsize=band.XSize, ysize=1,
                        buf_xsize=band.XSize, buf_ysize=1,
                        buf_type=gdal.GDT_Float32)
import struct
tuple_of_floats = struct.unpack('f' * band.XSize, scanline)
print(tuple_of_floats)

