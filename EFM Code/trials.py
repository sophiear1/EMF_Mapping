# -*- coding: utf-8 -*-
"""
Created on Tue Jul 6 11:52:40 2021

@author: sophi
"""
#Testing ideas

#%%
import main 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
p3ht = main.molecule(iz = 'P3HT 58k 11 5um 0V_190208_Z Height_Forward_001.tiff', ia = 'P3HT 58k 11 5um 0V_190208_EFM Amplitude_Forward_001.tiff', vz = 'P3HT 58k 11 5um EFM 2V_190208_Z Height_Forward_003.tiff', va = 'P3HT 58k 11 5um EFM 2V_190208_EFM Amplitude_Forward_003.tiff')
#p3ht.read()
#p3ht.bearray()
iz = p3ht.iz()
ia = p3ht.ia()
vz = p3ht.vz()
va = p3ht.va()

#%%Convert to Float
from skimage import img_as_float
from skimage import io
iz = io.imread(iz)
print(iz)
iz_float = img_as_float(iz)
print(iz_float)
#%%Visual image comparison Trial 1 = Difference 
from matplotlib.gridspec import GridSpec
from skimage import data, transform, exposure
#from skimage.util import compare
from skimage import io
import numpy as np
import skimage.segmentation as seg

def boolstr_to_floatstr(v):
    if v == 'True':
        return '1'
    elif v == 'False':
        return '0'
    else:
        return v

filename = vz # file
im_vz = io.imread(fname=filename, as_gray = True)
im_vz = seg.chan_vese(im_vz)
im_vz = np.vectorize(boolstr_to_floatstr)(im_vz).astype(float)
filename = va # file
im_va = io.imread(fname=filename, as_gray = True)
im_va = 1 - im_va
im_va = seg.chan_vese(im_va)
im_va = np.vectorize(boolstr_to_floatstr)(im_va).astype(float)


#%%
def diff_f(im1, im2):
    comparison = np.abs(im2-im1)
    return comparison

diff = diff_f(im_vz, im_va)
fig = plt.figure(figsize=(8, 9))

gs = GridSpec(3, 2)
ax0 = fig.add_subplot(gs[0, 0])
ax1 = fig.add_subplot(gs[0, 1])
ax2 = fig.add_subplot(gs[1:, :])

ax0.imshow(im_vz, cmap='gray')
ax0.set_title('vz')
ax1.imshow(im_va, cmap='gray')
ax1.set_title('va')
ax2.imshow(diff, cmap='gray')
ax2.set_title('Diff comparison')
for a in (ax0, ax1, ax2):
    a.axis('off')
plt.tight_layout()
plt.plot()
 
#%%Comparison blend
def blend_f(im1, im2):
    comparison = 0.5 * (im2 + im1)
    return comparison
blend = blend_f(im_vz, im_va)
fig = plt.figure(figsize=(8, 9))

gs = GridSpec(3, 2)
ax0 = fig.add_subplot(gs[0, 0])
ax1 = fig.add_subplot(gs[0, 1])
ax2 = fig.add_subplot(gs[1:, :])

ax0.imshow(im_vz, cmap='gray')
ax0.set_title('Original')
ax1.imshow(im_va, cmap='gray')
ax1.set_title('Rotated')
ax2.imshow(blend, cmap='gray')
ax2.set_title('Blend comparison')
for a in (ax0, ax1, ax2):
    a.axis('off')
plt.tight_layout()
plt.plot()
#%%Colour map changing 
import skimage
from skimage import color
from skimage import io
filename = vz # file
image = io.imread(fname=filename)
plt.imshow(image)
image = color.convert_colorspace(image,'RGB','HSV')
plt.imshow(image)
print(type(image))
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
plt.imshow(im2, cmap = plt.cm.colors)
image_blend = Image.blend(im1, im2, 0.5)
plt.imshow(image_blend)

#%%Plotting line graphs
x = list(range(0,256))
y1 = iz[155]
main.plt.plot(x,y1) # plot pixel 155 row
main.plt.show()
y2 = iz[:,155]
main.plt.plot(x,y2) # plot pixel 155 column
main.plt.show()

#%% Normalise to between 0 and 1
#Its also possible to give a Gaussian Distrubution if that's helpful 
#https://machinelearningmastery.com/how-to-manually-scale-image-pixel-data-for-deep-learning/
p3ht.bearray()
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

import exifread
f = open(va, 'rb')
tags = exifread.process_file(f)
for tag in tags.keys():
    if tag not in ('JPEGThumbnail', 'TIFFThumbnail', 'Filename', 'EXIF MakerNote'):
        print("Key: %s, value %s" % (tag, tags[tag]))
#%%
from raster2xyz.raster2xyz import Raster2xyz

input_raster = vz
out_csv = "extraction_attempt.xyz"

rtxyz = Raster2xyz()
rtxyz.translate(input_raster, out_csv)

