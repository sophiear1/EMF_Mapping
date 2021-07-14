# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 14:51:59 2021

@author: sophi
"""

#%%
import main 
p3ht = main.molecule(iz = 'P3HT 58k 11 5um 0V_190208_Z Height_Forward_001.tiff', ia = 'P3HT 58k 11 5um 0V_190208_EFM Amplitude_Forward_001.tiff', vz = 'P3HT 58k 11 5um EFM 2V_190208_Z Height_Forward_003.tiff', va = 'P3HT 58k 11 5um EFM 2V_190208_EFM Amplitude_Forward_003.tiff')
#p3ht.read()
#p3ht.bearray()
iz = p3ht.iz()
ia = p3ht.ia()
vz = p3ht.vz()
va = p3ht.va()

#%%Edge enhancement using Pillow
#https://pythontic.com/image-processing/pillow/edge-enhancement-filter
#https://pythontic.com/image-processing/pillow/edge-detection
from PIL import Image
from PIL import ImageFilter
import skimage
import skimage.feature
import skimage.viewer
imageObject = Image.open(vz)
imageObject = imageObject.convert('RGB') # need to convert toe RGB file for enhancement but I think this might make it harder to find edges so defeats the point of the enhancement
# Apply edge enhancement filter
edgeEnhanced = imageObject.filter(ImageFilter.EDGE_ENHANCE)
# Apply increased edge enhancement filter
moreEdgeEnhanced = imageObject.filter(ImageFilter.EDGE_ENHANCE_MORE)
# Show original image - before applying edge enhancement filters
imageObject.show() 
# Show image - after applying edge enhancement filter
edgeEnhanced.show()
# Show image - after applying increased edge enhancement filter
moreEdgeEnhanced.show()
moreEdgeEnhanced.save('Edge_enhanced_rgb.tiff')

#%% Issue is though that I'm not sure its sensitive enough for less obvious data  
#plus only opens one at a time 

"""
https://datacarpentry.org/image-processing/08-edge-detection/
https://towardsdatascience.com/canny-edge-detection-step-by-step-in-python-computer-vision-b49c3a2d8123
https://github.com/FienSoP/canny_edge_detector
 * Python script to demonstrate Canny edge detection
 * with sliders to adjust the thresholds.
 *
 * usage: python CannyTrack.py <'P3HT 58k 11 5um 0V_190208_Z Height_Forward_001.tiff'>
"""
import skimage
import skimage.feature
import skimage.viewer


filename = 'Edge_enhanced_rgb.tiff'
image = skimage.io.imread(fname=filename, as_gray=True)
viewer = skimage.viewer.ImageViewer(image)

# Create the plugin and give it a name
canny_plugin = skimage.viewer.plugins.Plugin(image_filter=skimage.feature.canny)
canny_plugin.name = "Canny Filter Plugin"

# Add sliders for the parameters
canny_plugin += skimage.viewer.widgets.Slider(
    name="sigma", low=0.0, high=7.0, value=2.0
)
canny_plugin += skimage.viewer.widgets.Slider(
    name="low_threshold", low=0.0, high=1.0, value=0.1
)
canny_plugin += skimage.viewer.widgets.Slider(
    name="high_threshold", low=0.0, high=1.0, value=0.2
)

# add the plugin to the viewer and show the window
viewer += canny_plugin
viewer.show()
#%%Sections https://www.youtube.com/watch?v=d1CIV9irQAY
import numpy as np
import matplotlib.pyplot as plt

import skimage
import skimage.feature
import skimage.viewer
import skimage.data as data
import skimage.segmentation as seg
from skimage import filters
from skimage import draw
from skimage import color
from skimage import exposure

#Function to display image throughout 
def image_show(image, nrows=1, ncols=1, cmap='gray', **kwargs):
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(16, 16))
    ax.imshow(image, cmap='gray')
    ax.axis('off')
    return fig, ax
#load in file
filename = vz
image = skimage.io.imread(fname=filename, as_gray=True)
image_show(image)
#histogram of pixels to help decided the threshold value
#image = image*255
fig, ax = plt.subplots(1,1)
ax.hist(image.ravel(), bins=256, range=[0,1])
ax.set_xlim(0,1)
threshold = 0.5 # or
# can use filter to determine threshold point e.g.filters.threshold_sauvola(image)
image_show(image > threshold)
#%%Flood Fill - experiment with seed point, tolerance and alpha 
#high alpha seems to split into 3 different regions
seed_point = (100, 220)
flood_mask = seg.flood(image, seed_point, tolerance = 0.3)
fig,ax = image_show(image)
ax.imshow(flood_mask,alpha = -0.5)

#%%Chan-Vese, seems quite good
filename = vz
image = skimage.io.imread(fname=filename, as_gray=True)
image_show(image)
chan_vese = seg.chan_vese(image)

fig, ax = image_show(image)
ax.imshow(chan_vese == 0, alpha=-0.3);

filename = va
image2 = skimage.io.imread(fname=filename, as_gray=True)
chan_vese2 = seg.chan_vese(image2)
fig, ax = image_show(image2)
ax.imshow(chan_vese2 == 0, alpha=-0.05);
#adjust and save image such that maybe do edge detction after this fill function?
#%%Difference filter in 2D
import scipy.ndimage as ndi
vertical_kernel = np.array([
    [-1],
    [ 0],
    [ 1],
])
    
filename = vz
image = skimage.io.imread(fname=filename, as_gray=True)
gradient_vertical = ndi.correlate(image.astype(float),
                                  vertical_kernel)
fig, ax = plt.subplots()
ax.imshow(gradient_vertical);
#%%Sobel Edge Filter
from skimage import img_as_float

def imshow_all(*images, titles=None):
    images = [img_as_float(img) for img in images]

    if titles is None:
        titles = [''] * len(images)
    vmin = min(map(np.min, images))
    vmax = max(map(np.max, images))
    ncols = len(images)
    height = 5
    width = height * len(images)
    fig, axes = plt.subplots(nrows=1, ncols=ncols,
                             figsize=(width, height))
    for ax, img, label in zip(axes.ravel(), images, titles):
        ax.imshow(img, vmin=vmin, vmax=vmax)
        ax.set_title(label)
imshow_all(image, filters.sobel(image))
pixelated_gradient = filters.sobel(image)
imshow_all(image, pixelated_gradient*1)

gradient = filters.sobel(image)
titles = ['gradient before smoothing', 'gradient after smoothing']
# Scale smoothed gradient up so they're of comparable brightness.
imshow_all(pixelated_gradient, gradient*1.8, titles=titles)
#%%Denoising 
from skimage.morphology import disk
neighborhood = disk(radius=1)  # "selem" is often the name used for "structuring element"
median = filters.rank.median(image, neighborhood)
titles = ['image', 'gaussian', 'median']
imshow_all(image, gradient*5, median, titles=titles)
