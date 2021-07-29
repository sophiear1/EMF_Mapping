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
import tifffile


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
from skimage import filters, img_as_float
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
image = img_as_float(image)
image_show(image)
#histogram of pixels to help decided the threshold value
#image = image*255
fig, ax = plt.subplots(1,1)
ax.hist(image.ravel(), bins=256, range=[0,1])
ax.set_xlim(0,1)
threshold = 0.6 # or
# can use filter to determine threshold point e.g.filters.threshold_sauvola(image)
fig, ax = plt.subplots(1,1)
ax.hist(image.ravel(), bins=256, range=[0.6,1])
ax.set_xlim(0,1)
#%%More Segementation
import scipy.ndimage as nd
image = skimage.io.imread(va, as_gray = True)
image = img_as_float(image)
segm1 = (image <= 0.25)
segm2 = (image > 0.25) & (image <= 0.5)
segm3 = (image > 0.5) & (image <= 0.75)
segm4 = (image > 0.75) 
all_segments = np.zeros((image.shape[0], image.shape[1], 3))
all_segments[segm1] = (1,0,0) # red
all_segments[segm2] = (0,1,0) # green
all_segments[segm3] = (0,0,1) # blue
all_segments[segm4] = (1,1,0) #yellow
plt.imshow(all_segments)
#%%Clean up a bit
segm = [segm1, segm2, segm3, segm4]
all_segm = []
for i in range(0,4):
    segm_opened = nd.binary_opening(segm[i], np.ones((3,3)))
    segm_closed = nd.binary_closing(segm_opened, np.ones((3,3)))
    all_segm.append(segm_closed)
all_segmmments = np.zeros((image.shape[0], image.shape[1], 3))
all_segments[all_segm[0]] = (1,0,0) # red
all_segments[all_segm[1]] = (0,1,0) # green
all_segments[all_segm[2]] = (0,0,1) # blue
all_segments[all_segm[3]] = (1,1,0) #yellow
plt.imshow(all_segments)
print(image)
#%%More Segementation
from skimage.segmentation import random_walker
from skimage.exposure import rescale_intensity
import skimage
filename = vz
image = skimage.io.imread(fname=filename, as_gray=True)
image = rescale_intensity(image, in_range=(0,256), out_range=(-1,1))
markers = np.zeros(image.shape, dtype=np.uint)
markers[image<-0.2]=1
markers[image>0.2] =2
labels = random_walker(image, markers, beta=10, mode='bf')

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8, 3.2),
                                    sharex=True, sharey=True)
ax1.imshow(image, cmap='gray')
ax1.axis('off')
ax1.set_title('Noisy data')
ax2.imshow(markers, cmap='magma')
ax2.axis('off')
ax2.set_title('Markers')
ax3.imshow(labels, cmap='gray')
ax3.axis('off')
ax3.set_title('Segmentation')

fig.tight_layout()
plt.show()

#%%Flood Fill - experiment with seed point, tolerance and alpha 
#high alpha seems to split into 3 different regions
filename = vz
image = img_as_float(skimage.io.imread(fname=filename, as_gray=True))
seed_point = (100, 220)
flood_mask = seg.flood(image, seed_point, tolerance = 0.3)
fig,ax = image_show(image, cmap = 'gray')
ax.imshow(flood_mask,alpha = 0.5, cmap = 'gray')

#%%Chan-Vese, seems quite good
filename = vz
image = skimage.io.imread(fname=filename, as_gray=True)
image_show(image)
chan_vese = seg.chan_vese(image)
fig, ax = image_show(image)
ax.imshow(chan_vese == 0, alpha=-0.3, cmap='gray');
filename = va
image2 = skimage.io.imread(fname=filename, as_gray=True)
image_show(image2)
chan_vese2 = seg.chan_vese(image2)
fig, ax = image_show(image2)
ax.imshow(chan_vese2 == 0, alpha=2, cmap = 'gray');#-0.05);
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
#image = 1-image
gradient_vertical = ndi.correlate(image.astype(float),
                                  vertical_kernel)
fig, ax = plt.subplots()
ax.imshow(gradient_vertical, cmap = 'gray');
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
        ax.imshow(img, vmin=vmin, vmax=vmax, cmap = 'gray')
        ax.set_title(label)
imshow_all(image, filters.sobel(image))
pixelated_gradient = filters.sobel(image)
imshow_all(image, pixelated_gradient*1)

gradient = filters.sobel(image)
titles = ['gradient before smoothing', 'gradient after smoothing']
# Scale smoothed gradient up so they're of comparable brightness.
imshow_all(pixelated_gradient, gradient*1.8, titles=titles)
#%%Denoising  with median filter
from skimage.morphology import disk
neighborhood = disk(radius=10)  # "selem" is often the name used for "structuring element"
median = filters.rank.median(image, neighborhood)
titles = ['image', 'gaussian', 'median']
imshow_all(image, gradient*5, median, titles=titles)

#%%Denoising with median filter
from scipy import ndimage as ndi
from skimage import util
denoised = ndi.median_filter(util.img_as_float(image), size = 0.5)
plt.imshow(denoised)
#%%Denoising with Gaussian
gaussian = ndi.gaussian_filter(image, sigma =3)
plt.imshow(gaussian)
from skimage.restoration import denoise_nl_means, estimate_sigma
sigma_est = np.mean(estimate_sigma(image, multichannel=True))
nlm = denoise_nl_means(image, h=1.5*sigma_est, fast_mode = True, patch_size=5, patch_distance=3, multichannel=True)
plt.imshow(nlm)
#%%
from skimage import img_as_float, img_as_ubyte, io
image=img_as_float(image)
denoise_ubyte = img_as_ubyte(nlm)
hist = plt.hist(denoise_ubyte, bins=50, range = (0,255))
plt.show(hist)
#%%https://www.youtube.com/watch?v=jyYl9n5ow-Q&list=PLHae9ggVvqPgyRQQOtENr6hK0m1UquGaG&index=50
#Apeer_Micro Course Follow Through 
import cv2 
import skimage
from skimage import io, img_as_float, img_as_ubyte, color, filters, feature, measure, exposure, util, 
from skimage.transform import rescale,resize, downscale_local_mean
from skimage.filters import gaussian, sobel, unsharp_mask, median, roberts, scharr, prewitt, threshold_multiotsu, threshold_otsu
from skimage.morphology import disk
from skimage.restoration import denoise_bilateral, denoise_nl_means, estimate_sigma, denoise_tv_chambolle
from skimage.color import rgb2gray, label2rgb
from skimage.segmentation import clear_border
import numpy as np
import cv2
import tifffile 
from matplotlib import pyplot as plt
import glob
import os
import pandas as pd 
#%%reading image
image = skimage.io.imread(vz, as_gray=True) # read in image
print(image) # show array
image_float = img_as_float(image) #convert from 8bit to float in array
image_8bit = img_as_ubyte(image_float) #convert from float to 8bit
gray_image = cv2.imread(vz,0)
color_image = cv2.imread(vz,1)
print(type(gray_image))
print(type(image)) # both are numpy arrays so I think methods from opencv and skimage will both work
image1 = tifffile.imread(vz)
print(np.shape(image1))
#%%saving file
io.imsave(wanted_filename, variable)#if tiff uses tifffile in background as below # better as 8 bit than float
cv2.imwrite(wanted_filename, variable)#looks better is not float
plt.imsave(wanted_filename, variable) # for gray scale defne colormap 
tifffile.imwrite(wanted_filename, variable)
#%%converting colors
RGB = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
#%%Display Images
io.imshow(image)
plt.imshow(image) # can change color map e.g. cmap='hot'
cv2.imshow('window name', image)
cv2.waitKey(0)   # need this waitkey I'm not sure why
cv2.destroyAllWindows()
#%%Plotting with plt
plt.imshow(gray_image, cmap = 'gray') 
#plt.hist(gray_image, bins=10, range=(0,255))
#takes forever
#plot 3 images horizontally
plt.figure(figsize = (16,6))
plt.subplot(131)
plt.imshow(image)
plt.subplot(132)
plt.imshow(gray_image)
plt.subplot(133)
plt.imshow(color_image)
plt.show()
#plot 3 images vertically 
plt.figure(figsize=(6, 16))
plt.subplot(311) 
plt.imshow(image)
plt.subplot(312)
plt.imshow(gray_image)
plt.subplot(313)
plt.imshow(color_image)
plt.show()
#plt as grid
plt.figure(figsize=(12, 12))
plt.subplot(221) 
plt.imshow(image)
plt.subplot(222)
plt.imshow(gray_image)
plt.subplot(223)
plt.imshow(color_image)
plt.show()
#Another Method
fig = plt.figure(figsize =(16,6))
ax1=fig.add_subplot(221)
ax1.set(title = 'image')
ax2=fig.add_subplot(222)
ax2.set(title='gray_scale')
ax3=fig.add_subplot(223)
ax3.set(title='color_map')
ax4=fig.add_subplot(224)
ax4.set(title='other')
ax1.imshow(image)
ax2.imshow(gray_image)
ax3.imshow(color_image)
plt.show()
#%% Reading in multiple files with glob and os
file_list = glob.glob('*.tiff*')
print(file_list)
path = '*.tiff*'
my_list = []
for file in glob.glob(path):
    print(file)
    a = cv2.imread(file)
    my_list.append(a)
plt.imshow(my_list[0])
#%% Image processing with scikit-image
image = io.imread(vz, as_gray = True)
img_rescale = rescale(image, 1.0 / 4.0, anti_aliasing=False)
img_resized = resize(image, (200,200), anti_aliasing=True)
img_downscaled = downscale_local_mean(image, (2,3))
plt.figure(figsize=(12, 12))
plt.subplot(221) 
plt.imshow(image)
plt.subplot(222)
plt.imshow(img_rescale)
plt.subplot(223)
plt.imshow(img_resized)
plt.subplot(224)
plt.imshow(img_downscaled)
plt.show()
gaussian_using_skimage = gaussian(image, sigma=1, mode = 'constant', cval = 0.0)
plt.imshow(gaussian_using_skimage)
sobel_img = sobel(image) # remember only works with gray 
plt.imshow(sobel_img, cmap = 'gray')
#%% Image processiong with OpenCV
image=cv2.imread(vz, 1) #import as BGR
resized = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
cv2.imshow('original pic', image)
cv2.imshow('resized pic', resized)
cv2.waitKey(0)
cv2.destroyAllWindows()
image_grey=cv2.imread(vz,0)
print(image.shape)
print('Top left', image[0,0])
print("Top right", image[0, 255]) 
print("Bottom Left", image[255, 0]) 
print("Bottom right", image[255, 255])
blue = image[:,:,0]
green = image[:,:,1]
red = image[:,:,2]
b,g,r = cv2.split(image)
image_merged = cv2.merge((b,g,r))
cv2.imshow('red pic', red)
cv2.imshow('color pic', image)
cv2.imshow('green pic', g)
cv2.imshow('blue pic', b)
cv2.imshow('image merged', image_merged)
cv2.waitKey(0)
cv2.destroyAllWindows()   
edges = cv2.Canny(image, 0, 255)
cv2.imshow('orginal', image)
cv2.imshow('canny', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()

#%%Sharpening using unsharp mask
image = io.imread(vz, as_gray=True)
#Radius affects blurring
#Amount affects multiplication factor for original-blurred
unsharped_image = unsharp_mask(image,radius=3,amount=2)
plt.figure(figsize=(12, 6))
plt.subplot(121) 
plt.imshow(image, cmap='gray')
plt.subplot(122)
plt.imshow(unsharped_image, cmap = 'gray')
plt.show()
#%%Denoising with Gaussian
image = io.imread(iz)
image = img_as_float(image)
#array = np.array([0.0])
gaussian_using_cv2 = cv2.GaussianBlur(image, (3,3), 0, borderType=cv2.BORDER_CONSTANT)
gaussian_using_skimage = gaussian(image, sigma=1)#, mode='constant', cval=0.0)
plt.figure(figsize = (16,6))
plt.subplot(131)
plt.imshow(image)
plt.subplot(132)
plt.imshow(gaussian_using_cv2)
plt.subplot(133)
plt.imshow(gaussian_using_skimage)
plt.show()
#%%Denoising with Median
image = img_as_ubyte(image)
median_using_cv2 = cv2.medianBlur(image,3)
image = img_as_float(image)
median_using_skimage = median(image, disk(1), mode='constant', cval=0.0)
plt.figure(figsize = (16,6))
plt.subplot(131)
plt.imshow(image)
plt.subplot(132)
plt.imshow(median_using_cv2)
plt.subplot(133)
plt.imshow(median_using_skimage)
plt.show()
#%%Denoising with bilateral
image = img_as_ubyte(image)
bilateral_using_cv2 = cv2.bilateralFilter(image,5,20,100, borderType=cv2.BORDER_CONSTANT)
image = img_as_float(image)
bilateral_using_skimage = denoise_bilateral(image, sigma_color=0.05, sigma_spatial=1.5, multichannel=False)
plt.figure(figsize = (16,6))
plt.subplot(131)
plt.imshow(image)
plt.subplot(132)
plt.imshow(bilateral_using_cv2)
plt.subplot(133)
plt.imshow(bilateral_using_skimage)
plt.show()
#%%Denoising with NLM
sigma_est = np.mean(estimate_sigma( image, multichannel=True))
nlm_using_skimage = denoise_nl_means(image, h=1.15*sigma_est, fast_mode=True, patch_size=5, patch_distance=3, multichannel=False)
plt.figure(figsize = (12,6))
plt.subplot(121)
plt.imshow(image)
plt.subplot(122)
plt.imshow(nlm_using_skimage)
plt.show()
#%%Denoise using Total Variance 
imgage = img_as_float(io.imread(vz, as_gray=True))
plt.hist(image.flat, bins=100)#, range=(0,1))
plt.show()
denoise_variance = denoise_tv_chambolle(imgage, weight=0.1, eps=0.0002, n_iter_max=200, multichannel=False)
plt.hist(denoise_variance.flat, bins=100)#, range=(0,1))  
plt.show()
plt.figure(figsize = (12,6))
plt.subplot(121)
plt.imshow(image)
plt.subplot(122)
plt.imshow(denoise_variance)
plt.show()
#%%Edge Detection Filters
#Roberts = Approximate gradient between diagonally adjacent = highlights high spatial gradient
#Sobel = Similar to Roberts but with horizontal/vertical kernels instead of diagonal
#Scharr = Similar to Sobel but x and y are independent 
#Prewitt = Faster than Sobel
#canny - can be automated or not automated depending similar to above 
image = cv2.imread(va, 0)
roberts_im = roberts(image)
sobel_im = sobel(image)
scharr_im = scharr(image)
prewitt_im = prewitt(image)
canny_manual_im = cv2.Canny(image, 150,150)
sigma = 1.5
median = np.median(image)
lower = int(max(0, (1-sigma)*median))
upper = int(min(255, (1+sigma)*median))
canny_auto_im = cv2.Canny(image, lower, upper)
plt.figure(figsize = (24,12))
plt.subplot(231)
plt.imshow(image, cmap = 'gray')
plt.subplot(232)
plt.imshow(roberts_im, cmap = 'gray')
plt.subplot(233)
plt.imshow(sobel_im, cmap = 'gray')
plt.subplot(234)
plt.imshow(scharr_im, cmap = 'gray')
plt.subplot(235)
plt.imshow(prewitt_im, cmap = 'gray')
plt.subplot(236)
plt.imshow(canny_auto_im, cmap = 'gray')
plt.show()
#%% Fourier Transform
#Outer regions represent high frequency components
#Use a low pass filter
image = cv2.imread(vz, 0)

dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)
magnitude_spectrum = 20 * np.log((cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))+1)
fig = plt.figure(figsize=(12, 12))
ax1 = fig.add_subplot(2,2,1)
ax1.imshow(image, cmap = 'gray')
ax1.title.set_text('Input Image')
ax2 = fig.add_subplot(2,2,2)
ax2.imshow(magnitude_spectrum, cmap = 'magma')
ax2.title.set_text('FFT of image')
plt.show()
#%% Circular HPF mask - edge dectection but amplifies noise
rows, cols = image.shape
crow, ccol = int(rows / 2), int(cols / 2)
mask = np.ones((rows, cols, 2), np.uint8)
r = 3
center = [crow, ccol]
x, y = np.ogrid[:rows, :cols]
mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r*r
mask[mask_area] = 0
#%% Circular LPF mask - smooth regions and blus edges
rows, cols = image.shape
crow, ccol = int(rows / 2), int(cols / 2)
mask = np.zeros((rows, cols, 2), np.uint8)
r = 100
center = [crow, ccol]
x, y = np.ogrid[:rows, :cols]
mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r*r
mask[mask_area] = 1
#%% Band Pass Filter 
rows, cols = image.shape
crow, ccol = int(rows / 2), int(cols / 2)
mask = np.zeros((rows, cols, 2), np.uint8)
r_out = 80
r_in = 10
center = [crow, ccol]
x, y = np.ogrid[:rows, :cols]
mask_area = np.logical_and(((x - center[0]) ** 2 + (y - center[1]) ** 2 >= r_in ** 2),
                           ((x - center[0]) ** 2 + (y - center[1]) ** 2 <= r_out ** 2))
mask[mask_area] = 1
#%%Inverse back to image domain
fshift = dft_shift * mask
fshift_mask_mag = 20 * np.log(cv2.magnitude(fshift[:, :, 0], fshift[:, :, 1]))
f_ishift = np.fft.ifftshift(fshift)
img_back = cv2.idft(f_ishift)
img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

fig = plt.figure(figsize=(12, 12))
ax1 = fig.add_subplot(2,2,1)
ax1.imshow(image, cmap='gray')
ax1.title.set_text('Input Image')
ax2 = fig.add_subplot(2,2,2)
ax2.imshow(magnitude_spectrum, cmap='magma')
ax2.title.set_text('FFT of image')
ax3 = fig.add_subplot(2,2,3)
ax3.imshow(fshift_mask_mag, cmap='magma')
ax3.title.set_text('FFT + Mask')
ax4 = fig.add_subplot(2,2,4)
ax4.imshow(img_back, cmap='gray')
ax4.title.set_text('After inverse FFT')
plt.show()
#%% Histogram equalisation = stretching to improve contrast but noise gets enhanced
img=cv2.imread(vz,1)
lab_img= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
l, a, b = cv2.split(lab_img)
equ = cv2.equalizeHist(l)
updated_lab_img1 = cv2.merge((equ,a,b))
hist_eq_img = cv2.cvtColor(updated_lab_img1, cv2.COLOR_LAB2BGR)
hist_eq_img = cv2.cvtColor(hist_eq_img, cv2.COLOR_BGR2GRAY)
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
clahe_img = clahe.apply(l)
updated_lab_img2 = cv2.merge((clahe_img,a,b))
CLAHE_img = cv2.cvtColor(updated_lab_img2, cv2.COLOR_LAB2BGR)
CLAHE_img = cv2.cvtColor(CLAHE_img, cv2.COLOR_BGR2GRAY)
plt.figure(figsize = (16,12))
plt.subplot(321)
plt.imshow(img, cmap = 'gray') 
plt.subplot(322) 
plt.hist(l.flat, bins=100, range=(0,255))
plt.subplot(323)
plt.imshow(clahe_img)#hist_eq_img, cmap = 'gray')
plt.subplot(324)
plt.hist(equ.flat, bins=100, range=(0,255))
plt.subplot(325)
plt.imshow(CLAHE_img, cmap = 'gray')
plt.subplot(326)
plt.hist(CLAHE_img.flat, bins=100, range=(0,255))
plt.show()
#%% More Segmentation 
#Automatic Thresholding
image = cv2.imread(vz,1)
blue_channel = image[:,:,0]
ret2, thresh2 = cv2.threshold(blue_channel,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
regions1=np.digitize(blue_channel, bins=np.array([ret2]))
plt.imshow(regions1, cmap = 'gray')
plt.show()
image = cv2.imread(vz,0)
denoised_img = denoise_tv_chambolle(image, weight=0.1, eps=0.0002, n_iter_max=200, multichannel=False)
plt.imshow(image, cmap='magma')
plt.show()
#plt.hist(image.flat, bins=100, range=(100,255))
thresholds = threshold_multiotsu(image, classes=4)
regions = np.digitize(image, bins=thresholds)
plt.imshow(regions, cmap = 'magma')
plt.show()
#can clean similar to above with binary opening/closing
#%% Measurments
image = img_as_ubyte(rgb2gray(io.imread(fname = vz)))
plt.imshow(image)
scale = 5
threshold = threshold_otsu(image)
thresholded_image = image < threshold
plt.imshow(thresholded_image)

#edge_touching_removed = clear_border(thresholded_image)
#plt.imshow(edge_touching_removed)
label_image = measure.label(thresholded_image, connectivity=image.ndim)
plt.imshow(label_image)
image_label_overlay = label2rgb(label_image, image=image)
plt.imshow(image_label_overlay)

all_props = measure.regionprops(label_image, image)
for prop in all_props:
    print('Label : {} Area: {}'.format(prop.label, prop.area))
props = measure.regionprops_table(label_image, image, 
                                  properties = ['label', 
                                                'area', 
                                                'equivalent_diameter',
                                                'mean_intensity',
                                                'solidity'])
df = pd.DataFrame(props)
print(df.head())
df = df[df['area'] > 50]
print(df.head())
#Available regionprops: area, bbox, centroid, convex_area, coords, eccentricity,
# equivalent diameter, euler number, label, intensity image, major axis length, 
#max intensity, mean intensity, moments, orientation, perimeter, solidity, and many more

#%%
image = img_as_float(skimage.io.imread(fname = va, as_gray = False))

through = []
def run_through(image, no_of_sections):
    intervals = np.linspace(0, 1, no_of_sections)
    for i in range(0,no_of_sections-1):
        mask = np.ma.masked_outside(image, intervals[i], intervals[1+i])
        through.append(mask)
build = []   
def build_up(image, no_of_sections):
    intervals = np.linspace(0, 1, no_of_sections)
    for i in range(0,no_of_sections):
        mask = np.ma.masked_greater(image, intervals[i])
        build.append(mask)
dig = []
def dig_up(image, no_of_sections):
    interval = np.linspace(0, 1, no_of_sections)
    intervals = np.flip(interval)
    for i in range(0,no_of_sections):
        mask = np.ma.masked_less_equal(image, intervals[i])
        dig.append(mask)
#%%
run_through(image, 10)
build_up(image, 9)
dig_up(image, 9)
for i in range(0, len(through)):
    fig, ax = plt.subplots()
    ax.imshow(image, cmap = 'gray')
    ax.imshow(through[i], cmap = 'magma', interpolation = 'none')

for i in range(0, len(build)):
    fig, ax = plt.subplots()
    ax.imshow(image, cmap = 'gray')
    ax.imshow(build[i], cmap = 'magma', interpolation = 'none')
    
for i in range(0, len(dig)):
    fig, ax = plt.subplots()
    ax.imshow(image, cmap = 'gray')
    ax.imshow(dig[i], cmap = 'magma', interpolation = 'none')
#%%
dig = []
image_vz = img_as_float(skimage.io.imread(fname = vz, as_gray = False))
dig_up(image_vz, 10)
build_vz = dig
dig = []
image_va = img_as_float(skimage.io.imread(fname = va, as_gray = False))
dig_up(image_va, 10)
for i in range(0, len(dig)):
    fig,( ax1, ax2) = plt.subplots(1,2)
    ax1.imshow(image_vz, cmap = 'gray')
    ax1.imshow(build_vz[i], cmap = 'magma', interpolation = 'none')
    ax2.imshow(image_va, cmap = 'gray')
    ax2.imshow(dig[i], cmap = 'magma', interpolation = 'none')
    plt.show()
#%%
plt.imshow(dig[1], cmap = 'winter', interpolation = 'none')
plt.imshow(build_vz[1], cmap = 'magma', interpolation = 'none')
efm = dig[1]
z_height = build_vz[1]
for c in contours:
    plt.plot(c[:,1],c[:,0], color = 'blue')
print(contours)
print(len(contours))
#%%

for i in range(0, len( contours)):
    cimg = np.zeros_like(efm)
    cv2.drawContours(cimg, contours, i, color=255, thickness = -1)
    pts = np.where(cimg == 255)
    lst_intensities.append(img[pts[0], pts[1]])
#%%
import math
import matplotlib.path
import numpy as np

for c in contours:
    polygon = np.array(c)
    left = np.min(polygon, axis=0)
    right = np.max(polygon, axis=0)
    x = np.arange(math.ceil(left[0]), math.floor(right[0])+1)
    y = np.arange(math.ceil(left[1]), math.floor(right[1])+1)
    xv, yv = np.meshgrid(x, y, indexing='xy')
    points = np.hstack((xv.reshape((-1,1)), yv.reshape((-1,1))))
    
    path = matplotlib.path.Path(polygon)
    mask = path.contains_points(points)
    mask.shape = xv.shape
    plt.plot(c[:,1],c[:,0], color = 'blue')
#%%
from skimage import data, img_as_float
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error
img_vz = build_vz[4]# img_as_float(io.imread(fname = vz, as_gray = False))
img_va = dig[4] #img_as_float(io.imread(fname = va, as_gray = False))
#img_va = 1 - img_va
rows,cols = img_vz.shape
fig, axes = plt.subplots(nrows=1, ncols=2, figsize = (12,6))
ax = axes.ravel()

mse_vz = mean_squared_error(img_vz, img_vz)
ssim_vz = ssim(img_vz, img_vz, data_range = img_vz.max() - img_vz.min())
mse_va = mean_squared_error(img_vz, img_va)
ssim_va = ssim(img_vz, img_va, data_range = img_va.max() - img_va.min())

label = 'MSE: {:.2f}, SSIM: {:.2f}'
ax[0].imshow(img_vz, cmap = 'gray')
ax[0].set_xlabel(label.format(mse_vz, ssim_vz))
ax[1].imshow(img_va, cmap = 'gray')
ax[1].set_xlabel(label.format(mse_va, ssim_va))
plt.tight_layout()
plt.show()


