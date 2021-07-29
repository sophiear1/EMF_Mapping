# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 14:07:52 2021

@author: sophi
"""

#%%
import main 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({ # big image + gray
        'figure.figsize': (10,10),
        'image.cmap' : 'gray'
        })
p3ht = main.molecule(iz = 'P3HT 58k 11 5um 0V_190208_Z Height_Forward_001.tiff', ia = 'P3HT 58k 11 5um 0V_190208_EFM Amplitude_Forward_001.tiff', vz = 'P3HT 58k 11 5um EFM 2V_190208_Z Height_Forward_003.tiff', va = 'P3HT 58k 11 5um EFM 2V_190208_EFM Amplitude_Forward_003.tiff')
#p3ht.read()
#p3ht.bearray()
iz = p3ht.iz()
ia = p3ht.ia()
vz = p3ht.vz()
va = p3ht.va()
#%%Other Molecule Test
p3ht = main.molecule(iz = 'P3HT 58k 11 5um 0V_190208_Z Height_Forward_001.tiff', 
                     ia = 'P3HT 58k 11 5um 0V_190208_EFM Amplitude_Forward_001.tiff', 
                     vz = 'P3HT 59k 11 Vdep 4V_210617_Z Height_Forward_021.tiff', 
                     va = 'P3HT 59k 11 Vdep 4V_210617_EFM Amplitude_Forward_021.tiff')
vz = p3ht.vz()
va = p3ht.va()
#%%Advanceed Workflow Ideas
import skimage
import skimage.feature
import skimage.viewer
import skimage.data as data
import skimage.segmentation as seg
from skimage import filters
from skimage import draw
from skimage import color
from skimage import exposure
from skimage import io

def image_show(image, nrows=1, ncols=1, cmap='gray', **kwargs):
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(16, 16))
    ax.imshow(image, cmap='gray')
    ax.axis('off')
    return fig, ax
filename = vz # file
image = skimage.io.imread(fname=filename, as_gray=True) #read file
image_show(image) # another way to view
#%%
#import image
import skimage
from skimage import io
image_n = io.imread(vz, as_gray = True) 
#show image
plt.imshow(image_n)

#%%invert image since function looks for dark
image = 255 - image_n 
#image = image_n
#show inversion
plt.imshow(image)
#%%
#use median filter to denoise - can adjust size to change amount filtered out
#Gets rid of speckle noise 
#Increase more for EFM data I think
from scipy import ndimage as ndi
from skimage import util
denoised = ndi.median_filter(util.img_as_float(image), size = 5)
plt.imshow(denoised)
#%%Threshold for seperation
from skimage import exposure
image_g = exposure.adjust_gamma(denoised, 0.7)
plt.imshow(image_g)

#%%Threshold decesicion - can attempt to automate this with algorithms later
t = 0.5
thresholded = (image_g <= t)
plt.imshow(thresholded)
#%%
#try filters to estimate value for threshold
from skimage import filters
filters.try_all_threshold(image_g)
t = filters.threshold_li(image_g)
#%%
thresholded = (image_g <= t)
plt.imshow(thresholded)

#%%distance from maximum points to minimum points
from skimage import segmentation, morphology, color
distance = ndi.distance_transform_edt(thresholded)
plt.imshow(exposure.adjust_gamma(distance,t))
plt.title('Distance to background map')
#%%maxima_local 
local_maxima = morphology.local_maxima(distance)
fig,ax = plt.subplots(1,1)
maxi_coords = np.nonzero(local_maxima)
ax.imshow(image)
plt.scatter(maxi_coords[1],maxi_coords[0])
#%%shuffle label
def shuffle_labels(labels):
    indices = np.unique(labels[labels != 0])
    indices = np.append(
            [0],
            np.random.permutation(indices)
            )
    return indices[labels]

markers = ndi.label(local_maxima)[0]
labels = segmentation.watershed(denoised, markers)
f, (axo,ax1,ax2) = plt.subplots(1,3)
axo.imshow(thresholded)
ax1.imshow(np.log(1 + distance))
ax2.imshow(shuffle_labels(labels), cmap = 'magma')

#%%colours won't overrun
labels_masked = segmentation.watershed(thresholded,markers,mask = thresholded, connectivity = 2)
f, (axo,ax1,ax2) = plt.subplots(1,3)
axo.imshow(thresholded)
ax1.imshow(np.log(1 + distance))
ax2.imshow(shuffle_labels(labels_masked), cmap = 'magma')

#%%plot contors over the top so can see circles around region
from skimage import measure
contours = measure.find_contours(labels_masked,level = t)
plt.imshow(image)
for c in contours:
    plt.plot(c[:,1],c[:,0])

#%%
contours2 = contours
#%%
for c in contours:
    plt.plot(c[:,1],c[:,0],color = 'red')

for c in contours2:
    plt.plot(c[:,1],c[:,0], color = 'blue')
#%%
regions = measure.regionprops(labels_masked)
f,ax = plt.subplots()
ax.hist([r.area for r in regions], bins = 50, range = (0,1200))
#maybe delete bins for super small 
#%%
print("The percentage white region is:", np.sum(thresholded ==1)*100/(np.sum(thresholded ==0) + np.sum(thresholded ==1)))

#%% Attempt to use machine learning but not working yet 
from keras import models, layers
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
M=76, 75
N=int(23.76*M)*2
model = models.Sequential()
model.add(
        Conv2D(
                32,
                kernel_size=(2,2),
                activation='relu',
                input_shape=(N,N,1),
                padding='same',
            )
    )
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Conv2D(64,(3,3),activation='relu',padding='same'))
model.add(Conv2D(64,(3,3),activation='relu',padding='same'))
model.add(UpSampling2D(size=(2,2)))
model.add(
        Conv2D(
                1,
                kernel_size=(2,2),
                activation='sigmoid',
                padding='same',
            )
    )
model.compile(loss='mse', optimizer='Adam', metrics=['accuracy'])

#%%
def f_c(i):
    #import
    im = io.imread(i, as_gray = True)
    #invert -adjust for efm/z depending on start
    image = 255 - im
    #denoise
    denoised = ndi.median_filter(util.img_as_float(image), size = 5)
    #increase exposure
    image_g = exposure.adjust_gamma(denoised, 0.7)
    #thresholding
    t = filters.threshold_otsu(image_g)
    thresholded = (image_g <= t)
    #distance map
    distance = ndi.distance_transform_edt(thresholded)
    local_maxima = morphology.local_maxima(distance)
    markers = ndi.label(local_maxima)[0]
    labels_masked = segmentation.watershed(thresholded,markers,mask = thresholded, connectivity = 2)
    contours = measure.find_contours(labels_masked,level = t)
    return contours

def f_c_i(i):
    #import
    im = io.imread(i, as_gray = True)
    #invert -adjust for efm/z depending on start
    image = im
    #denoise
    denoised = ndi.median_filter(util.img_as_float(image), size = 5)
    #increase exposure
    image_g = exposure.adjust_gamma(denoised, 0.7)
    #thresholding
    t = filters.threshold_otsu(image_g)
    thresholded = (image_g <= t)
    #distance map
    distance = ndi.distance_transform_edt(thresholded)
    local_maxima = morphology.local_maxima(distance)
    markers = ndi.label(local_maxima)[0]
    labels_masked = segmentation.watershed(thresholded,markers,mask = thresholded, connectivity = 2)
    contours = measure.find_contours(labels_masked,level = t)
    return contours

image = io.imread(va, as_gray = True) 

plt.imshow(image)
con_one = f_c(va)
con_two = f_c_i(va)
for c in con_one:
    plt.plot(c[:,1],c[:,0])
for c in con_two:
    plt.plot(c[:,1],c[:,0])
#%%
image = io.imread(vz, as_gray = True) 
plt.imshow(image, cmap = 'gray')
from mpl_toolkits.mplot3d import Axes3D
x = np.linspace(0,-255, 256)
y = np.linspace(0, 255, 256)
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.contour3D(x, y, image, 150, cmap='magma')
ax.view_init(45, 45)
fig
#%%
import cv2
image = io.imread(vz, as_gray = True) 
plt.imshow(image, cmap = 'gray')
#contours = plt.contour(image, cmap = 'magma')
#contours = measure.find_contours(image, 200)
for c in contours:
    plt.plot(c[:,1], c[:,0])
#%%
image = io.imread(vz, as_gray = True) 
empty = np.empty((256,256))
for i in list(range(0,255)):
    a = list(range(0,255))
    empty[i] = np.append(empty,[a])

print(empty)

#plt.imshow(image, cmap = 'gray')
#plt.imshow(image, interpolation = 'none')





