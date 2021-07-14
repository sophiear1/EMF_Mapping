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
#%%Advanceed Workflow Example 
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
image_n = io.imread(iz, as_gray = True) 
#show image
plt.imshow(image_n)
#invert image since function looks for dark
image = 1-image_n 
#show inversion
plt.imshow(image)
#use median filter to denoise - can adjust size to change amount filtered out
#Gets rid of speckle noise 
from scipy import ndimage as ndi
from skimage import util
denoised = ndi.median_filter(util.img_as_float(image), size = 3)
plt.imshow(denoised)
#Threshold for seperation
from skimage import exposure
image_g = exposure.adjust_gamma(denoised, 0.7)
plt.imshow(image_g)
#Threshold decesicion - can attempt to automate this with algorithms later
t = 0.5
thresholded = (image_g <= t)
plt.imshow(thresholded)
#try filters to estimate value for threshold
from skimage import filters
#filters.try_all_threshold(image_g)
#distance from maximum points to minimum points
from skimage import segmentation, morphology, color
distance = ndi.distance_transform_edt(thresholded)
plt.imshow(exposure.adjust_gamma(distance,0.5))
plt.title('Distance to background map')
#maxima_local 
local_maxima = morphology.local_maxima(distance)
fig,ax = plt.subplots(1,1)
maxi_coords = np.nonzero(local_maxima)
ax.imshow(image)
plt.scatter(maxi_coords[1],maxi_coords[0])
#shuffle label
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
#colours won't overrun
labels_masked = segmentation.watershed(thresholded,markers,mask = thresholded, connectivity = 2)
f, (axo,ax1,ax2) = plt.subplots(1,3)
axo.imshow(thresholded)
ax1.imshow(np.log(1 + distance))
ax2.imshow(shuffle_labels(labels_masked), cmap = 'magma')
#plot contors over the top so can see circles around region
from skimage import measure
contours = measure.find_contours(labels_masked,level = 0.5)
plt.imshow(image)
for c in contours:
    plt.plot(c[:,1],c[:,0])
    #%%
regions = measure.regionprops(labels_masked)
f,ax = plt.subplots()
ax.hist
    