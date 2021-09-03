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
filename = va # file
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
t = 0.4
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
#image = 255-image
for c in contours:
    c = (c*10)-0.75

plt.imshow(image)
for c in contours:
    plt.plot(c[:,1],c[:,0])
    
    

#%%
image = io.imread(vz, as_gray = True) 
#image2 = image
print(labels_masked)
contours2 = contours
for i in image2:
    if labels_masked[i] = 0:
        image2[i] = 0


#%%
for c in contours:
    plt.plot(c[:,1],c[:,0],color = 'red')
#%%
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
x = np.linspace(0,-1, 256)
y = np.linspace(0, 1, 256)
fig = plt.figure(figsize = [16,16])
ax = fig.gca(projection='3d')
ax.contour3D(x, y, image, 150, cmap='magma')
ax.view_init(60, 60)
fig
#%%

x ,y= np.meshgrid(np.linspace(0,-1, 256),np.linspace(0, 1, 256))
fig = plt.figure(figsize = [16,16])
ax = fig.gca(projection='3d')
ax.plot_surface(x, y, Z1_masked, cmap='magma')
ax.view_init(60, 60)
fig
from skimage import img_as_float, io
image = img_as_float(io.imread(vz, as_gray = True)) 
a = uneven_through(image, 0.363,0.78)

#%%
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage import img_as_float, io
import numpy as np
image = img_as_float(io.imread(vz, as_gray = True)) 
a = uneven_through(image, 0.363,0.78)

Z1_masked = a[0]
for i in range(0,len(Z1_masked)):
    for p in range(0,256):
        if isinstance(Z1_masked[i][p], np.ma.core.MaskedConstant):
            Z1_masked[i][p] = 0
Z2_masked = a[1]
for i in range(0,len(Z2_masked)):
    for p in range(0,256):
        if isinstance(Z2_masked[i][p], np.ma.core.MaskedConstant):
            Z2_masked[i][p] = 0.363
Z3_masked = a[2]
for i in range(0,len(Z3_masked)):
    for p in range(0,256):
        if isinstance(Z3_masked[i][p], np.ma.core.MaskedConstant):
            Z3_masked[i][p] = 0.78
            
x_mg, y_mg = np.meshgrid(np.linspace(0,-1, 256), np.linspace(0,1,256))
fig = plt.figure(figsize = (20,20))
ax = fig.add_subplot(1,1,1, projection = '3d')
surf = ax.plot_surface(x_mg, y_mg, Z1_masked, vmin = 0, vmax = 0.5,cmap = 'magma', linewidth = 0,antialiased = False)
cset = ax.contour(x_mg, y_mg, Z1_masked, zdir='z', offset=-1, cmap='winter')
ax.set_zlim(-1, 1)
ax.view_init(30, 45)
fig.colorbar(surf)
plt.show()

fig = plt.figure(figsize = (20,20))
ax = fig.add_subplot(1,1,1, projection = '3d')
surf = ax.plot_surface(x_mg, y_mg, Z2_masked, vmin = 0, vmax = 1, cmap = 'magma', linewidth = 0,antialiased = False)
cset = ax.contour(x_mg, y_mg, Z2_masked, zdir='z', offset=-1, cmap='winter')
ax.set_zlim(-1, 1)
ax.view_init(30, 45)
fig.colorbar(surf)
plt.show()

fig = plt.figure(figsize = (20,20))
ax = fig.add_subplot(1,1,1, projection = '3d')
surf = ax.plot_surface(x_mg, y_mg, Z3_masked, vmin = 0.5, vmax = 1, cmap = 'magma', linewidth = 0,antialiased = False)
#surf = ax.contour3D(x, y, Z3_masked, 150, cmap='magma')
cset = ax.contour(x_mg, y_mg, Z3_masked, zdir='z', offset=-1, cmap='winter')
#cset = ax.contour(x_mg, y_mg, Z3_masked, zdir='x', offset=0, cmap='winter')
#cset = ax.contour(x_mg, y_mg, Z3_masked, zdir='y', offset=0, cmap='winter')
ax.set_zlim(-1, 1)

ax.view_init(30, 45)
fig.colorbar(surf)
#%%
import plotly.graph_objects as go

x, y = np.linspace(0,-1, 256), np.linspace(0,1,256)
x_mg, y_mg = np.meshgrid(np.linspace(0,-1, 256), np.linspace(0,1,256))
fig = go.Figure(data=[go.Surface(x=x_mg, y=y_mg,z=Z3_masked, colorscale='magma'), go.Surface(x=x, y=y, z=im_va, colorscale = 'picnic')])
#fig.update_traces(contours_z=dict(show=True, usecolormap=True,
                                  #highlightcolor="limegreen", project_z=True))
fig.update_layout(title='Z Height', autosize=False,
                  width=1000, height=1000,
                  margin=dict(l=65, r=50, b=65, t=90))
fig.show()
fig.write_html("testfile.html")

#%%
import matplotlib as mpl
image = img_as_float(io.imread(vz, as_gray = True)) 
im_va = img_as_float(io.imread(va, as_gray = True))
a = uneven_through(im_va, 0.1, 0.90)
E3_masked = a[2]
#cmap, norm = mpl.colors.from_levels_and_colors([0], ['white'])
x_mg, y_mg = np.meshgrid(np.linspace(0,-1, 256), np.linspace(0,1,256))
x, y = np.linspace(0,-1, 256), np.linspace(0,1,256)
for i in range(0,len(E3_masked)):
    for p in range(0,256):
        if isinstance(E3_masked[i][p], np.ma.core.MaskedConstant):
            E3_masked[i][p] = False

E3_masked = E3_masked*1
fig = plt.figure(figsize = (20,20))
ax = fig.add_subplot(1,1,1, projection = '3d')
image = image

#surf = ax.plot_surface(x_mg, y_mg, E3_masked, vmin = 0.78, cmap = 'twilight', linewidth = 0,antialiased = False)
#zheight = ax.plot(image, cmap = 'magma')
#cset = ax.contour(x_mg, y_mg, image, zdir='z', offset=0, cmap='plasma')
surf = ax.contour3D(x, y, E3_masked, 150, vmin = 0, vmax = 10, cmap ='winter')
#cset = ax.plot_surface(x_mg, y_mg, image, cmap='plasma')
cset = ax.contour3D(x, y, image, 150,cmap='plasma')

ax.set_zlim3d(0, 5)
#ax.view_init(45, 60)
#%%
im_va = img_as_float(io.imread(va, as_gray = True))
image = img_as_float(io.imread(vz, as_gray = True))
a = uneven_through(im_va, 0.1, 0.50)
E1_masked = a[0]
for i in range(0,len(E1_masked)):
    for p in range(0,256):
        if isinstance(E1_masked[i][p], np.ma.core.MaskedConstant):
            E1_masked[i][p] = np.nan
E1_masked = E1_masked + 0.78

x1, y1 = np.meshgrid(np.linspace(0,-1,256), np.linspace(0,1,256))
z1 = im_va
x,y,z = x1.flatten(), y1.flatten(), z1.flatten()

fig = go.Figure(data=[
    go.Scatter3d(
    x=x,
    y=y,
    z=z,
    mode='markers',
    marker=dict(
        size=5,
        color=z,                # set color to an array/list of desired values
        colorscale='picnic',# choose a colorscale
        opacity=1
    )),
    go.Surface(x=x_mg, y=y_mg,z=image, colorscale='magma', opacity=0.7)
])
fig.show()
fig.write_html("testfile.html")

#%%
import plotly.graph_objects as go
contours2 = contours
contours2 = [x.astype(int) for x in contours2]
labels = []
for c in contours2:
    label = measure.label(c)
    labels.append(label)
for index in range(1, labels.max()):
    cont = measure.find_contours(labels == label, 0.5)[0]
    y,x = cont.T()
    fig.add_trace(go.Scatter(x=x, y=y))
    

#%%
import cv2
contours2=contours
image = io.imread(vz, as_gray = True) 
lst_intensities = []
#for i in contours2:
    # Create a mask image that contains the contour filled in
#    cimg = np.zeros_like(image)
#    cimg = plt.plot(i[:,1], i[:,0], color = 'black')
#    pts = np.where(cimg == 256)
    #lst_intensities.append(image[pts[0], pts[1]])

empty = []
w,h = 256, 256;
empty = np.array([[0 for x in range(w)] for y in range(h)])
print(empty)
contours2 = contours
contours2 = [x.astype(int) for x in contours]
fig = plt.figure(figsize = (256, 256))
for con in contours2:
    empty[con[:,1], con[:,0]] = 256
    ax = plt.fill(con[:,1], con[:,0])
plt.axis('off')
#%%
plt.savefig('tester.png')
area = io.imread(fname = 'tester.png', as_gray = True)
plt.imshow(area)
#%%
import main
import numpy as np
import matplotlib.pyplot as plt
from numpy import sqrt, array, float64, vstack, ma
from skimage import img_as_float, io

def uneven_through(image,b,c):
    through = []
    intervals = array([0,b,c,1])
    for i in range(0,len(intervals)-1):
        mask = ma.masked_outside(image, intervals[i], intervals[1+i])
        through.append(mask)
    return through 

def euclidean_distance(row1, row2, col1, col2):
    dist = (((row1 - row2)**2)+((col1 - col2)**2))
    dist = sqrt(dist)
    return dist

def one_p_to_set(row, col, sets):
    p_row = row
    p_col = col
    shortest_distance = []
    distances = []
    sets = map(tuple, sets)
    sets = tuple(sets)
    for rs in range([len(a) for a in sets][0]):
        for cs in range([len(a) for a in sets][1]):
            if isinstance(sets[rs][cs], np.float64) == True:
                s_row = rs
                s_col = cs
                dist = euclidean_distance(p_row, s_row, p_col, s_col)
                dist = array([dist])
                distances.append(dist)
    distances.sort()
    shortest_distance.append(distances[0])
    return shortest_distance

def line_points_to_sets(points, sets):
    dis = []
    points = map(tuple, points)
    points = tuple(points)
    for r in range(1):
        for c in range([len(a) for a in points][1]):
            if isinstance(points[r][c] , float64) == True:
                dis.append(one_p_to_set(r,c,sets))
    return dis

#%%

p3ht = main.molecule(iz = 'P3HT 58k 11 5um 0V_190208_Z Height_Forward_001.tiff', ia = 'P3HT 58k 11 5um 0V_190208_EFM Amplitude_Forward_001.tiff', vz = 'P3HT 58k 11 5um EFM 2V_190208_Z Height_Forward_003.tiff', va = 'P3HT 58k 11 5um EFM 2V_190208_EFM Amplitude_Forward_003.tiff')
#p3ht = main.molecule(iz = 'P3HT 59k 11 Vdep 0V_210617_Z Height_Forward_016.tiff', ia = 'P3HT 59k 11 Vdep 0V_210617_EFM Amplitude_Forward_016.tiff', vz = 'P3HT 59k 11 Vdep 4V_210617_Z Height_Forward_021.tiff', va = 'P3HT 59k 11 Vdep 4V_210617_EFM Amplitude_Forward_021.tiff')

#p3ht.read()
#p3ht.bearray()
iz = p3ht.iz()
ia = p3ht.ia()
vz = p3ht.vz()
va = p3ht.va()

#%%

im_va = img_as_float(io.imread(fname = ia, as_gray = True))
image = img_as_float(io.imread(fname = vz, as_gray = True))
a = uneven_through(image, 0.363,0.78)
sets0 = a[0]
sets1 = a[1]
sets2 = a[2]
points = im_va

d0 = vstack(line_points_to_sets(points, sets0))
d1 = vstack(line_points_to_sets(points, sets1))
d2 = vstack(line_points_to_sets(points, sets2))
#%%

im_va = img_as_float(io.imread(fname = ia, as_gray = True))
image = img_as_float(io.imread(fname = vz, as_gray = True))
a = uneven_through(image, 0.363,0.78)
sets0 = a[0]
sets1 = a[1]
sets2 = a[2]
points = im_va

a0 = vstack(line_points_to_sets(points, sets0))
a1 = vstack(line_points_to_sets(points, sets1))
a2 = vstack(line_points_to_sets(points, sets2))

#%% No voltage
charges = np.reshape(im_va[0], (256,1))

c0 = np.hstack((d0, charges))
c1 = np.hstack((d1, charges))
c2 = np.hstack((d2, charges))
import pandas as pd
cs0 = c0[c0[:,0].argsort()]
cs1 = c1[c1[:,0].argsort()]
cs2 = c2[c2[:,0].argsort()]

west0 = pd.DataFrame(cs0, columns = ['di', 'ch'])
west1 = pd.DataFrame(cs1, columns = ['di', 'ch'])
west2 = pd.DataFrame(cs2, columns = ['di', 'ch'])

west0['di'] *= 5/255
west0['ch'] *= 0.47
west1['di'] *= 5/255
west1['ch'] *= 0.47
west2['di'] *= 5/255
west2['ch'] *= 0.47

plt.scatter(west0['di'], west0['ch'], c = 'blue')
plt.scatter(west1['di'], west1['ch'], c = 'yellow')
plt.scatter(west2['di'], west2['ch'], c = 'red')
plt.show()
#%% Voltage
import pandas as pd
charges = np.reshape(im_va[0], (256,1))

e0 = np.hstack((d0, charges))
e1 = np.hstack((d1, charges))
e2 = np.hstack((d2, charges))

es0 = e0[e0[:,0].argsort()]
es1 = e1[e1[:,0].argsort()]
es2 = e2[e2[:,0].argsort()]

east0 = pd.DataFrame(es0, columns = ['di', 'ch'])
east1 = pd.DataFrame(es1, columns = ['di', 'ch'])
east2 = pd.DataFrame(es2, columns = ['di', 'ch'])

east0['di'] *= 5/255
east0['ch'] *= 2.7
east1['di'] *= 5/255
east1['ch'] *= 2.7
east2['di'] *= 5/255
east2['ch'] *= 2.7

plt.scatter(east0['di'], east0['ch'], c = 'pink')
plt.scatter(east1['di'], east1['ch'], c = 'purple')
plt.scatter(east2['di'], east2['ch'], c = 'red')
plt.show()


#%%
import matplotlib.pyplot as plt
mycolors = ['lightcoral', 'red','lightskyblue','blue', 'lightgreen','green']      

def plotting(dataset, points):
    x = np.linspace(0,dataset['di'].max(), points)
    y = []
    i=dataset[dataset['di']==0]
    j=i['ch'].sum()
    k=i['ch'].count()
    y.append(j/k)
    for i in range(0,len(x)-1):
        h= dataset[dataset['di'].between(x[i], x[i+1], inclusive = 'right')]
        s = h['ch'].sum()
        c = h['ch'].count()
        y.append(s/c)
    return x,y


plt.figure(figsize=(16, 10), dpi= 80, facecolor='w', edgecolor='k')
x1,y1 = plotting(east0,20)
x2,y2 = plotting(west0,20)
plt.scatter(x1,y1, c=mycolors[0], s=150)
plt.scatter(x2,y2, c=mycolors[1], s=150)
plt.fill_between(x = x1, y1=y1, y2=0, alpha=0.5, color=mycolors[0], linewidth=2)
plt.fill_between(x = x2, y1=y2, y2=0, alpha=1, color=mycolors[1], linewidth=2)
plt.xlabel('Distance from Set (μm)', fontsize = 30)
plt.ylabel('Charge Density (mV/pixel)', fontsize = 30)
plt.title("Set 0", fontsize=22)
plt.xticks(fontsize=20); plt.yticks(fontsize=20)
plt.legend(labels = ['2V', '0V'], frameon = True, fontsize=30)    
plt.show()

plt.figure(figsize=(16, 10), dpi= 80, facecolor='w', edgecolor='k')
x1,y1 = plotting(east1,20)
x2,y2 = plotting(west1,20)
plt.scatter(x1,y1, c=mycolors[2], s=150)
plt.scatter(x2,y2, c=mycolors[3], s=150)
plt.fill_between(x = x1, y1=y1, y2=0, alpha=0.5, color=mycolors[2], linewidth=2)
plt.fill_between(x = x2, y1=y2, y2=0, alpha=1, color=mycolors[3], linewidth=2)
plt.xlabel('Distance from Set (μm)', fontsize = 30)
plt.ylabel('Charge Density (mV/pixel)', fontsize = 30)
plt.title("Set 1", fontsize=22)
plt.xticks(fontsize=20); plt.yticks(fontsize=20)
plt.legend(labels = ['2V', '0V'],fontsize=30, frameon = True)    
plt.show()


plt.figure(figsize=(16, 10), dpi= 80, facecolor='w', edgecolor='k')
x1,y1 = plotting(east2,20)
x2,y2 = plotting(west2,20)
plt.scatter(x1,y1, c=mycolors[4], s=150)
plt.scatter(x2,y2, c=mycolors[5], s=150)
plt.fill_between(x = x1, y1=y1, y2=0, alpha=0.5, color=mycolors[4], linewidth=2)
plt.fill_between(x = x2, y1=y2, y2=0, alpha=1, color=mycolors[5], linewidth=2)
plt.xlabel('Distance from Set (μm)', fontsize = 30)
plt.ylabel('Charge Density (mV/pixel)', fontsize = 30)
plt.title("Set 2", fontsize=22)
plt.xticks(fontsize=20); plt.yticks(fontsize=20)
plt.legend(labels = ['2V', '0V'],fontsize=30, frameon = True)    
plt.show()   

x1,y1 = plotting(east0,20)
x2,y2 = plotting(east1,20)
x3, y3 = plotting(east2, 20)
x4,y4 = plotting(west0,20)
x5,y5 = plotting(west1,20)
x6, y6 = plotting(west2, 20)     
columns = ['Set 1 - 2V', 'Set 2 - 2V', 'Set 3 - 2V', 'Set 1 - 0V', 'Set 2 - 0V', 'Set 3 - 0V']
fig, ax = plt.subplots(1, 1, figsize=(16,9), dpi= 80)
ax.fill_between(x = x1, y1=y1, y2=0, label=columns[0], alpha=0.5, color=mycolors[0], linewidth=2)
ax.fill_between(x = x2, y1=y2, y2=0, label=columns[1], alpha=0.5, color=mycolors[2], linewidth=2)
ax.fill_between(x = x3, y1=y3, y2=0, label=columns[2], alpha=0.5, color=mycolors[4], linewidth=2)
ax.fill_between(x = x4, y1=y4, y2=0, label=columns[3], alpha=0.6, color=mycolors[1], linewidth=2)
ax.fill_between(x = x5, y1=y5, y2=0, label=columns[4], alpha=0.6, color=mycolors[3], linewidth=2)
ax.fill_between(x = x6, y1=y6, y2=0, label=columns[5], alpha=0.6, color=mycolors[5], linewidth=2)
ax.set_title('0V and 2V, All Sets', fontsize=30)
ax.legend(loc='best', fontsize=20)
plt.xticks(fontsize=18, horizontalalignment='center')
plt.yticks(fontsize=18)
plt.xlim(0, max(x3))
plt.ylim(0, max(y1))
plt.xlabel('Distance from Set (μm)', fontsize = 20)
plt.ylabel('Charge Density (mV/pixel)', fontsize = 20)
plt.show()
#%%
import scipy as sp
plt.figure(figsize=(16, 10), dpi= 80, facecolor='w', edgecolor='k')
x1,y1 = plotting(east2,20)
x2,y2 = plotting(west2,20)
plt.scatter(x1,y1, c=mycolors[4], s=150)
plt.scatter(x2,y2, c=mycolors[5], s=150)
plt.fill_between(x = x1, y1=y1, y2=0, alpha=0.5, color=mycolors[4], linewidth=2)
plt.fill_between(x = x2, y1=y2, y2=0, alpha=1, color=mycolors[5], linewidth=2)
plt.xlabel('Distance from Set (μm)', fontsize = 30)
plt.ylabel('Charge Density (mV/pixel)', fontsize = 30)
plt.title("Set 2", fontsize=22)
plt.xticks(fontsize=20); plt.yticks(fontsize=20)
plt.legend(labels = ['2V', '0V'],fontsize=30, frameon = True)      
p = np.poly1d(np.polyfit(x1, y1, 8))
t = np.linspace(0, x1[-1], 20)
plt.plot(t, p(t), '-', c = mycolors[5])
plt.scatter(x2,y2)
plt.show()

#%%
x1 = np.linspace(0, east2['di'].max(), 20)
y1=[]
h = east2[east2['di'] == 0] 
s = h['ch'].sum()
c = h['ch'].count()
y1.append(s/c)
for i in range(0, len(x1)-1):
    h = east2[east2['di'].between(x1[i], x1[i+1], inclusive = 'right')]
    s = h['ch'].sum()
    c = h['ch'].count()
    y1.append(s/c)

x = np.linspace(0, west2['di'].max(), 20)
y=[]
h = west2[west2['di'] == 0] 
s = h['ch'].sum()
c = h['ch'].count()
y.append(s/c)
for i in range(0, len(x)-1):
    h = west2[west2['di'].between(x[i], x[i+1], inclusive = 'right')]
    s = h['ch'].sum()
    c = h['ch'].count()
    y.append(s/c)
plt.plot(x, y)
plt.plot(x1, y1)
plt.show()
#%%
x1,y1 = plotting(east0,20)
x2,y2 = plotting(east1,20)
x3, y3 = plotting(east2, 20)

mycolors = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange', 'tab:brown', 'tab:grey', 'tab:pink', 'tab:olive']      
columns = ['Set 1', 'Set 2', 'Set 3']

fig, ax = plt.subplots(1, 1, figsize=(16,9), dpi= 80)
ax.fill_between(x = x1, y1=y1, y2=0, label=columns[0], alpha=0.5, color=mycolors[0], linewidth=2)
ax.fill_between(x = x2, y1=y2, y2=0, label=columns[1], alpha=0.5, color=mycolors[1], linewidth=2)
ax.fill_between(x = x3, y1=y3, y2=0, label=columns[2], alpha=0.5, color=mycolors[2], linewidth=2)

ax.set_title('2V, All Sets', fontsize=30)
ax.legend(loc='best', fontsize=20)
plt.xticks(fontsize=18, horizontalalignment='center')
plt.yticks(fontsize=18)
plt.xlim(0, max(x3))
plt.ylim(0, max(y1))
plt.xlabel('Distance from Set (μm)', fontsize = 20)
plt.ylabel('Charge Density (mV/pixel)', fontsize = 20)

plt.show()


