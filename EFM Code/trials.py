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
#p3ht = main.molecule(iz = 'P3HT 59k 11 Vdep 0V_210617_Z Height_Forward_016.tiff', ia = 'P3HT 59k 11 Vdep 0V_210617_EFM Amplitude_Forward_016.tiff', vz = 'P3HT 59k 11 Vdep 4V_210617_Z Height_Forward_021.tiff', va = 'P3HT 59k 11 Vdep 4V_210617_EFM Amplitude_Forward_021.tiff')

#p3ht.read()
#p3ht.bearray()
iz = p3ht.iz()
ia = p3ht.ia()
vz = p3ht.vz()
va = p3ht.va()
#%%

#al image comparison Trial 1 = Difference 
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
im_va = im_va
im_va = seg.chan_vese(im_va)
im_va = np.vectorize(boolstr_to_floatstr)(im_va).astype(float)

plt.imshow(im_vz)
plt.show()
plt.imshow(im_va)
plt.show()
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

#%% Invert image 
vai = main.np.invert(va)
imagevai = main.Image.fromarray(vai)
imagevai.show()
#%% Roughness
#image = dig[1]
#some = image[~image.mask]
some = np.array([1,2,3,4])
N = len(some)
mean = np.mean(some)
print(mean)
total = 0
for i in range(0,len(some),1):
    dif = np.abs(some[i]-mean)
    total = total+dif
print(total)
#%%

a = uneven_through(image, 0.379, 0.771)
for i in range(0, len(a)):
    b = central_moments(a[i], 2)
    print(b)
    plt.imshow(a[i], cmap = 'autumn')
    plt.show()
#%%
a = uneven_through(image, 0.363, 0.78)
fig, (ax1, ax2, ax3) = plt.subplots(figsize = (16,3), ncols = 3)
set_one = ax1.imshow(a[0]*235, cmap = 'autumn', extent = [0,5,0,5])
ax1.title.set_text('Set 1')
ax1.set_xlabel('Distance (µm)')
ax1.set_ylabel('Distance (µm)')
cb1 = fig.colorbar(set_one, ax=ax1)
cb1.set_label('Z-height (nm)')
set_two = ax2.imshow(a[1]*235, cmap = 'autumn')
ax2.title.set_text('Set 2')
ax2.set_xlabel('Distance (µm)')
ax2.set_ylabel('Distance (µm)')
cb2 = fig.colorbar(set_two, ax = ax2)
cb2.set_label('Z-height (nm)')
set_three = ax3.imshow(a[2]*235, cmap = 'autumn')
ax3.title.set_text('Set 3')
ax3.set_xlabel('Distance (µm)')
ax3.set_ylabel('Distance (µm)')
cb3 = fig.colorbar(set_three, ax = ax3)
cb3.set_label('Z-height (nm)')
plt.show()
#%%
c = average_rms_roughness([0.2,0.8], image)
print(c)

#%%
image = main.img_as_float(main.skimage.io.imread(fname = vz, as_gray = True))
def uneven_through(image, boundaries):
    through = []
    boundaries = np.array(boundaries)
    boundaries = np.append(boundaries, 1)
    boundaries = np.insert(boundaries, 0, 0)
    intervals = boundaries
    for i in range(0,len(intervals)-1):
        mask = np.ma.masked_outside(image, intervals[i], intervals[1+i])
        through.append(mask)
    return through 

boundaries = [0.2, 0.4, 0.6, 0.8]
print(len(boundaries))
a = uneven_through(image, [0.3, 0.6, 0.8])
for i in range(0,len(a)):
    fig, ax = plt.subplots()
    ax.imshow(a[i], cmap = 'magma')
#%%
from skimage import img_as_float
import skimage
image = img_as_float(skimage.io.imread(fname = iz, as_gray = False))
im_va = img_as_float(skimage.io.imread(fname = va, as_gray = True))

def run_through(image, no_of_sections):
    through = []
    intervals = np.linspace(0, 1, no_of_sections+1)
    for i in range(0,no_of_sections):
        mask = np.ma.masked_outside(image, intervals[i], intervals[1+i])
        through.append(mask)
    return through 

def uneven_through(image,b,c):
    through = []
    intervals = np.array([0,b,c,1])
    for i in range(0,len(intervals)-1):
        mask = np.ma.masked_outside(image, intervals[i], intervals[1+i])
        through.append(mask)
    return through 

def central_moments(masked_im, p):
    im = masked_im[~masked_im.mask]
    N = len(im)
    mean = np.mean(im)
    total = 0
    for i in range(0, N, 1):
        dif = im[i] - mean
        if p == 1:
            dif = np.abs(dif)
        if p != 1:
            dif = np.power(dif,p)
        total = total + dif
    if N==0:
        N=1
    mu = total/N
    if p == 2:
        mu = np.power(mu, 0.5)
    return mu

def central_moments_unmasked(unmasked_im, p):
    im = unmasked_im#[~masked_im.mask]
    N = im.size
    row = im.shape[0]
    col = im.shape[1]
    mean = np.mean(im)
    total = 0
    for r in range(0, row, 1):
        for c in range(0,col,1):
            dif = im[r][c] - mean
            if p == 1:
                dif = np.abs(dif)
            if p != 1:
                dif = np.power(dif,p)
            total = total + dif
    mu = total/N
    if p == 2:
        mu = np.power(mu, 0.5)
    return mu

def average_rms_roughness(params, im):
    b,c = params
    through = uneven_through(im, b,c)
    total_mu = 0
    for i in range(0, len(through),1):
        mu = central_moments(through[i], 2)
        total_mu = total_mu + np.power(mu,2)
    average_mu = np.power(total_mu/len(through), 0.5)
    return average_mu
#%%
image = img_as_float(skimage.io.imread(fname = vz))
d= uneven_through(image = image, b = 0.3, c = 0.7)
for i in range(len(d)):
    print(central_moments(d[i], p=2))
#%%
a = average_rms_roughness([0.287,0.848], image)
print(a)

b = run_through(image, 3)
for i in range(0, len(b)):
    fig, ax = plt.subplots()
    ax.imshow(b[i], cmap = 'magma', interpolation = 'none')

c = uneven_through(image, 0.287, 0.848)
for i in range(0, len(c)):
    fig, ax = plt.subplots()
    ax.imshow(c[i], cmap = 'magma', interpolation = 'none')
 #%%   
a = [[0,1,2,3], [4,5,6], [7,8,9]]
print(a[0])
 
 #%%
for d in contors:
    for c in average_rms_roughness:
        fig, ax = plt.subplots()
        ax.imshow(development[i], cmap = 'winter')
 
#%%
image = img_as_float(skimage.io.imread(fname = iz, as_gray = True))
bnds = ((0,1),(0,1))
x0 = [0.3, 0.6]
mi = sp.optimize.differential_evolution(average_rms_roughness, bounds = bnds, args = (image,), strategy='best1exp',  maxiter = 50)
print(mi)
#%%
image = img_as_float(skimage.io.imread(fname = iz, as_gray = True))
bnds = ((0.01,1),(0.01,1))
x0 = [0.33, 0.66]
mi = sp.optimize.dual_annealing(average_rms_roughness, bounds=bnds, args=[image], maxfun =500, x0=x0)
print(mi)

#%% 'trust-constr', 
import scipy as sp
image = img_as_float(skimage.io.imread(fname = vz, as_gray = True))
x0 = [0.33, 0.66]
bnds = ((0,1),(0,1))
mi = sp.optimize.minimize(average_rms_roughness, x0, args=image, method = 'trust-constr', bounds = bnds)
print(mi)
    
#%% Use values of [0.363,0.780]
im_va = img_as_float(skimage.io.imread(fname = va, as_gray = True))
image = img_as_float(skimage.io.imread(fname = vz, as_gray = True))
a = uneven_through(image, 0.41, 0.58)
for i in range(0, len(a)):
    b = central_moments(a[i], 2)
    print('RMS roughness of layer: ', b)
    plt.imshow(a[i], cmap = 'gray')
    plt.show()
    
for i in range(0, len(a)):
    charges = np.ma.masked_array(im_va, mask = a[i].mask)
    plt.imshow(charges, cmap = 'gray')
    plt.show()
    
#%%
im_va = img_as_float(skimage.io.imread(fname = va, as_gray = True))
image = img_as_float(skimage.io.imread(fname = vz, as_gray = True))


#%% Line Graph of Mean Charge against Height
a = run_through(image, 100)
#plt.imshow(a[80])
y_one = np.array([])
y_two = np.array([])


for i in range(0, len(a)):
    charges = np.ma.masked_array(im_va, mask = a[i].mask)
    b = np.array([charges.sum()/charges.count()])
    #b = np.array([charges.sum()])
    y_one = np.append(y_one,b)
    for c in range(0, charges.count()):
        y_two = np.append(y_two, i)
        
# x: array([0.41353745, 0.58158146])  - 12 Iterations
# x: array([0.37903858, 0.7714333 ]) - 119 Iterations

x = np.linspace(0, 1, 100)
plt.plot(x,y_one)
plt.axvline(x=0.363, color = 'red')
plt.axvline(x=0.780, color = 'red')
plt.show()

plt.hist(y_two, bins = 50)
plt.axvline(x=0.363*100, color = 'red')
plt.axvline(x=0.780*100, color = 'red')
plt.show()

#%%
a = run_through(image, 100)
y_one = np.array([])
for i in range(0, len(a)):
    charges = np.ma.masked_array(im_va, mask = a[i].mask)
    total = 0.0
    n = 0
    for ch in charges:
        for c in ch:
            if isinstance(c, np.float64) == True:
                total += c
                n += 1
    b = np.array([total/n])
    y_one = np.append(y_one, b)
    
x = np.linspace(0, 1, 100)
plt.plot(x,y_one)
plt.axvline(x=0.37, color = 'red')
plt.axvline(x=0.77, color = 'red')
plt.show()

#%%
im_va = img_as_float(skimage.io.imread(fname = va, as_gray = True))
image = img_as_float(skimage.io.imread(fname = vz, as_gray = True))
a = run_through(image, 100)
sets = a[99]
points = im_va
points = np.ma.masked_outside(points, 0.5, 1)
         
def dis_p_to_s(points, sets):
    shortest_distances = []
    for rp in range(points.shape[0]):
        for cp in range(points.shape[1]):
            if isinstance(points[rp][cp] , np.float64) == True:
                p_row = rp
                p_col = cp
                distances = []
                for rs in range(sets.shape[0]):
                    for cs in range(sets.shape[1]):
                        if isinstance(sets[rs][cs], np.float64) == True:
                           s_row = rs
                           s_col = cs
                           dist = euclidean_distance(p_row, s_row, p_col, s_col)
                           distances.append(dist)
                distances.sort()
                shortest_distances.append(distances[0])
    return shortest_distances

#short_dist = dis_p_to_s(points, sets)
#print(short_dist)

#%%
import time
start_time = time.time()
def one_p_to_set(row, col, sets):
    p_row = row
    p_col = col
    shortest_distance = []
    distances = []
    for rs in range(sets.shape[0]):
        for cs in range(sets.shape[1]):
            if isinstance(sets[rs][cs], np.float64) == True:
                s_row = rs
                s_col = cs
                dist = euclidean_distance(p_row, s_row, p_col, s_col)
                dist = np.array([dist])
                distances.append(dist)
    distances.sort()
    shortest_distance.append(distances[0])
    return shortest_distance
b = one_p_to_set(np.array([1]), np.array([2]), sets)
print(b)
print("--- %s seconds ---" % (time.time() - start_time))

#%% Decreasing Run-Time Attempts - Using Tuple :
from numpy import sqrt, array, float64, vstack, ma
import time
from skimage import img_as_float, io
start_time = time.time()

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

im_va = img_as_float(io.imread(fname = va, as_gray = True))
image = img_as_float(io.imread(fname = vz, as_gray = True))
a = uneven_through(image, 0.363,0.78)
sets = a[2]
points = im_va
d1 = vstack(line_points_to_sets(points, sets))


print("--- %s seconds ---" % (time.time() - start_time))

#%%
im_va = img_as_float(skimage.io.imread(fname = va, as_gray = True))
image = img_as_float(skimage.io.imread(fname = vz, as_gray = True))
a = run_through(image, 100)
sets = a[99]
points = im_va
points = np.ma.masked_outside(points, 0.5, 1)

def line_points_to_sets(points, sets):
    dis = []
    for r in range(1):
        for c in range(points.shape[1]):
            if isinstance(points[r][c] , np.float64) == True:
                dis.append(one_p_to_set(r,c,sets))
    return dis
                
#%%
image = img_as_float(skimage.io.imread(fname = vz, as_gray = True))
a = uneven_through(image, 0.181,0.528)
sets = a[2]
points = im_va
points = np.ma.masked_outside(points, 0.95, 1)
d1 = np.vstack(line_points_to_sets(points, sets))
points = im_va
points = np.ma.masked_outside(points, 0, 1)
d2 = np.vstack(line_points_to_sets(points,sets))
#%%
plt.hist(d1/360, density = True)
#plt.hist(d2/360, density = True)
plt.xlabel('Distance from EFM Point to Z-height Set')
plt.ylabel('Frequency Density of Number of Pixels')
plt.title('Set 3, 4V')
plt.legend(labels = ('Charges above 1.14mV (top 5%)', 'All Charges'))
plt.show()
#%% Saving Data 
#d_va_0_one = d2
#d_va_0_nine
#d_va_0_all 
#d_ia_0_nine 
#d_ia_0_all
#d_va_1_one 
#d_va_1_nine 
#d_va_1_all 
#d_ia_1_nine
#d_ia_1_all 
#d_va_2_one 
#d_va_2_nine
#d_va_2_all 
#d_ia_2_nine 
#d_ia_2_all 
 
    
#%% Histogram of Distance 
# To Histogram of Charge Density 
distances = d1
#%%
bins = 10
a = points[0]
a = np.vstack(a)
distance_charge = np.vstack((distances,a))
plt.plot(distances, points[0])
interval = max(distances)/bins
#%%
x = []

for i in range(0,bins+1):
    value = i*interval
    x.append(value)
#%%
charges_tot = np.zeros(len(x)-1)
charges_sum = np.zeros(len(x)-1)
for i in range(len(x)-1):
    for d in range(len(distance_charge)):
        if distance_charge[d][0] < x[i + 1] and distance_charge[d][0] > x[i]:
            charges_tot[i] += distance_charge[i][1]
            charges_sum[i] += 1
density = charges_tot/charges_sum
x.pop(0)
x = np.hstack(x)
density = np.hstack(density)
plt.bar(x, density, align = 'center')
plt.show()
#%%
plt.scatter(x, density)
plt.show()
#%%

def euclidean_distance(row1, row2, col1, col2):
    dist = (((row1 - row2)**2)+((col1 - col2)**2))
    dist = np.sqrt(dist)
    return dist
def dist_point_to_set(points, sets):
    shortest_distances = []
    for p in points:
        if isinstance(p, bool) == False:
            p_row = np.where(points == p)[0]
            p_col = np.where(points == p)[1]
            distances = []
            for s in sets:
                if isinstance(s, bool) == False:
                    s_row,s_col = np.where(sets ==s)
                    dis = euclidean_distance(p_row, s_row, p_col, s_col)
                    distances.append(dis)
            distances.sort()
            shortest_distances.append(distances[0])
    return shortest_distances

image = img_as_float(skimage.io.imread(fname = vz, as_gray = True))
a = uneven_through(image, 0.181,0.528)
sets = a[2]
points = im_va
points = np.ma.masked_outside(points, 0.95, 1)
dist_point_to_set(points, sets)

        
#%%
from sklearn.neighbors import NearestNeighbors
import numpy as np
nbrs = NearestNeighbors(n_neighbors=1, algorithm='brute').fit(sets)
distances, indices = nbrs.kneighbors(points)

#%%


import numpy as np
from numpy import fft 

def radial_coordinate(size):
    center_x = size[0]/2.0-0.5
    center_y = size[1]/2.0-0.5
    y, x = np.indices(size)
    return np.sqrt((x-center_x)**2 + (y-center_y)**2)

def pad(image):
    return np.pad(image, ((image.shape[0], image.shape[0]),
                           (image.shape[1], image.shape[1])), mode='constant')

def conv(image_A, image_B, padding=True):
    if padding:
        image_A = pad(image_A)
        image_B = pad(image_B)
    return fft.fftshift(np.real(fft.ifft2(fft.fft2(image_A) *
                                             fft.fft2(image_B))))
rho = 0.02*radial_coordinate(sets.shape)
