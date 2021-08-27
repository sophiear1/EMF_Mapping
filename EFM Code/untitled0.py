# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 08:42:32 2021

@author: sophi
"""
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
import main
p3ht = main.molecule(iz = 'P3HT 58k 11 5um 0V_190208_Z Height_Forward_001.tiff', ia = 'P3HT 58k 11 5um 0V_190208_EFM Amplitude_Forward_001.tiff', vz = 'P3HT 58k 11 5um EFM 2V_190208_Z Height_Forward_003.tiff', va = 'P3HT 58k 11 5um EFM 2V_190208_EFM Amplitude_Forward_003.tiff')
#p3ht = main.molecule(iz = 'P3HT 59k 11 Vdep 0V_210617_Z Height_Forward_016.tiff', ia = 'P3HT 59k 11 Vdep 0V_210617_EFM Amplitude_Forward_016.tiff', vz = 'P3HT 59k 11 Vdep 5V_210617_Z Height_Forward_022.tiff', va = 'P3HT 59k 11 Vdep 0V_210617_EFM Amplitude_Forward_016.tiff')

#p3ht.read()
#p3ht.bearray()
iz = p3ht.iz()
ia = p3ht.ia()
vz = p3ht.vz()
va = p3ht.va()



#%%

im_va = img_as_float(io.imread(fname = va, as_gray = True))
image = img_as_float(io.imread(fname = vz, as_gray = True))

a = uneven_through(image, 0.363,0.780)
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

b0 = vstack(line_points_to_sets(points, sets0))
b1 = vstack(line_points_to_sets(points, sets1))
b2 = vstack(line_points_to_sets(points, sets2))

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
plt.legend(labels = ['Set 1', 'Set 2', 'Set 3'])
plt.xlabel('Distance from Set (μm)')
plt.ylabel('Charge (mV)')
plt.title('Charge against Distance to Set - 2V')
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
plt.legend(labels = ['0V', '0V'], frameon = True, fontsize=30)  
plt.xlim(0)
plt.ylim(0)  
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
plt.legend(labels = ['0V', '0V'],fontsize=30, frameon = True)    
plt.xlim(0)
plt.ylim(0) 
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
plt.legend(labels = ['0V', '0V'],fontsize=30, frameon = True)    
plt.xlim(0)
plt.ylim(0) 
plt.show()   

x1,y1 = plotting(east0,20)
x2,y2 = plotting(east1,20)
x3, y3 = plotting(east2, 20)
x4,y4 = plotting(west0,20)
x5,y5 = plotting(west1,20)
x6, y6 = plotting(west2, 20)     
columns = ['Set 1 - 0V', 'Set 2 - 0V', 'Set 3 - 0V', 'Set 1 - 0V', 'Set 2 - 0V', 'Set 3 - 0V']
fig, ax = plt.subplots(1, 1, figsize=(16,9), dpi= 80)
ax.fill_between(x = x1, y1=y1, y2=0, label=columns[0], alpha=0.5, color=mycolors[0], linewidth=2)
ax.fill_between(x = x2, y1=y2, y2=0, label=columns[1], alpha=0.5, color=mycolors[2], linewidth=2)
ax.fill_between(x = x3, y1=y3, y2=0, label=columns[2], alpha=0.5, color=mycolors[4], linewidth=2)
ax.fill_between(x = x4, y1=y4, y2=0, label=columns[3], alpha=0.6, color=mycolors[1], linewidth=2)
ax.fill_between(x = x5, y1=y5, y2=0, label=columns[4], alpha=0.6, color=mycolors[3], linewidth=2)
ax.fill_between(x = x6, y1=y6, y2=0, label=columns[5], alpha=0.6, color=mycolors[5], linewidth=2)
ax.set_title('0V and 0V, All Sets', fontsize=30)
ax.legend(loc='best', fontsize=20)
plt.xticks(fontsize=18, horizontalalignment='center')
plt.yticks(fontsize=18)
plt.xlim(0)
plt.ylim(0)
plt.xlabel('Distance from Set (μm)', fontsize = 20)
plt.ylabel('Charge Density (mV/pixel)', fontsize = 20)
plt.show()

#%%
import scipy as sp
import scipy.optimize as op
plt.figure(figsize=(16, 10), dpi= 80, facecolor='w', edgecolor='k')
x1,y1 = plotting(east2,20)
x2,y2 = plotting(west2,20)
#x1 = np.insert(x1, 0, 0)
#y1 = np.insert(y1, 0, 0)
plt.scatter(x1,y1, c=mycolors[4], s=150)
plt.scatter(x2,y2, c=mycolors[5], s=150)
plt.fill_between(x = x1, y1=y1, y2=0, alpha=0.5, color=mycolors[4], linewidth=2)
plt.fill_between(x = x2, y1=y2, y2=0, alpha=1, color=mycolors[5], linewidth=2)
plt.xlabel('Distance from Set (μm)', fontsize = 30)
plt.ylabel('Charge Density (mV/pixel)', fontsize = 30)
plt.title("Set 2", fontsize=22)
plt.xticks(fontsize=20); plt.yticks(fontsize=20)
plt.legend(labels = ['0V', '0V'],fontsize=30, frameon = True)  
plt.xlim(0)
plt.ylim(0 )   
fit,cov = sp.polyfit(x1,y1,13,cov=True)
fit_values = sp.poly1d(fit)
plt.plot(x1, fit_values(x1), c = 'orange', linewidth = 5)
fit_values = -1*sp.poly1d(fit)
op_x = op.fmin(fit_values, 0.1)
op_y = -1*fit_values(op_x)
plt.scatter(op_x, op_y, c = 'orange', s = 200)
plt.show()
