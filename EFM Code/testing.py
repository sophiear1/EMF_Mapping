# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 12:36:49 2021

@author: sophi
"""

import main

#%% Create object using P3HT 11 
p3ht = main.molecule(
    iz = 'P3HT 58k 11 5um 0V_190208_Z Height_Forward_001.tiff', 
    ia = 'P3HT 58k 11 5um 0V_190208_EFM Amplitude_Forward_001.tiff', 
    vz = 'P3HT 58k 11 5um EFM 2V_190208_Z Height_Forward_003.tiff', 
    va = 'P3HT 58k 11 5um EFM 2V_190208_EFM Amplitude_Forward_003.tiff'
    )

#%%
arr = p3ht.central_moments(2)
print(arr)
#%%
import numpy
a = numpy.ma.core.MaskedArray()
if isinstance(a, list):
    a.append(['is_list'])
if isinstance(a, numpy.ndarray):
    numpy.append(a, ['is_array'])
print(a)
#%%
p3ht.uneven_sections(dividers = [[0.363, 0.78]])
p3ht.image_masked_arrays(cmap = 'magma')

#%%
#p3ht.read_as_skimage()
#iz = p3ht.iz()
#p3ht.convert_to()
#p3ht.sections(no_of_sections = 3, split = 'Full_Low')
#b = isinstance(iz[0][0], np.uint8)
#print(b)

a,b,c,d = p3ht.uneven_sections(dividers = [[0.2, 0.4, 0.6, 0.8]])

e = p3ht.central_moments(p=2, specific = a[0])

#%%
p3ht.line_graph('h', 155)


#%%
a,b,c,d = p3ht.frequency_domain()
print(a)

#%%
p3ht.frequency_domain()
p3ht.high_pass_filter()
#p3ht.image_frequency_domain()
#p3ht.image_show(four=True, cmap = 'magma')
#main.plt.show()
p3ht.spatial_domain()
p3ht.image_show(four=True, cmap = 'winter')
main.plt.show()
#%%
a,b,c,d = p3ht.find_contours()
p3ht.read_as_skimage()
#main.plt.imshow(p3ht.iz(), cmap = 'magma')
for con in a:
    main.plt.plot(con[:,1],con[:,0])

#%% Read as gdal and be array for all 4 files (should work with either hashed out)
gdal = p3ht.read_as_gdal()
array = p3ht.bearray()
print(gdal)
print(array)
#%%Read as Image
Image = p3ht.read_as_skimage()
print(Image)
#%%
p3ht.image()
print(p3ht) #should be <main.molecule object at ..location...>
print(type(p3ht)) #should be <class 'main.molecule'>
#%%
cv2 = p3ht.read_as_cv2()
print(cv2, p3ht.iz())
#%% Fetching individual files 
iz = p3ht.iz()
ia = p3ht.ia()
vz = p3ht.vz()
va = p3ht.va()
print(iz) #if after bearray() should be array (256,256) of greyscale pixels ranging between 1 and 255
print(type(iz)) #if after bearray() should be <class 'numpy.ndarray'>
#%% Testing specific function - don't run directly after above as bug since cannot call bearray then read
iz = p3ht.iz()
print(iz) 
read_iz = p3ht.read_as_gdal(specific = iz)
print(read_iz) # should be the same as above
array_from_read = p3ht.bearray(specific = read_iz)
array_from_iz = p3ht.bearray(specific = iz)
print(array_from_read)
print(array_from_iz) #both should be the same array
image_from_iz = p3ht.image(specific = iz)
image_from_array = p3ht.image(specific = array_from_iz)
image_from_read = p3ht.image(specific = read_iz)
#will automatically open the 3 images, which should all be the same 
#%% Testing inversion
p3ht.bearray()
p3ht.invert_np()
p3ht.image()
#%% Develop so can specify more than one? + Bug only works after inital load in of molecule
vz = p3ht.iz()
va = p3ht.va()
array_va = p3ht.bearray(specific = va)
invert_va = p3ht.invert_np(specific = array_va)
image_va = p3ht.image(specific = invert_va)
image_vz = p3ht.image(specific = vz)