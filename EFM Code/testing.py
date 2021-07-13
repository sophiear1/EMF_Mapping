# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 12:36:49 2021

@author: sophi
"""

import main 
#%% Create object using P3HT 11 
p3ht = main.molecule(iz = 'P3HT 58k 11 5um 0V_190208_Z Height_Forward_001.tiff', ia = 'P3HT 58k 11 5um 0V_190208_EFM Amplitude_Forward_001.tiff', vz = 'P3HT 58k 11 5um EFM 2V_190208_Z Height_Forward_003.tiff', va = 'P3HT 58k 11 5um EFM 2V_190208_EFM Amplitude_Forward_003.tiff')
#%% Read and be array for all 4 files (should work with either hashed out)
p3ht.read()
p3ht.bearray()
#%%
p3ht.image()
print(p3ht) #should be <main.molecule object at ..location...>
print(type(p3ht)) #should be <class 'main.molecule'>
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
read_iz = p3ht.read(specific = iz)
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
p3ht.invert()
p3ht.image()
#%% Develop so can specify more than one? + Bug only works after inital load in of molecule
vz = p3ht.iz()
va = p3ht.va()
array_va = p3ht.bearray(specific = va)
invert_va = p3ht.invert(specific = array_va)
image_va = p3ht.image(specific = invert_va)
image_vz = p3ht.image(specific = vz)