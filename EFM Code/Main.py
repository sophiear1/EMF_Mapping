# -*- coding: utf-8 -*-
"""
Created on Wed Jul 7 14:34:54 2021

@author: sophi
"""
import gdal
import numpy as np

class molecule:
    """ Read data from TIFF file
    
    Attributes
    ----------
    """
    def __init__(
            self,
            iz = None, 
            ia = None,
            vz = None, 
            va = None
        ):
       
        """Initialise data
        
        Parameters
        ----------
        iz : TIFF file
            Z height with no voltage
        ia : TIFF file
            EFM amplitude with no voltage
        vz : TIFF file
            Z height with voltage 
        za : TIFF file
            EFM amplitude with voltage 
        """
        self.__iz = iz
        self.__ia = ia
        self.__vz = vz
        self.__va = va
    
    def iz(self):
        """Returns iz file"""
        return self.__iz
    def ia(self):
        """Returns ia file"""
        return self.__ia
    def vz(self):
        """Returns vz file"""
        return self.__vz
    def va(self):
        """Returns va file"""
        return self.__va

    def read(self, specific = None):
        """Read TIFF File 
        If specific = None then reads all 4 files
        If specific = 'parameter' then only reads that specific file
        Parameters
        ----------
        specific 
        """
        if specific == None:
            self.__iz = gdal.Open(self.__iz)
            self.__ia = gdal.Open(self.__ia)
            self.__vz = gdal.Open(self.__vz)
            self.__va = gdal.Open(self.__va)
            return self.__iz, self.__ia, self.__vz, self.__va
        else: 
            file = gdal.Open(specific)
            return file
     
    def bearray(self, specific = None):
        """Take the read TIFF File and convert to an array
        Parameters
        ----------
        """
        if specific == None:
            if isinstance(self.__iz, str):
                self.read()
            self.__iz = self.__iz.ReadAsArray()
            self.__ia = self.__ia.ReadAsArray()
            self.__vz = self.__vz.ReadAsArray()
            self.__va = self.__va.ReadAsArray()
            return self.__iz, self.__ia, self.__vz, self.__va
        else:
            if isinstance(specific, str):
                specific = self.read(specific = specific)
            array = specific.ReadAsArray()
            return array

#%% Testing 
p3ht = molecule(iz = 'P3HT 58k 11 5um 0V_190208_Z Height_Forward_001.tiff', ia = 'P3HT 58k 11 5um 0V_190208_EFM Amplitude_Forward_001.tiff', vz = 'P3HT 58k 11 5um EFM 2V_190208_Z Height_Forward_003.tiff', va = 'P3HT 58k 11 5um EFM 2V_190208_EFM Amplitude_Forward_003.tiff')
p3ht.read()
p3ht.bearray()
iz = p3ht.iz()
ia = p3ht.ia()
vz = p3ht.vz()
va = p3ht.va()
print(vz)
print(va)
#%%Attempt Mean Squared didn't work
err = np.sum((iz.astype("float") - vz.astype("float")) ** 2)
err /= float(iz.shape[0] * vz.shape[1])
print(err)
err2 = np.sum((ia.astype("float") - va.astype("float")) ** 2)
err2 /= float(ia.shape[0] * va.shape[1])
print(err2)
err3 = np.sum((ia.astype("float") - iz.astype("float")) ** 2)
err3 /= float(ia.shape[0] * iz.shape[1])
print(err3)
err4 = np.sum((va.astype("float") - vz.astype("float")) ** 2)
err4 /= float(va.shape[0] * vz.shape[1])
print(err4)
#%%
def average(a,n):
    sum=0
    for i in range(n):
        sum += a[i]
    return sum/n
print(average(iz, len(iz)))
#%%
print(p3ht)
print(type(p3ht))
print(type(iz))
print(iz)
p3ht.read(specific = p3ht.iz())
a = p3ht.bearray(specific = p3ht.vz())
print(a)
#%%
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
