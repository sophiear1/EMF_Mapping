# -*- coding: utf-8 -*-
"""
Created on Wed Jul 7 14:34:54 2021

@author: sophi
"""
import gdal
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import skimage
from skimage import io
from skimage import img_as_float


class molecule:
    """ Read data from TIFF file and hopefully analyse
    
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
        iz : str name of TIFF file
            Z height with no voltage
        ia : str name of TIFF file
            EFM amplitude with no voltage
        vz : str name of TIFF file
            Z height with voltage 
        za : str name of TIFF file
            EFM amplitude with voltage 
        """
        self.__iz = iz
        self.__ia = ia
        self.__vz = vz
        self.__va = va
    
    def iz(self):
        """Returns Z height with no voltage file"""
        return self.__iz
    def ia(self):
        """Returns EFM amplitude with no voltage file"""
        return self.__ia
    def vz(self):
        """Returns Z height with voltage file"""
        return self.__vz
    def va(self):
        """Returns EFM amplitude with voltage file"""
        return self.__va

    def read_as_gdal(self, specific = None):
        """Read TIFF File using gdal library
        
        Parameters
        ----------
        specific : str name of TIFF file 
            If None (default), the four files will be read
            If not None, the specified file will be read 
        """
        if specific is None:
            self.__iz = gdal.Open(self.__iz)
            self.__ia = gdal.Open(self.__ia)
            self.__vz = gdal.Open(self.__vz)
            self.__va = gdal.Open(self.__va)
            return self.__iz, self.__ia, self.__vz, self.__va
        else: 
            file = gdal.Open(specific)
            return file
        
    def read_as_Image(self, specific = None):
        """Read TIFF File using Image library
        
        Parameters
        ----------
        specific : str name of TIFF file 
            If None (default), the four files will be read
            If not None, the specified file will be read 
        """
        if specific is None:
            self.__iz = Image.open(self.__iz)
            self.__ia = Image.open(self.__ia)
            self.__vz = Image.open(self.__vz)
            self.__va = Image.open(self.__va)
            return self.__iz, self.__ia, self.__vz, self.__va
        else: 
            file = Image.open(specific)
            return file
        
    def read_as_skimage(self, specific = None, gray = False):
        """Read TIFF File using Image library
        
        Parameters
        ----------
        specific : str name of TIFF file 
            If None (default), the four files will be read
            If not None, the specified file will be read 
        gray : True of False
            If False (default), imports in color
            If True , imports in grey scale
        """
        if specific is None:
            self.__iz = io.imread(fname = self.__iz, as_gray = gray)
            self.__ia = io.imread(fname = self.__ia, as_gray = gray)
            self.__vz = io.imread(fname = self.__vz, as_gray = gray)
            self.__va = io.imread(fname = self.__va, as_gray = gray)
            return self.__iz, self.__ia, self.__vz, self.__va
        else: 
            file = io.imread(specific, as_gray = gray)
            return file
        

    def bearray(self, specific = None):
        """Take the read TIFF File and convert to an array
        
        Parameters
        specific : str name of TIFF file or open gdal object (e.g. p3ht.iz())
            If None (default), all four files converted to numpy.ndarray
            If not None, data from specified file converted to numpy.ndarray
        ----------
        """
        if specific is None:
            if isinstance(self.__iz, str):
                self.read_as_gdal()
            self.__iz = self.__iz.ReadAsArray()
            self.__ia = self.__ia.ReadAsArray()
            self.__vz = self.__vz.ReadAsArray()
            self.__va = self.__va.ReadAsArray()
            return self.__iz, self.__ia, self.__vz, self.__va
        else:
            if isinstance(specific, str):
                specific = self.read_as_gdal(specific = specific)
            array = specific.ReadAsArray()
            return array
    
    def image(self, specific = None):
        """Open arrays as an image using PIL Image from array function
        
        Parameters
        specific : numpy.ndarray, opened gdal file or str of TIFF file name
            If None (default), opens all 4 arrays as separated images
            If not None, opens specified file/array as an image
        ----------
        """
        if specific is None:
            if not isinstance(self.__iz, np.ndarray):
                self.bearray()
            self.__iz = Image.fromarray(self.__iz)
            self.__iz.show()
            self.__ia = Image.fromarray(self.__ia)
            self.__ia.show()
            self.__vz = Image.fromarray(self.__vz)
            self.__vz.show()
            self.__va = Image.fromarray(self.__va)
            self.__va.show()
            return self.__iz, self.__ia, self.__vz, self.__va
        else:
            if not isinstance(specific, np.ndarray):
                specific = self.bearray(specific = specific)
            image = Image.fromarray(specific)
            image.show()
            return image
    
    
    def invert_np(self, specific = None):
        """ Inverts the array unsing numpy invert function
        
        Parameters
        specific : numpy.ndarray
            If None (defalut), will invert both EFM arrays
            If not None, will invert specified array
        """
        if specific is None:
            self.__ia = np.invert(self.__ia)
            self.__va = np.invert(self.__va)
            return self.__ia, self.__va
        else:
            inversion = np.invert(specific)
            return inversion
        
    def convert_to_float(self, specific = None):
        """ Converts from 0-255 range to float (0-1) range 
        
        Parameters
        specific : numpy.ndarray
            If None (defalut), will convert all arrays to float
            If not None, will convert specified array
        """
        if specific is None:
            if isinstance(self.__iz, str):
                self.read_as_skimage()
            self.__iz = img_as_float(self.__iz)
            self.__ia = img_as_float(self.__ia)
            self.__vz = img_as_float(self.__vz)
            self.__va = img_as_float(self.__va)
            return self.__iz, self.__ia, self.__vz, self.__va
        else:
            if isinstance(specific, str):
                specific = self.read_as_skimage(specific = specific)
            array = img_as_float(specific)
            return array
        
    
        
    
        
    