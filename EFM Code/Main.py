# -*- coding: utf-8 -*-
"""
Created on Wed Jul 7 14:34:54 2021

@author: sophi
"""
import gdal
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

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

    def read(self, specific = None):
        """Read TIFF File 
        
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
    
    def image(self, specific = None):
        """Open arrays as an image
        
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
    
    #def invert(self, )
