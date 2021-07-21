# -*- coding: utf-8 -*-
"""
Created on Wed Jul 7 14:34:54 2021

@author: sophi
"""
import gdal
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage import io
from skimage import img_as_float
import cv2 
import skimage
from skimage import io, img_as_float, img_as_ubyte, color, filters, feature, measure
from skimage.transform import rescale,resize, downscale_local_mean
from skimage.filters import gaussian, sobel, unsharp_mask, median, roberts, scharr, prewitt, threshold_multiotsu, threshold_otsu
from skimage.morphology import disk
from skimage.restoration import denoise_bilateral, denoise_nl_means, estimate_sigma, denoise_tv_chambolle
from skimage.color import rgb2gray, label2rgb
from skimage.segmentation import clear_border
import tifffile 
import glob
import os
import pandas as pd 


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
        
        Returns
        -------
        
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
        """Read TIFF File using scikit-image library
        
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
            specific = io.imread(specific, as_gray = gray)
            return specific
        
    def read_as_cv2(self, specific = None, gray = False):
        """Read TIFF File using cv2 library
        
        Parameters
        ----------
        specific : str name of TIFF file 
            If None (default), the four files will be read
            If not None, the specified file will be read 
        gray : True of False
            If False (default), imports in color
            If True , imports in grey scale
        """
        if gray is False:
            gray = 1
        if gray is True:
            gray = 0 
        if specific is None:
            self.__iz = cv2.imread(self.__iz, gray)
            self.__ia = cv2.imread(self.__ia, gray)
            self.__vz = cv2.imread(self.__vz, gray)
            self.__va = cv2.imread(self.__va, gray)
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
            if isinstance(self.__iz[0,0], np.floating ):
                self.convert_to(conversion = 'to_8bit')
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
            specific = Image.fromarray(specific)
            specific.show()
            return specific
    
    
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
        
    def convert_to(self, conversion = 'to_float' ,specific = None):
        """ Converts from 0-255 range to float (0-1) range or other way
        
        Parameters
        conversion : to_float or to_8bit
        specific : numpy.ndarray
            If None (defalut), will convert all arrays to float
            If not None, will convert specified array
        """
        if specific is None:
            if isinstance(self.__iz, str):
                self.read_as_skimage()
            if conversion == 'to_float': 
                self.__iz = img_as_float(self.__iz)
                self.__ia = img_as_float(self.__ia)
                self.__vz = img_as_float(self.__vz)
                self.__va = img_as_float(self.__va)
            if conversion == 'to_8bit':
                self.__iz = img_as_ubyte(self.__iz)
                self.__ia = img_as_ubyte(self.__ia)
                self.__vz = img_as_ubyte(self.__vz)
                self.__va = img_as_ubyte(self.__va)
            return self.__iz, self.__ia, self.__vz, self.__va
        else:
            if isinstance(specific, str):
                specific = self.read_as_skimage(specific = specific)
            if conversion == 'to_float':
                array = img_as_float(specific)
            if conversion == 'to_8bit' :
                array = img_as_ubyte(specific)
            specific = array
            return specific
    
    def sharpen_edges_with_unsharpen_mask(self, r, a, specific = None):
        """ Sharpens the edges of the image by subtracting background mask
        
        Parameters
        r : float or integer
            Is the radius circle of pixels blurred to create the mask 
        a : float or integer
            If the multiplication of the original-blurred
        specific : numpy.ndarray
            If None (defalut), will convert all arrays to float
            If not None, will convert specified array
        """
        if specific is None:
            if isinstance(self.__iz, str):
                self.read_as_skimage()
            self.__iz = unsharp_mask(self.__iz,radius=r,amount=a)
            self.__ia = unsharp_mask(self.__ia,radius=r,amount=a)
            self.__vz = unsharp_mask(self.__vz,radius=r,amount=a)
            self.__va = unsharp_mask(self.__va,radius=r,amount=a)
        else:
            if isinstance(specific, str):
                specific = self.read_as_skimage(specific = specific) 
            sharpened = unsharp_mask(specific, radius=r, amount=a)
            specific = sharpened
            return specific
        
    def line_graph(self, direction, pixel, graph = True):
        """ Generates a horizontal or vertical line graph of Z height or EFM 
            at specified pixel 
       ######## ADD SPECIFC and UBYTE OPTION and diagram option
        Parameters
        direction : str
            horizontal or vertical
            Dictates the direction of the line
        pixel : integer between 0 and 256 
            Decided the point on the y or x axis that dictates the start of 
            the line
        graph : boolean
            If True (default), will show the graph after the function is called
            If False, will not show the graph after the function is called

        Returns
        -------
        Line Graph

        """
        x = list(range(0,256))
        if direction == 'h':
            self.__iz = self.__iz[pixel]
            self.__ia = self.__ia[pixel]
            self.__vz = self.__vz[pixel]
            self.__va = self.__va[pixel]
        if direction == 'v':
            self.__iz = self.__iz[:,pixel]
            self.__ia = self.__ia[:,pixel]
            self.__vz = self.__vz[:,pixel]
            self.__va = self.__va[:,pixel]
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows = 2, ncols=2)
        ax1.plot(x, self.__iz)
        ax2.plot(x, self.__ia)
        ax3.plot(x, self.__vz)
        ax4.plot(x, self.__va)
        if graph == True:
            plt.show()
        
    def denoise_gaussian(self, auto = True, sig = [], to_float = True, specific = None):
        """ Denoises using skimage gaussian package
        ###Type doesn't matter but later many skimage want float so also converts to float

        Parameters
        sig : array or list
            Sigma values for each one
        to_float : boolean
            If True (default), function also converts image array to float
            If False, doesn't do converion'
        specific : numpy.ndarray
            If None (defalut), will convert all arrays to float
            If not None, will convert specified array
       
        Returns
        -------
        Line Graph
        """
        if specific is None:
            if isinstance(self.__iz, str):
                self.read_as_skimage()
            if to_float is True:
                self.convert_to(conversion = 'to_float')
            if auto is True:
                sig = np.array([
                    np.mean(estimate_sigma(self.__iz, multichannel=True)),
                    np.mean(estimate_sigma(self.__ia, multichannel=True)),
                    np.mean(estimate_sigma(self.__vz, multichannel=True)), 
                    np.mean(estimate_sigma(self.__va, multichannel=True))
                    ])
            self.__iz = gaussian(self.__iz, sigma= sig[0], mode='constant', cval=0.0)
            self.__ia = gaussian(self.__ia, sigma=sig[1], mode='constant', cval=0.0)
            self.__vz = gaussian(self.__vz, sigma=sig[2], mode='constant', cval=0.0)
            self.__va = gaussian(self.__va, sigma=sig[3], mode='constant', cval=0.0)
            return self.__iz, self.__ia, self.__vz, self.__va
        else:
           if isinstance(specific, str):
                specific = self.read_as_skimage(specific=specific)
           if to_float is True:
                specific = self.convert_to(conversion = 'to_float', specific = specific)
           if auto is True:
                sig = np.array([
                    np.mean(estimate_sigma(specific, multichannel=True))])
           specific = gaussian(specific, sigma= sig[0], mode='constant', cval=0.0)
           return specific
   
    def denoise_median(self, auto = True, r = [], to_float = True, specific = None ):
        """ Denoises using skimage median function using a disk of radius r

        Parameters
        r : array of the 4 sigma values
            radius of disk for denoising 
        to_float : boolean
            If True (default), function also converts image array to float
            If False, doesn't do converion'
        specific : numpy.ndarray
            If None (defalut), will convert all arrays to float
            If not None, will convert specified array
       
        Returns
        -------
        Line Graph
        median(image, disk(3), mode='constant', cval=0.0)
        """
        if specific is None:
            if isinstance(self.__iz, str):
                self.read_as_skimage()
            if to_float is True:
                self.convert_to(conversion = 'to_float')
            if auto is True:
                r = [1,1,1,1]
            self.__iz = median(self.__iz, disk(r[0]), mode='constant', cval=0.0)
            self.__ia = median(self.__ia, disk(r[1]), mode='constant', cval=0.0)
            self.__vz = median(self.__vz, disk(r[2]), mode='constant', cval=0.0)
            self.__va = median(self.__va, disk(r[3]), mode='constant', cval=0.0)
            return self.__iz, self.__ia, self.__vz, self.__va
        else:
           if isinstance(specific, str):
                specific = self.read_as_skimage(specific=specific)
           if to_float is True:
                specific = self.convert_to(conversion = 'to_float', specific = specific)
           if auto is True:
                r= [1]
           specific = median(specific, disk(r[0]), mode='constant', cval=0.0)
           return specific
       
    def denoise_bilateral(self, auto = True, color_sig = [], distance_sig = [], to_float = True, specific = None):
        """
        Denoise the images using bilateral function from skimage

        Parameters
        ----------
        auto : BOOLEAN, optional
            DESCRIPTION. The default is True.
            If True, the function automatcially generates the values of sigma
            If False, values of sigma can be manually entered into function
        color_sig : NUMPY ARRAY or LIST, optional
            DESCRIPTION. The default is [].
            Values for the sigma of the denoising in relation to the color difference
            larger sigma means averaging a larger number of color levels
        distance_sig : NUMPY ARRAY or LIST, optional
            DESCRIPTION. The default is [].
            Values for the sigma of the denoising in relation to distance 
            Larger sigma means averaging a larger number of pixels 
        to_float : BOOLEAN, optional
            DESCRIPTION. The default is True.
            If True, converts the array of values to floats
            If False, doesn't make the conversion - note function used requires float values
        specific : STRING or ARRAY, optional
            DESCRIPTION. The default is None.
            Can be used to complete function on only one 

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        if specific is None:
            if isinstance(self.__iz, str):
                self.read_as_skimage()
            if to_float is True:
                self.convert_to(conversion = 'to_float')
            if auto is True:
                distance_sig = np.array([
                    np.mean(estimate_sigma(self.__iz, multichannel=True)),
                    np.mean(estimate_sigma(self.__ia, multichannel=True)),
                    np.mean(estimate_sigma(self.__vz, multichannel=True)), 
                    np.mean(estimate_sigma(self.__va, multichannel=True))
                    ])
                color_sig = [0.05, 0.05, 0.05, 0.05]
            self.__iz = denoise_bilateral(self.__iz, 
                                          sigma_color=color_sig[0],
                                          sigma_spatial=distance_sig[0], 
                                          multichannel=False)
            self.__ia = denoise_bilateral(self.__ia, 
                                          sigma_color=color_sig[1], 
                                          sigma_spatial=distance_sig[1], 
                                          multichannel=False)
            self.__vz = denoise_bilateral(self.__vz, 
                                          sigma_color=color_sig[2], 
                                          sigma_spatial=distance_sig[2], 
                                          multichannel=False)
            self.__va = denoise_bilateral(self.__va, 
                                          sigma_color=color_sig[3], 
                                          sigma_spatial=distance_sig[3], 
                                          multichannel=False)
            return self.__iz, self.__ia, self.__vz, self.__va
        else:
           if isinstance(specific, str):
                specific = self.read_as_skimage(specific=specific)
           if to_float is True:
                specific = self.convert_to(conversion = 'to_float', specific = specific)
           if auto is True:
                 distance_sig = np.array([
                    np.mean(estimate_sigma(specific, multichannel=True))])
                 color_sig = [0.5]
           specific = denoise_bilateral(specific, 
                                          sigma_color=color_sig[0], 
                                          sigma_spatial=distance_sig[0], 
                                          multichannel=False)
           return specific

    def denoise_nlm(self, specific = None):
        
        nlm_using_skimage = denoise_nl_means(image, h=1.15*sigma_est, fast_mode=True, patch_size=5, patch_distance=3, multichannel=False)
        
        
        
        
        
        