# -*- coding: utf-8 -*-
"""
Created on Wed Jul 7 14:34:54 2021

Class Requirements = gdal, numpy, PIL, matplotlib, skimage, cv2, tifffile, glob, os, pandas
will import but may need update 

@author: sophi
"""
import gdal
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import scipy as sp
from scipy import ndimage as ndi
import cv2 
import tifffile 
import glob
import os
import pandas as pd 
import skimage
import skimage.feature
import skimage.viewer
import skimage.data as data
import skimage.segmentation as seg
from skimage import io, img_as_float, img_as_ubyte, color, filters, feature, measure, draw, exposure, util, morphology, segmentation
from skimage.transform import rescale,resize, downscale_local_mean
from skimage.filters import gaussian, sobel, unsharp_mask, median, roberts, scharr, prewitt, threshold_multiotsu, threshold_otsu
from skimage.morphology import disk
from skimage.restoration import denoise_bilateral, denoise_nl_means, estimate_sigma, denoise_tv_chambolle
from skimage.color import rgb2gray, label2rgb
from skimage.segmentation import clear_border

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
        iz : STRING
            DESCRIPTION. 
            Name of TIFF file
            Z height with no voltage
        ia : STRING
            DESCRIPTION. 
            Name of TIFF file
            EFM amplitude with no voltage
        vz : STRING
            DESCRIPTION. 
            Name of TIFF file
            Z height with voltage 
        za : STRING 
            DESCRIPTION.
            Name of TIFF file
            EFM amplitude with voltage 
        
        Returns
        -------
        
        """
        self._iz = iz
        self._ia = ia
        self._vz = vz
        self._va = va
        self.__iz = iz
        self.__ia = ia
        self.__vz = vz
        self.__va = va
        self.__all = np.array([self.__iz, self.__ia, self.__vz, self.__va])
    
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
        """
        Read TIFF File using gdal library
        
        Parameters
        ----------
        specific : STRING, optional
            DESCRIPTION. The default is None
            Name of TIFF file 
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
        """
        Read TIFF File using Image library
        
        Parameters
        ----------
        specific : STRING, 
            DESCRIPTION. The default is None
            Name of TIFF file 
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
        """
        Read TIFF File using scikit-image library
        
        Parameters
        ----------
        specific : STRING, optional
            DESCRIPTION. The default is None
            Name of TIFF file 
            If None (default), the four files will be read
            If not None, the specified file will be read 
        gray : BOOLEAN, optional
            DESCRIPTION. The default is False
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
        """
        Read TIFF File using cv2 library
        
        Parameters
        ----------
        specific : STRING, optional
            DESCRIPTION. The default is None
            Name of TIFF file
            If None (default), the four files will be read
            If not None, the specified file will be read 
        gray : BOOLEAN, optional
            DESCRIPTION. The default is False
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
        """
        Take the read TIFF File and convert to an array
        
        Parameters
        ----------
        specific : STRING, optional
            DESCRIPTION. The default is None
            Name of TIFF file or open gdal object
            If None (default), all four files converted to numpy.ndarray
            If not None, data from specified file converted to numpy.ndarray

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
        """
        Open arrays as an image using PIL Image from array function
        
        Parameters
        ----------
        specific : ARRAY or STRING, optional
            DESCRIPTION. The default is None
            If None (default), opens all 4 arrays as separated images
            If not None, opens specified file/array as an image

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

    def image_show(self, image = None, nrows=1, ncols=1, cmap='gray', four = False, **kwargs):
        """ 
        Show the image using plt imshow function with the axes labels turned off
        
        Parameters
        ----------
        image : ARRAY
            DESCRIPTION
            the array of the image that should be shown
        nrows : INTEGER
            DESCRIPTION. The default is 2
            number of rows required
        ncols : INTEGER
            DESCRIPTION. The default is 2
            number of coloumns required
        cmap : STRING
            DESCRIPTION. The default is gray
            ideal colour map for the image
            
        Returns
        -------
        fig
            DESCRIPTION.
        ax
        
        """
        if four == True:
            fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(nrows=2, ncols=2, figsize=(16, 16))
            ax0.imshow(self.__iz, cmap = cmap)
            ax0.axis('off')
            ax1.imshow(self.__ia, cmap = cmap)
            ax1.axis('off')
            ax2.imshow(self.__vz, cmap = cmap)
            ax2.axis('off')
            ax3.imshow(self.__va, cmap = cmap)
            ax3.axis('off')
            return fig, ax0, ax1, ax2, ax3

        else:
            fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(16, 16))
            ax.imshow(image, cmap=cmap)
            ax.axis('off')
            return fig, ax
    
    def invert_np(self, specific = None):
        """ 
        Inverts the array unsing numpy invert function
        
        Parameters
        ----------
        specific : ARRAY, optional
            DESCRIPTION. The default is None
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
        """ 
        Converts from 0-255 range to float (0-1) range or other way
        
        Parameters
        ----------
        conversion : STRING
            DESCRIPTION
            If 'to_float' (default), will convert all values in the array 
            to a float
            If 'to_8bit', will convert all values within the array to 8bit
        specific : ARRAY, optional
            DESCRIPTION. The default is None
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
        ----------
        r : FLOAT or INTEGER
            DESCRIPTION
            Is the radius circle of pixels blurred to create the mask 
        a : FLOAT or INTEGER
            DESCRIPTION
            If the multiplication of the original-blurred
        specific : ARRAY or STRING, optional
            DESCRIPTION, The default is None
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
        
    def sharpen_edges_with_unsharpen_mask(self, r, a, specific = None):
        """ Sharpens the edges of the image by subtracting background mask
        
        Parameters
        ----------
        r : FLOAT or INTEGER
            DESCRIPTION
            Is the radius circle of pixels blurred to create the mask 
        a : FLOAT or INTEGER
            DESCRIPTION
            If the multiplication of the original-blurred
        specific : ARRAY or STRING, optional
            DESCRIPTION, The default is None
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
        """ 
        Generates a horizontal or vertical line graph of Z height or EFM 
        at specified pixel 
################ ADD SPECIFC and UBYTE OPTION and diagram option

        Parameters
        ----------
        direction : STRING
            DESCRIPTION.
            If 'h', will read horizontally
            If 'v', will read vertically
        pixel : INTEGER, in range 0 to 256 
            DESCRIPTION.
            Dictates the point on the y or x axis that dictates the start of 
            the line
        graph : BOOLEAN, optional
            DESCRIPTION. The default is True
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
        """ 
        Denoises using skimage gaussian package
############TYPE doesn't matter but later many skimage want float so also converts to float

        Parameters
        ----------
        auto : BOOLEAN, optional
            DESCRIPTION. The defualt is True
            If True (default), will automatically decide values for sigma
            If False, the manually entered values can be taken
        sig : ARRAY or LIST, optional if auto is True
            DESCRIPTION.
            Sigma values for each image to be denoised so if specific = False, 
            the 4 values needed
        to_float : BOOLEAN, optional
            DESCRIPTION.  The default is True
            If True (default), function also converts image array to float
            If False, doesn't do converion
        specific : ARRAY or STRING, optional
            DESCRIPTION. The default is None
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
        """ 
        Denoises using skimage median function using a disk of radius r

        Parameters
        ----------
        auto : BOOLEAN, optional
            DESCRIPTION. The defualt is True
            If True (default), will automatically decide values for r
            If False, the manually entered values can be taken
        r : ARRAY, optional is auto is True
            DESCRIPTION. The default is [1,1,1,1]
            The 4 sigma values to be the radius of disk for denoising 
        to_float : BOOLEAN, optional
            DESCRIPTION. The default is True
            If True (default), function also converts image array to float
            If False, doesn't do converion'
        specific : ARRAY or STRING, optional
            DESCRIPTION. The default is None
            If None (defalut), will convert all arrays to float
            If not None, will convert specified array
       
        Returns
        -------
        
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

    def denoise_nlm(self,  auto = True, sig = [], to_float = True, specific = None):
        """
        Denoise the images using nlm function from skimage

        Parameters
        ----------
        auto : BOOLEAN, optional
            DESCRIPTION. The default is True.
            If True, the function automatcially generates the values of sigma
            If False, values of sigma can be manually entered into function
        sig : NUMPY ARRAY or LIST, optional
            DESCRIPTION. The default is [].
            Values for the sigma of the denoising 
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
                sig = np.array([
                    np.mean(estimate_sigma(self.__iz, multichannel=True)),
                    np.mean(estimate_sigma(self.__ia, multichannel=True)),
                    np.mean(estimate_sigma(self.__vz, multichannel=True)), 
                    np.mean(estimate_sigma(self.__va, multichannel=True))
                    ])
            self.__iz = denoise_nl_means(self.__iz, 
                                         h=1.15*sig[0], 
                                         fast_mode=True, 
                                         patch_size=5, patch_distance=3, 
                                         multichannel=False)
            self.__ia = denoise_nl_means(self.__va, 
                                         h=1.15*sig[1], 
                                         fast_mode=True, 
                                         patch_size=5, patch_distance=3, 
                                         multichannel=False)
            self.__vz = denoise_nl_means(self.__vz, 
                                         h=1.15*sig[2], 
                                         fast_mode=True, 
                                         patch_size=5, patch_distance=3, 
                                         multichannel=False)
            self.__va = denoise_nl_means(self.__va, 
                                         h=1.15*sig[3], 
                                         fast_mode=True, 
                                         patch_size=5, patch_distance=3, 
                                         multichannel=False)
            return self.__iz, self.__ia, self.__vz, self.__va
        else:
           if isinstance(specific, str):
                specific = self.read_as_skimage(specific=specific)
           if to_float is True:
                specific = self.convert_to(conversion = 'to_float', specific = specific)
           if auto is True:
                sig = np.array([
                    np.mean(estimate_sigma(specific, multichannel=True))])
           specific = denoise_nl_means(specific, 
                                         h=1.15*sig[0], 
                                         fast_mode=True, 
                                         patch_size=5, patch_distance=3, 
                                         multichannel=False)
           return specific
     
    def denoise_total_variance(self, auto = True, weight = [], to_float = True, specific = None):
        """
        Denoise the images using nlm function from skimage

        Parameters
        ----------
        auto : BOOLEAN, optional
            DESCRIPTION. The default is True.
            If True, the function automatcially generates the values of sigma
            If False, values of sigma can be manually entered into function
        weight : NUMPY ARRAY or LIST, optional
            DESCRIPTION. The default is [].
            Values for the weight of denoising
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
                weight = [0.1,0.1,0.1,0.1]
            self.__iz = denoise_tv_chambolle(self.__iz, 
                                             weight=weight[0], eps=0.0002,
                                             n_iter_max=200, multichannel=False)
            self.__ia = denoise_tv_chambolle(self.__ia, 
                                             weight=weight[1], eps=0.0002,
                                             n_iter_max=200, multichannel=False)
            self.__vz = denoise_tv_chambolle(self.__vz, 
                                             weight=weight[2], eps=0.0002,
                                             n_iter_max=200, multichannel=False)
            self.__va = denoise_tv_chambolle(self.__va, 
                                             weight=weight[3], eps=0.0002,
                                             n_iter_max=200, multichannel=False)
            return self.__iz, self.__ia, self.__vz, self.__va
        else:
           if isinstance(specific, str):
                specific = self.read_as_skimage(specific=specific)
           if to_float is True:
                specific = self.convert_to(conversion = 'to_float', specific = specific)
           if auto is True:
                weight = [0.1]
           specific = denoise_tv_chambolle(specific, 
                                             weight=weight[0], eps=0.0002,
                                             n_iter_max=200, multichannel=False)
           return specific
       
    def basic_edge_detection(self, function = 'sobel', specific = None):
        """
        Basic edge dection using the users choice of edge filter 
        Choices are Roberts, Sobel, Scharr, and Prewitt

        Parameters
        ----------
        function : STRING, optional
            DESCRIPTION. The default is 'sobel'.
            If 'sobel' (default), edge detection using sobel parmeters
            If 'roberts', edge detection using roberts parmeters
            If 'scharr', edge detection using scharr parmeters
            If 'prewitt', edge detection using prewitt parmeters
        specific : STRING or ARRAY, optional
            DESCRIPTION. The default is None.
            Can be used to complete function on only one 
        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        edges_str = ['roberts', 'sobel', 'scharr', 'prewitt']
        edges = [roberts, sobel, scharr, prewitt]
        if specific is None:
            if isinstance(self.__iz, str):
                self.read_as_skimage(gray = True)
            index = edges_str.index(function)
            self.__iz = edges[index](self.__iz)
            self.__ia = edges[index](self.__ia)
            self.__vz = edges[index](self.__vz)
            self.__va = edges[index](self.__va)
            return self.__iz, self.__ia, self.__vz, self.__va
        else: 
            if isinstance(specific, str):
                specific = self.read_as_skimage(specific = specific, gray = True)
            index = edges_str.index(function)
            specific = edges[index](specific)
            return specific
       
    def canny_edges(self, option = 'auto', sig = [], lower = [], upper = [], specific = None):
        """
        Uses Canny edge decetion to determine the loactions of the edges in the image

        Parameters
        ----------
        option : STRING, optional
            DESCRIPTION. The default is 'auto'.
            If 'auto' (default), function automatically decides vaues for Canny function
            If 'semi', values for sigma need to be entered in the sig = part of the function
            If 'manual', values for lower and upper need to entered
        sig : ARRAY or LIST, required if option = 'semi'
            DESCRIPTION. The default is [].
            Values for the sigma of each image
        lower : ARRAY or LIST, required if option = 'manual'
            DESCRIPTION. The default is [].
            Values for the lower part of function
        upper : ARRAY or LIST, required is option = 'manual'
            DESCRIPTION. The default is [].
            Value for the higher part of the function
        specific : STRING or ARRAY, optional
            DESCRIPTION. The default is None.
            Can be used to complete function on only one
        Returns
        -------
        None.

        """
        if specific is None:
            if isinstance(self.__iz, str):
                self.read_as_cv2(gray = True)
            if isinstance(self.__iz[0,0], np.floating ):
                self.convert_to(conversion = 'to_8bit')
#            self.__ia = 255 - self.__ia
#            self.__va = 255 - self.__va
            if option == 'auto':
                self.convert_to(conversion = 'to_float')
                sig = np.array([
                    np.mean(estimate_sigma(self.__iz, multichannel=True)),
                    np.mean(estimate_sigma(self.__ia, multichannel=True)),
                    np.mean(estimate_sigma(self.__vz, multichannel=True)), 
                    np.mean(estimate_sigma(self.__va, multichannel=True))
                    ])
                self.convert_to(conversion = 'to_8bit')
            if option == 'auto' or 'semi':
                self.convert_to(conversion = 'to_float')
                median = np.array([
                    np.median(self.__iz), np.median(self.__ia), 
                    np.median(self.__vz), np.median(self.__va)
                    ])
                self.convert_to(conversion = 'to_8bit')
                lower = np.array([
                    int(max(1, (1-sig[0])*median[0])), 
                    int(max(1, (1-sig[1])*median[1])),
                    int(max(1, (1-sig[2])*median[2])),
                    int(max(1, (1-sig[3])*median[3]))
                    ])
                upper = np.array([
                    int((1+sig[0])*median[0]),
                    int((1+sig[1])*median[1]),
                    int((1+sig[2])*median[2]),
                    int((1+sig[3])*median[3])
                    ])
            self.__iz = cv2.Canny(self.__iz, lower[0], upper[0])
            self.__ia = cv2.Canny(self.__ia, lower[1], upper[1])
            self.__vz = cv2.Canny(self.__vz, lower[2], upper[2])
            self.__va = cv2.Canny(self.__va, lower[3], upper[3])
            return self.__iz, self.__ia, self.__vz, self.__va
        
      
    def frequency_domain(self):
        """
        Converts the image to the frequency domain
        
        Parameters
        ----------
        r : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        def f_d(i):
            dft = cv2.dft(np.float32(i), flags=cv2.DFT_COMPLEX_OUTPUT)
            dft_shift = np.fft.fftshift(dft)
            return dft_shift
        self.__iz = f_d(self.__iz)
        self.__ia = f_d(self.__ia)
        self.__vz = f_d(self.__vz)
        self.__va = f_d(self.__va)
        return self.__iz, self.__ia, self.__vz, self.__va
    
    def image_frequency_domain(self):
        """
        Generates the images of the Z height and EFM in the frequency domain
        
        Parameters
        ----------
        r : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        def m_s(i):
            magnitude_spectrum = 20 * np.log((cv2.magnitude(i[:, :, 0], i[:, :, 1]))+1)
            return magnitude_spectrum
        self.__iz = m_s(self.__iz)
        self.__ia = m_s(self.__ia)
        self.__vz = m_s(self.__vz)
        self.__va = m_s(self.__va)
        return self.__iz, self.__ia, self.__vz, self.__va
        
    def high_pass_filter(self, r):
        """
        Passes the images through a high pass filter

        Parameters
        ----------
        r : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        def h_p_f(oi, r):
            rows, cols = oi.shape
            crow, ccol = int(rows / 2), int(cols / 2)
            mask = np.ones((rows, cols, 2), np.uint8)
            center = [crow, ccol]
            x, y = np.ogrid[:rows, :cols]
            mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r*r
            mask[mask_area] = 0
            return mask
        mask_iz = h_p_f(self.read_as_cv2(specific = self._iz, gray = True), r) 
        self.__iz = mask_iz *self.__iz
        mask_ia = h_p_f(self.read_as_cv2(specific = self._ia, gray = True), r)
        self.__ia = mask_ia*self.__ia
        mask_vz = h_p_f(self.read_as_cv2(specific = self._vz, gray = True), r)
        self.__vz = mask_vz*self.__vz
        mask_va = h_p_f(self.read_as_cv2(specific = self._va, gray = True), r)
        self.__va = mask_va*self.__vz
        return self.__iz, self.__ia, self.__vz, self.__va
        
    def spatial_domain(self):
        """
        Converts from the frequency domain to the spatial domain
        
        Parameters
        ----------
        r : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        def s_d(i):
            f_ishift = np.fft.ifftshift(i)
            img_back = cv2.idft(f_ishift)
            img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
            return img_back
        self.__iz = s_d(self.__iz)
        self.__ia = s_d(self.__ia)
        self.__vz = s_d(self.__vz)
        self.__va = s_d(self.__va)
        return self.__iz, self.__ia, self.__vz, self.__va
    
    def histogram_equalisation(self, specific = None):
        """
        Edge enhancement using histogram equalisation 
        
        Parameters
        ----------
        r : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        if specific is None:
            if isinstance(self.__iz, str):
                self.read_as_cv2(gray = False)
            if isinstance(self.__iz[0,0], np.floating ):
                self.convert_to(conversion = 'to_8bit')
            def h_e(i):
                lab_img= cv2.cvtColor(i, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab_img)
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
                clahe_img = clahe.apply(l)
                updated_lab_img2 = cv2.merge((clahe_img,a,b))
                CLAHE_img = cv2.cvtColor(updated_lab_img2, cv2.COLOR_LAB2BGR)
                CLAHE_img = cv2.cvtColor(CLAHE_img, cv2.COLOR_BGR2GRAY)
                return CLAHE_img
            self.__iz = h_e(self.__iz)
            self.__ia = h_e(self.__ia)
            self.__vz = h_e(self.__vz)
            self.__va = h_e(self.__va)
            return self.__iz, self.__ia, self.__vz, self.__va

    def find_contours(self):
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
            t = filters.threshold_li(image_g)
            thresholded = (image_g <= t)
            #distance map
            distance = ndi.distance_transform_edt(thresholded)
            local_maxima = morphology.local_maxima(distance)
            markers = ndi.label(local_maxima)[0]
            labels_masked = segmentation.watershed(thresholded,markers,mask = thresholded, connectivity = 2)
            contours = measure.find_contours(labels_masked,level = t)
            return contours
        contours_iz = f_c(self.__iz)
        contours_ia = f_c(self.__ia)
        contours_vz = f_c(self.__vz)
        contours_va = f_c(self.__va)
        return contours_iz , contours_ia , contours_vz , contours_va 
    
            
    
        
    
        
  
    