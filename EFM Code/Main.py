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
from PIL import ImageFilter
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
        
    def read_as_skimage(self, gray = False, specific = None):
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
        
    def read_as_cv2(self, gray = False, specific = None):
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
    
    def image(self, show = False, specific = None):
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
            self.__ia = Image.fromarray(self.__ia)
            self.__vz = Image.fromarray(self.__vz)
            self.__va = Image.fromarray(self.__va)
            if show is True:
                self.__iz.show()
                self.__ia.show()
                self.__vz.show()
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
        if four is True:
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
        """ 
        Sharpens the edges of the image by subtracting background mask
        
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
        
    def enhance_edges_with_pil(self, more = True, specific = None):
        """ 
        Enhance the edges of the image using PIL module
        
        Parameters
        ----------
        
        specific : ARRAY or STRING, optional
            DESCRIPTION, The default is None
            If None (defalut), will convert all arrays to float
            If not None, will convert specified array
        """
        if specific is None:
            if isinstance(self.__iz, str):
                self.read_as_skimage()
            self.image()
            if more is True:
                enhancement = ImageFilter.EDGE_ENHANCE_MORE
            if more is False:
                enhancement = ImageFilter.EDGE_ENHANCE
            self.__iz = self.__iz.filter(enhancement)
            self.__ia = self.__ia.filter(enhancement)
            self.__vz = self.__vz.filter(enhancement)
            self.__va = self.__va.filter(enhancement)
        else:
            if isinstance(specific, str):
                specific = self.read_as_skimage(specific = specific) 
            self.image()
            if more is True:
                enhancement = ImageFilter.EDGE_ENHANCE_MORE
            if more is False:
                enhancement = ImageFilter.EDGE_ENHANCE
            specific = specific.filter(enhancement)
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
        if isinstance(self.__iz, str):
                self.read_as_skimage()
                self.denoise_gaussian()
                self.frequency_domain()
                self.low_pass_filter()
                self.spatial_domain()
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
        if graph is True:
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
            If None (defalut), will denoise all images
            If not None, will denoise specified array
       
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
            If None (defalut), will denoise all images
            If not None, will denoise specified image

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
            If None (defalut), will denoise all images
            If not None, will denoise specified image

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
            If None (defalut), will denoise all images
            If not None, will denoise specified image

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
       
    def open_and_close_clean(self, specific = None):
        """
        Open and Close using morphology to clean up image

        Parameters
        ----------
        specific : STRING or ARRAY, optional
            DESCRIPTION. The default is None.
            If None (defalut), will denoise all images
            If not None, will denoise specified image

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        if specific is None:
            if isinstance(self.__iz, str):
                self.read_as_skimage()
            self.__iz = morphology.opening(self.__iz)
            self.__iz = morphology.closing(self.__iz)
            self.__ia = morphology.opening(self.__ia)
            self.__ia = morphology.closing(self.__ia)
            self.__vz = morphology.opening(self.__vz)
            self.__vz = morphology.closing(self.__vz)
            self.__va = morphology.opening(self.__va)
            self.__va = morphology.closing(self.__va)
            return self.__iz, self.__ia, self.__vz, self.__va    
        else:
            if isinstance(specific, str):
                self.read_as_skimage(specific = specific)
            specific = morphology.opening(specific)
            specific = morphology.closing(specific)
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
            If 'chan_vese', edge detection using Chan-Vese segmentation
        specific : STRING or ARRAY, optional
            DESCRIPTION. The default is None.
            If None (defalut), will denoise all images
            If not None, will denoise specified image
        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        edges_str = ['roberts', 'sobel', 'scharr', 'prewitt', 'chan_vese']
        edges = [roberts, sobel, scharr, prewitt, seg.chan_vese]
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
            If None (defalut), will denoise all images
            If not None, will denoise specified image
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
        
    def flood_fill(self, auto = True, seed_point = (), tolerance = 0.3, specific = None):
        """
        Flood Fill function

        Parameters
        ----------
        auto : STRING, optional
            DESCRIPTION. The default is True.
            If True (default), seed point and tolerance are automatic
            If False, seed point and tolerance must be entered
        seed_point : LIST, required is auto = False
            DESCRIPTION. The default is ().
            The point in the image from which the flood is generated
        tolerance : ARRAY or LIST, required if auto = False
            DESCRIPTION. The default is 0.3.
            The tolerance of the flood fill function
        specific : STRING or ARRAY, optional
            DESCRIPTION. The default is None.
            If None (defalut), will denoise all images
            If not None, will denoise specified image

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        if specific is None:
            if isinstance(self.__iz, str):
                self.read_as_skimage(gray = True)
            if isinstance(self.__iz[0,0], np.ubyte ):
                self.convert_to(conversion = 'to_float')
            if auto is True:
                seed_point = ((128,128), (128,128), (128,128), (128,128))
                tolerance = (0.3, 0.3, 0.3, 0.3)
            self.__iz = seg.flood(self.__iz, seed_point = seed_point[0], tolerance = tolerance[0])
            self.__ia = seg.flood(self.__ia, seed_point = seed_point[1], tolerance = tolerance[1])
            self.__vz = seg.flood(self.__vz, seed_point = seed_point[2], tolerance = tolerance[2])
            self.__va = seg.flood(self.__va, seed_point = seed_point[3], tolerance = tolerance[3])
            return self.__iz, self.__ia, self.__vz, self.__va
        else:
            if isinstance(specific, str):
                self.read_as_skimage(gray = True, specific = specific)
            if isinstance(specific, np.ubyte):
                self.convert_to(sonversion = 'to_float', specific = specific)
            if auto is True:
                seed_point = (1, 1)
                tolerance = 0.3
            specific = seg.flood(specific, seed_point = seed_point, tolerance = tolerance)  
            return specific
      
    def frequency_domain(self, specific = None):
        """
        Converts the image to the frequency domain
        
        Parameters
        ----------
        specific : STRING or ARRAY, optional
            DESCRIPTION. The default is None.
            If None (defalut), will denoise all images
            If not None, will denoise specified image

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        def f_d(i):
            dft = cv2.dft(np.float32(i), flags=cv2.DFT_COMPLEX_OUTPUT)
            dft_shift = np.fft.fftshift(dft)
            return dft_shift
        if specific is None:
            if isinstance(self.__iz, str):
                self.read_as_skimage(gray = True) 
            self.__iz = f_d(self.__iz)
            self.__ia = f_d(self.__ia)
            self.__vz = f_d(self.__vz)
            self.__va = f_d(self.__va)
            return self.__iz, self.__ia, self.__vz, self.__va
        else: 
            if isinstance(specific, str):
                self.read_as_skimage(gray = True, specific = specific)
            specific = f_d(specific)
            return specific
    
    def image_frequency_domain(self, magnitude = 20, specific = None):
        """
        Generates the images of the Z height and EFM in the frequency domain
        
        Parameters
        ----------
        magnitude : INTEGER or FLOAT, optional
            DESCRIPTION. The default is 20
            Can enter chosen value for the magnitude spectrum 
        specific : STRING or ARRAY, optional
            DESCRIPTION. The default is None.
            If None (defalut), will denoise all images
            If not None, will denoise specified image

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        def m_s(i):
            magnitude_spectrum = magnitude * np.log((cv2.magnitude(i[:, :, 0], i[:, :, 1]))+1)
            return magnitude_spectrum
        if specific is None:
            if isinstance(self.__iz, str):
                self.read_as_skimage(gray = True)
            if self.__iz.ndim == 2:
                flat = [x for sublist in self.__iz for x in sublist]
                if any(x<0 for x in flat) ==False:
                    self.frequency_domain()
            if self.__iz.ndim == 3:
                flat = [x for sublist in self.__iz for x in sublist]
                more_flat = [x for sublist in flat for x in sublist]
                if any(x<0 for x in more_flat) ==False:
                    self.frequency_domain()
            self.__iz = m_s(self.__iz)
            self.__ia = m_s(self.__ia)
            self.__vz = m_s(self.__vz)
            self.__va = m_s(self.__va)
            return self.__iz, self.__ia, self.__vz, self.__va
        else:
            if isinstance(specific, str):
                self.read_as_skimage(gray = True, specific = specific)
            if specific.ndim == 2:
                flat = [x for sublist in specific for x in sublist]
                if any(x<0 for x in flat) == False:
                    self.frequency_domain(specific = specific)
            if specific.ndim == 3:
                flat = [x for sublist in specific for x in sublist]
                more_flat = [x for sublist in flat for x in sublist]
                if any(x<0 for x in more_flat) == False:
                    self.frequency_domain(specific = specific)
            specific = m_s(specific)
            return specific
            
    def high_pass_filter(self, auto = True, r = [], specific = None):
        """
        Passes the images through a high pass filter

        Parameters
        ----------
        auto : BOOLEAN, optional
            DESCRIPTION. The default is True
            If True (default), automatic values of r are taken
            If False, values fo r need to be manually entered
        r : LIST or ARRAY, required is auto = False
            DESCRIPTION. The default is []
            Used as the radius of the circle to remove frequencies from within 
        specific : STRING or ARRAY, optional
            DESCRIPTION. The default is None.
            If None (defalut), will denoise all images
            If not None, will denoise specified image

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
        if specific is None:
            if isinstance(self.__iz, str):
                self.read_as_skimage(gray = True)
            if self.__iz.ndim == 2:
                flat = [x for sublist in self.__iz for x in sublist]
                if any(x<0 for x in flat) ==False:
                    self.frequency_domain()
            if self.__iz.ndim == 3:
                flat = [x for sublist in self.__iz for x in sublist]
                more_flat = [x for sublist in flat for x in sublist]
                if any(x<0 for x in more_flat) ==False:
                    self.frequency_domain()   
            if auto is True:
                r = [1,1,1,1]
            mask_iz = h_p_f(self.read_as_skimage(specific = self._iz, gray = True), r=r[0]) 
            self.__iz = mask_iz *self.__iz
            mask_ia = h_p_f(self.read_as_skimage(specific = self._ia, gray = True), r=r[1])
            self.__ia = mask_ia*self.__ia
            mask_vz = h_p_f(self.read_as_skimage(specific = self._vz, gray = True), r=r[2])
            self.__vz = mask_vz*self.__vz
            mask_va = h_p_f(self.read_as_skimage(specific = self._va, gray = True), r=r[3])
            self.__va = mask_va*self.__va
            return self.__iz, self.__ia, self.__vz, self.__va
        else:
            if isinstance(specific, str):
                self.read_as_skimage(gray = True, specific = specific)
            original = specific
            if specific.ndim ==2:
                flat = [x for sublist in specific for x in sublist]
                if any(x<0 for x in flat):
                    self.frequecy_domain(specific=specific)
            if specific.ndim ==3:
                flat = [x for sublist in specific for x in sublist]
                more_flat = [x for sublist in flat for x in sublist]
                if any(x<0 for x in more_flat):
                    self.frequency_domain(specific=specific)
            if auto is True:
                r = [1]
            mask = h_p_f(original, r = r[0])
            specific = mask *specific
            return specific 
        
    def low_pass_filter(self, auto = True, r = [], specific = None):
        """
        Passes the images through a high pass filter

        Parameters
        ----------
        auto : BOOLEAN, optional
            DESCRIPTION. The default is True
            If True (default), automatic values of r are taken
            If False, values fo r need to be manually entered
        r : LIST or ARRAY, required is auto = False
            DESCRIPTION. The default is []
            Used as the radius of the circle to remove frequencies from outside of 
        specific : STRING or ARRAY, optional
            DESCRIPTION. The default is None.
            If None (defalut), will denoise all images
            If not None, will denoise specified image
        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        def l_p_f(oi, r):
            rows, cols = oi.shape
            crow, ccol = int(rows / 2), int(cols / 2)
            mask = np.zeros((rows, cols, 2), np.uint8)
            center = [crow, ccol]
            x, y = np.ogrid[:rows, :cols]
            mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r*r
            mask[mask_area] = 1
            return mask
        if specific is None:
            if isinstance(self.__iz, str):
                self.read_as_skimage(gray = True)
            if self.__iz.ndim == 2:
                flat = [x for sublist in self.__iz for x in sublist]
                if any(x<0 for x in flat) ==False:
                    self.frequency_domain()
            if self.__iz.ndim == 3:
                flat = [x for sublist in self.__iz for x in sublist]
                more_flat = [x for sublist in flat for x in sublist]
                if any(x<0 for x in more_flat) ==False:
                    self.frequency_domain()   
            if auto is True:
                r = [100,100,100,100]
            mask_iz = l_p_f(self.read_as_cv2(specific = self._iz, gray = True), r=r[0]) 
            self.__iz = mask_iz *self.__iz
            mask_ia = l_p_f(self.read_as_skimage(specific = self._ia, gray = True), r=r[1])
            self.__ia = mask_ia*self.__ia
            mask_vz = l_p_f(self.read_as_skimage(specific = self._vz, gray = True), r=r[2])
            self.__vz = mask_vz*self.__vz
            mask_va = l_p_f(self.read_as_skimage(specific = self._va, gray = True), r=r[3])
            self.__va = mask_va*self.__va
            return self.__iz, self.__ia, self.__vz, self.__va
        else:
            if isinstance(specific, str):
                self.read_as_skimage(gray = True, specific = specific)
            original = specific
            if specific.ndim ==2:
                flat = [x for sublist in specific for x in sublist]
                if any(x<0 for x in flat):
                    self.frequecy_domain(specific=specific)
            if specific.ndim ==3:
                flat = [x for sublist in specific for x in sublist]
                more_flat = [x for sublist in flat for x in sublist]
                if any(x<0 for x in more_flat):
                    self.frequency_domain(specific=specific)
            if auto is True:
                r = [100]
            mask = l_p_f(original, r = r[0])
            specific = mask *specific
            return specific 
    
    def band_pass_filter(self, auto = True, r_in = [], r_out = [], specific = None):
        """
        Passes the images through a high pass filter

        Parameters
        ----------
        auto : BOOLEAN, optional
            DESCRIPTION. The default is True
            If True (default), automatic values of r are taken
            If False, values fo r need to be manually entered
        r_in : LIST or ARRAY, required is auto = False
            DESCRIPTION. The default is []
            Used as the radius of the circle to remove frequencies from within
        r_out : LIST or ARRAY, required is auto = False
            DESCRIPTION. The default is []
            Used as the radius of the circle to remove frequencies from outside of
        specific : STRING or ARRAY, optional
            DESCRIPTION. The default is None.
            If None (defalut), will denoise all images
            If not None, will denoise specified image
            
        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        def b_p_f(oi, r_in, r_out):
            rows, cols = oi.shape
            crow, ccol = int(rows / 2), int(cols / 2)
            mask = np.zeros((rows, cols, 2), np.uint8)
            center = [crow, ccol]
            x, y = np.ogrid[:rows, :cols]
            mask_area = np.logical_and(((x - center[0]) ** 2 + (y - center[1]) ** 2 >= r_in ** 2),
                           ((x - center[0]) ** 2 + (y - center[1]) ** 2 <= r_out ** 2))
            mask[mask_area] = 1
            return mask
        if specific is None:
            if isinstance(self.__iz, str):
                self.read_as_skimage(gray = True)
            if self.__iz.ndim == 2:
                flat = [x for sublist in self.__iz for x in sublist]
                if any(x<0 for x in flat) ==False:
                    self.frequency_domain()
            if self.__iz.ndim == 3:
                flat = [x for sublist in self.__iz for x in sublist]
                more_flat = [x for sublist in flat for x in sublist]
                if any(x<0 for x in more_flat) ==False:
                    self.frequency_domain()   
            if auto is True:
                r_out = [80,80,80,80]
                r_in = [5,5,5,5]
            mask_iz = b_p_f(self.read_as_skimage(specific = self._iz, gray = True), r_in=r_in[0], r_out=r_out[0]) 
            self.__iz = mask_iz *self.__iz
            mask_ia = b_p_f(self.read_as_skimage(specific = self._ia, gray = True), r_in=r_in[1], r_out=r_out[1])
            self.__ia = mask_ia*self.__ia
            mask_vz = b_p_f(self.read_as_skimage(specific = self._vz, gray = True), r_in=r_in[2], r_out=r_out[2])
            self.__vz = mask_vz*self.__vz
            mask_va = b_p_f(self.read_as_skimage(specific = self._va, gray = True), r_in=r_in[3], r_out=r_out[3])
            self.__va = mask_va*self.__va
            return self.__iz, self.__ia, self.__vz, self.__va
        else:
            if isinstance(specific, str):
                self.read_as_skimage(gray = True, specific = specific)
            original = specific
            if specific.ndim ==2:
                flat = [x for sublist in specific for x in sublist]
                if any(x<0 for x in flat):
                    self.frequecy_domain(specific=specific)
            if specific.ndim ==3:
                flat = [x for sublist in specific for x in sublist]
                more_flat = [x for sublist in flat for x in sublist]
                if any(x<0 for x in more_flat):
                    self.frequency_domain(specific=specific)
            if auto is True:
                r_out = [80]
                r_in = [5]
            mask = b_p_f(original, r_in=r_in[0], r_out=r_out[0])
            specific = mask *specific
            return specific 
        
    def spatial_domain(self, specific = None):
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
        if specific is None:
            self.__iz = s_d(self.__iz)
            self.__ia = s_d(self.__ia)
            self.__vz = s_d(self.__vz)
            self.__va = s_d(self.__va)
            return self.__iz, self.__ia, self.__vz, self.__va
        else:
            specific = s_d(specific)
            return specific
    
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
    
    def sections(self, no_of_sections = 3, split = 'Segment', show = True, cmap = 'magma', specific = None):
        """
        Splits the image into sections and returns an array of values required

        Parameters
        ----------
        image : TYPE
            DESCRIPTION.
        no_of_sections : TYPE
            DESCRIPTION.
        split : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        split_str = ['Segment', 'Empty_Low', 'Empty_High', 'Full_Low','Full_High']
        index = split_str.index(split)
        intervals = np.linspace(0, 1, no_of_sections+1)
        def segment(image, no_of_sections):
            arr = []
            for i in range(0,no_of_sections):
                mask = np.ma.masked_outside(image, intervals[i], intervals[1+i])
                arr.append(mask)
            return arr
        def empty_low(image, no_of_sections):
            arr = []
            for i in range(0,no_of_sections):
                mask = np.ma.masked_greater(image, intervals[i])
                arr.append(mask)
            return arr
        def full_low(image, no_of_sections):
            arr = []
            for i in range(0,no_of_sections):
                mask = np.ma.masked_less_equal(image, intervals[i])
                arr.append(mask)
            return arr
        if index == 2 or index == 4:
            intervals = np.flip(intervals)
        def empty_high(image, no_of_sections):
            arr = []
            for i in range(0,no_of_sections):
                mask = np.ma.masked_less_equal(image, intervals[i])
                arr.append(mask)
            return arr
        def full_high(image, no_of_sections):
            arr = []
            for i in range(0,no_of_sections):
                mask = np.ma.masked_greater(image, intervals[i])
                arr.append(mask)
            return arr
        split = [segment, empty_low, empty_high, full_low, full_high]
        if specific is None:
            if isinstance(self.__iz, str):
                self.read_as_skimage()
            if isinstance(self.__iz[0][0], np.uint8):
                self.convert_to()
            self.__iz = split[index](self.__iz, no_of_sections)
            self.__ia = split[index](self.__ia, no_of_sections)
            self.__vz = split[index](self.__vz, no_of_sections)
            self.__va = split[index](self.__va, no_of_sections)
            if show is True:
                self.image_masked_arrays(cmap = cmap)
            return self.__iz, self.__ia, self.__vz, self.__va
        else:
            if isinstance(specific, str):
                self.read_as_skimage(specific = specific)
            if isinstance(specific, np.float64):
                self.convert_to(specific = specific)
            specific = split[index](specific, no_of_sections)
            return specific 
    
    def image_masked_arrays(self, cmap = 'magma', specific = None):
        if specific is None:
            for i in range(0, len(self.__iz)):
                fig, ax = plt.subplots()
                ax.imshow(self.__iz[i], cmap = cmap)
            for i in range(0, len(self.__ia)):
                fig, ax = plt.subplots()
                ax.imshow(self.__ia[i], cmap = cmap)
            for i in range(0, len(self.__vz)):
                fig, ax = plt.subplots()
                ax.imshow(self.__vz[i], cmap = cmap)
            for i in range(0, len(self.__va)):
                fig, ax = plt.subplots()
                ax.imshow(self.__va[i], cmap = cmap)
        else:
            for i in range(0, len(specific)):
                fig, ax = plt.subplots()
                ax.imshow(specific[i], cmap = cmap) 
        return None

        
    def uneven_sections(self, dividers, specific = None):
        """

        Parameters
        ----------
        dividers : TYPE
            DESCRIPTION.
        specific : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
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
        if specific is None:
            if len(dividers) == 4:
                div1, div2, div3, div4 = dividers
            if len(dividers) == 2:
                div1, div2 = dividers
                div3, div4 = dividers
            if len(dividers) == 1:
                div1 = dividers
                div2 = dividers
                div3 = dividers
                div4 = dividers
            if isinstance(self.__iz, str):
                self.read_as_skimage()
            if isinstance(self.__iz[0][0], np.uint8):
                self.convert_to()
            self.__iz = uneven_through(self.__iz, div1)
            self.__ia = uneven_through(self.__ia, div2)
            self.__vz = uneven_through(self.__vz, div3)
            self.__va = uneven_through(self.__va, div4)
            return self.__iz, self.__ia, self.__vz, self.__va
        else:
            if isinstance(specific, str):
                self.read_as_skimage(specific = specific)
            if isinstance(specific[0][0], np.uint8):
                self.convert_to(specific = specific)
            specific = uneven_through(specific, dividers)
            return specific
        
    def central_moments(self, p, specific = None) :
        """
        """
        def central_moments_masked(masked_im, p):
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
        if specific is None:
            if isinstance(self.__iz, str):
                self.read_as_skimage()
            if isinstance(self.__iz[0][0], np.uint8):
                self.convert_to()
            if isinstance(self.__iz, np.ma.core.MaskedArray) is True:
                roughness_iz = central_moments_masked(masked_im = self.__iz, p=p)
                roughness_ia = central_moments_masked(masked_im = self.__ia, p=p)
                roughness_vz = central_moments_masked(masked_im = self.__vz, p=p)
                roughness_va = central_moments_masked(masked_im = self.__va, p=p)
            if isinstance(self.__iz, np.ndarray):
                roughness_iz = central_moments_unmasked(unmasked_im = self.__iz, p=p)
                roughness_ia = central_moments_unmasked(unmasked_im = self.__ia, p=p)
                roughness_vz = central_moments_unmasked(unmasked_im = self.__vz, p=p)
                roughness_va = central_moments_unmasked(unmasked_im = self.__va, p=p)
            roughness =  [roughness_iz, roughness_ia, roughness_vz, roughness_va]
        else:
            if isinstance(specific, str):
                self.read_as_skimage(specific = specific)
            if isinstance(specific[0][0], np.uint8):
                self.convert_to(specific = specific)
            if isinstance(specific, np.ma.core.MaskedArray):
                roughness = central_moments_masked(masked_im = specific, p=p)
                roughness = [roughness]
            if isinstance(specific, np.ndarray):
                roughness = central_moments_unmasked(unmasked_im = specific, p=p)
                roughness = [roughness]
        return roughness
        
    def average_rms_roughness(self, params, optimize = False, 
                              bnds = ((0.01, 1), (0.01, 1)), x0 = [0.33, 0.66], 
                              args = [image], specific = None):
        def average(im, params):
            through = self.uneven_sections(specific = im, dividers = params)
            total_mu = 0
            for i in range(0, len(through)):
                mu = self.central_moments(p=2, specific =through[i])
                total_mu = total_mu + np.power(mu, 2)
            average_mu = np.power(total_mu/len(through), 0.5)
            return average_mu
        if optimize is False:
            if specific is None:
                if isinstance(self.__iz, str):
                    self.read_as_skimage()
                if isinstance(self.__iz[0][0], np.uint8):
                    self.convert_to()
                roughness_iz = average(im = self.__iz, params=params[0])
                roughness_ia = average(im = self.__ia, params=params[1])
                roughness_vz = average(im = self.__vz, params=params[2])
                roughness_va = average(im = self.__va, params=params[3])
                roughness = [ roughness_iz, roughness_ia, roughness_vz, roughness_va]
            else:
                if isinstance(specific, str):
                    self.read_as_skimage(specific = specific)
                if isinstance(specific[0][0], np.uint8):
                    self.convert_to(specific = specific)
                roughness = average(im = specific, params = params)
                roughness = [roughness]
        return roughness
            
                    
                
                
            
            
#image = img_as_float(skimage.io.imread(fname = iz, as_gray = True))
#bnds = ((0.01,1),(0.01,1))
#x0 = [0.33, 0.66]
#mi = sp.optimize.dual_annealing(average_rms_roughness, bounds=bnds, args=[image], maxfun =500, x0=x0)
#print(mi)

        
    
        
    
        
  
    