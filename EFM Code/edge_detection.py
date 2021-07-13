# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 14:51:59 2021

@author: sophi
"""

#%%
import main 
p3ht = main.molecule(iz = 'P3HT 58k 11 5um 0V_190208_Z Height_Forward_001.tiff', ia = 'P3HT 58k 11 5um 0V_190208_EFM Amplitude_Forward_001.tiff', vz = 'P3HT 58k 11 5um EFM 2V_190208_Z Height_Forward_003.tiff', va = 'P3HT 58k 11 5um EFM 2V_190208_EFM Amplitude_Forward_003.tiff')
#p3ht.read()
#p3ht.bearray()
iz = p3ht.iz()
ia = p3ht.ia()
vz = p3ht.vz()
va = p3ht.va()

#%%Edge enhancement using Pillow
from PIL import Image
from PIL import ImageFilter
import skimage
import skimage.feature
import skimage.viewer
imageObject = Image.open(va)
imageObject = imageObject.convert('RGB') # need to convert toe RGB file for enhancement but I think this might make it harder to find edges so defeats the point of the enhancement
# Apply edge enhancement filter
edgeEnhanced = imageObject.filter(ImageFilter.EDGE_ENHANCE)
# Apply increased edge enhancement filter
moreEdgeEnhanced = imageObject.filter(ImageFilter.EDGE_ENHANCE_MORE)
# Show original image - before applying edge enhancement filters
imageObject.show() 
# Show image - after applying edge enhancement filter
edgeEnhanced.show()
# Show image - after applying increased edge enhancement filter
moreEdgeEnhanced.show()
moreEdgeEnhanced.save('Edge_enhanced_rgb.tiff')

#%% Issue is though that I'm not sure its sensitive enough for less obvious data  
#plus only opens one at a time 

"""
from https://datacarpentry.org/image-processing/08-edge-detection/
 * Python script to demonstrate Canny edge detection
 * with sliders to adjust the thresholds.
 *
 * usage: python CannyTrack.py <'P3HT 58k 11 5um 0V_190208_Z Height_Forward_001.tiff'>
"""
import skimage
import skimage.feature
import skimage.viewer


filename = 'Edge_enhanced_rgb.tiff'
image = skimage.io.imread(fname=filename, as_gray=True)
viewer = skimage.viewer.ImageViewer(image)

# Create the plugin and give it a name
canny_plugin = skimage.viewer.plugins.Plugin(image_filter=skimage.feature.canny)
canny_plugin.name = "Canny Filter Plugin"

# Add sliders for the parameters
canny_plugin += skimage.viewer.widgets.Slider(
    name="sigma", low=0.0, high=7.0, value=2.0
)
canny_plugin += skimage.viewer.widgets.Slider(
    name="low_threshold", low=0.0, high=1.0, value=0.1
)
canny_plugin += skimage.viewer.widgets.Slider(
    name="high_threshold", low=0.0, high=1.0, value=0.2
)

# add the plugin to the viewer and show the window
viewer += canny_plugin
viewer.show()
