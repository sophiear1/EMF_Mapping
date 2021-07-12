# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 16:58:36 2021

@author: sophi
"""

import pandas as pd
from matplotlib import pyplot as plt
import scipy as sp
import numpy as np
import glob

#practise_data_1 = pd.read_csv('P3HT 58k 11 5um 0V_190208_Z Height_Forward_001.txt', sep = '\t')
#plt.plot(practise_data)

#practise_data 
practise_data_2 = pd.read_csv('P3HT 58k 11 5um 0V_190208_Z Height_Forward_001.txt', sep = '\t')
print(type(practise_data_2))
print(practise_data_2)
print(practise_data_2.columns[2])
#practise_data_2.colums = ['x','y','z']
#plt.plot(x,y,z)