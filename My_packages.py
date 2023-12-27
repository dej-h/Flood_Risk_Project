# -*- coding: utf-8 -*-
#%%

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import numba as nb
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.animation import FuncAnimation
import concurrent.futures

dikeringarea = pd.read_csv('Dikeringarea.txt',header=None, delimiter = ';').to_numpy()
inhabitants = pd.read_csv('Inhabitants.txt',header=None, delimiter = ';').to_numpy()
landuse = pd.read_csv('landuse.txt',header=None, delimiter = ';').to_numpy()
AHN_max = pd.read_csv('AHN_max.txt', header = None, delimiter = ';').to_numpy()
AHN_gem = pd.read_csv('AHN_avg.txt', header = None, delimiter = ';').to_numpy()
# %%
