# -*- coding: utf-8 -*-
#%%
# =============================================================================
#                                Loading data
# =============================================================================
import pandas as pd
import numpy as np
import numba as nb
import time
import matplotlib.pyplot as plt
import warnings

inhabitants = pd.read_csv('Inhabitants.txt',header=None, delimiter = ';').to_numpy()
AHN_max = pd.read_csv('AHN_max.txt', header = None, delimiter = ';').to_numpy()
AHN_gem = pd.read_csv('AHN_avg.txt', header = None, delimiter = ';').to_numpy()
#script_to_run = "My_packages.py"
#exec(open(script_to_run).read())

# Create a grid representing the landscape with separate attributes for land height and water height
landscape_shape = AHN_gem.shape
land_height = np.copy(AHN_max)  # Initialize with land heights
land_height[np.isnan(land_height)] = -999
# =============================================================================
#                            Casualties function
# =============================================================================
# This function will calculate the casualities per tile
@nb.jit(fastmath = True)
def calc_casualties_tile(w,h,f_e,population):
    # w = Rise rate of water(m\hour)
    # v = flow rate of the water (m/s) (will be ignored for simplicity)
    # h = height of the water (m)
    # f_e = The evacuation factor
    # population = population of the area (people)
    # f = casuality factor
    # casualties = casualties per tile
    f = 0
    # Determine the casualy factor:
    
    #if (h*v >= 7) and v >= 2:
    #    print("h*v >= 7 and v >= 2")
    #    f = 1
        
    if (w >= 0.5) and (1.5 <= h) and (h <= 4.7):
        #print("w >= 0.5 and 1.5 <= h <= 4.7")
        f = 1.45*(10**(-3)*np.exp(0.39*h))
    
    elif (w >= 0.5) and (h > 4.7):
        #print("w >= 0.5 and h > 4.7")
        f = 1

    elif ((w< 0.5) and h >0) or ((w > 0.5) and (h < 1.5)):
        #print("((w< 0.5) and h >0) or ((w > 0.5) and (h < 1.5))")
        f = 1.34*(10**(-3)*np.exp(0.39*h))
    if f > 1:
        f = 1
    
    # Calculate the casualties per tile
    casualties = (1-f_e)*f*population
    casualties = round(casualties)
    return casualties

        


def test_calc_casualties():
    # Test case 1
    w = 0.5
    v = 2
    h = 2
    f_e = 0.5
    population = 1000
    print("inputs: w = 0.5, v = 2, h = 2, f_e = 0.5, population = 1000")
    casualties =calc_casualties_tile(w,  h, f_e, population)
    print("outputs:", casualties)
    
    # Test case 2
    w = 0.2
    v = 1
    h = 1
    f_e = 0.2
    population = 500
    print("inputs: w = 0.2, v = 1, h = 1, f_e = 0.2, population = 500")
    casualties =calc_casualties_tile(w,  h, f_e, population)
    print("outputs:", casualties)
    
    # Test case 3
    w = 1
    v = 3
    h = 5
    f_e = 0.8
    population = 2000
    print("inputs: w = 1, v = 3, h = 5, f_e = 0.8, population = 2000")
    casualties =calc_casualties_tile(w,  h, f_e, population)
    print("outputs:", casualties)
    
    # Test case 4
    w = 0.5
    v = 2
    h = 6
    f_e = 0.5
    population = 10000
    print("inputs: w = 0.5, v = 2, h = 6, f_e = 0.5, population = 10000")
    casualties =calc_casualties_tile(w,  h, f_e, population)
    print("outputs:", casualties)
#test_calc_casualties()


    

# =============================================================================
#                        Highest casualties over time calculation
# =============================================================================
@nb.jit(fastmath = True)
def max_death(water_height_figures,w,i,j,f_e):
    # this function will calculate the casualties for every time step and take the highest value
   # f_e = 0.98
    casualties = 0
    # Simulate the total casualties
    #Loop over every tile in water_height_figures
    for t in nb.prange(len(water_height_figures)):
        #Skip a cell information if it is nan or if water level is 0
        if  water_height_figures[t][i,j] == 0 and w[t][i,j] == 0:
            continue
        # Calculate the casualties per tile
        casualties_tile = calc_casualties_tile(w[t][i,j],water_height_figures[t][i,j],f_e,inhabitants[i,j])
        #if casualties_tile > 1000:
        #    print("inhabitants:",inhabitants[i,j])
        #    print(" rise rate:",w[t][i,j],"water_height:",water_height_figures[t][i,j],"casualties:",casualties_tile,"time:",t,"tile:",i,j)
        # Check if the casualties are higher than the previous ones
        if casualties_tile > casualties:
            casualties = casualties_tile
            
    return casualties


#==============================================================================
#                            Casualties simulation
# =============================================================================
# this function will calculate the total amount of casualties


@nb.jit(fastmath = True,nogil = True)
def simulation(water_height_figures,w,f_e):
    casualty_map = np.zeros(landscape_shape)
    casualties = 0
    # Simulate the total casualties
    #Loop over every tile in water_height_figures
    for i in nb.prange(landscape_shape[0]):
        for j in nb.prange(landscape_shape[1]):
            #print("tile:",i,j)
            #Skip a cell information if it is nan or if water level is 0
            if land_height[i,j] == -999:
                
                continue
            # Calculate the casualties per tile
            casualties_tile = max_death(water_height_figures,w,i,j,f_e)
            casualty_map[i,j] = casualties_tile
            #if casualties_tile > 0:
                #print("casualties_tile:",i,j,casualties_tile)
            # Add the casualties to the total casualties
            casualties += casualties_tile
    #print("Evacuation factor:",f_e,"Percentage of people escaping the flood")
    return casualties,casualty_map
# Ignore all Numba warnings

# This function will run the simulation for different evacuation factors
def run_casualties(water_height_figures,w,low_fe,high_fe):
    warnings.filterwarnings("ignore", category=nb.NumbaWarning)
    start_time = time.time()
    print(" ")
    print("population:",np.nansum(inhabitants))
    
    casualties_low_fe,casaulty_map = simulation(water_height_figures,w,low_fe)
    casualties_high_fe,casaulty_map = simulation(water_height_figures,w,high_fe)
    
    print("Evacuation factor:",low_fe,"Total casualties are:", casualties_low_fe)
    print("Percentage population died:", 100*casualties_low_fe/np.nansum(inhabitants),"%")
    
    print("Evacuation factor:",high_fe,"Total casualties are:", casualties_high_fe)
    print("Percentage population died:", 100*casualties_high_fe/np.nansum(inhabitants),"%")
    
    end_time = time.time()
    print("Casualty estimation simulation time taken:",end_time-start_time)
    print(" ")
    warnings.resetwarnings()
    return casualties_low_fe,casualties_high_fe,casaulty_map