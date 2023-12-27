# -*- coding: utf-8 -*-
#%%
# =============================================================================
#                                Loading data
# =============================================================================
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import numba as nb
import time
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.animation import FuncAnimation
from matplotlib.colors import SymLogNorm
import concurrent.futures
import warnings

dikeringarea = pd.read_csv('Dikeringarea.txt',header=None, delimiter = ';').to_numpy()
inhabitants = pd.read_csv('Inhabitants.txt',header=None, delimiter = ';').to_numpy()
landuse = pd.read_csv('landuse.txt',header=None, delimiter = ';').to_numpy()
AHN_max = pd.read_csv('AHN_max.txt', header = None, delimiter = ';').to_numpy()
AHN_gem = pd.read_csv('AHN_avg.txt', header = None, delimiter = ';').to_numpy()


landuse[np.isnan(landuse)] = -999
landscape_shape = AHN_gem.shape
# =============================================================================
#                                Damage Calculation
# =============================================================================
# Damage calculations consists of 3 parts:
# 1. a: Damage Factor category i
# 2. n: Number of units in category i
# 3. S: Maximum damage per unit in category i
# For combining all of the tiles
#Example:
"""
def Damage_calculation_total(a,n,S):
    damage = sum(a)*sum(n)*sum(S) #Probably wrong
    return damage
# For single tiles:
def Damage_calculation(a,n,S):
    damage = a*n*S
    return damage
"""

# =============================================================================
#                                Damage Factors
# =============================================================================
# Damage factors are based on the damage factors from the Dutch government
"""
The damage factors shown below are applied to define the damage in low-frequency flooded areas.
The damage factors are based on the damage factors from the Dutch government (Rijkswaterstaat, 2004).
Parameters used in the functions are:
- d = flood depth (m)
- u = flow rate (m/s)
- ukr = critical flow rate (m/s)
- w = rise rate (m/hour)
- β = material factor
- r = shelter factor
- s = presence of storm (waves)
    """
    
"""
Land-use classes:
1. Infrastructure
2. Residential
3. Industry/commercial
4. Governmental institutions / services
5. Recreation
6. Greenhouses
7. Agriculture
8. Water
9. Nature
"""


# For all these functions d is the flood-depth
# damage factor for agriculture, recreation, airports, greenhouses
@nb.njit(fastmath=True)  
def damage_factor_agriculture(d):
    damage = min(d,0.24*d + 0.4, 0.07*d + 0.75, 1)
    return damage

# damage factor for vechiles
@nb.njit(fastmath=True)  
def damage_factor_vechicles(d):
    damage = min(0.17*d - 0.03, 0.72*d - 0.3, 0.31*d + 0.1, 1)
    return damage

# damage factor for infarstructure
@nb.njit(fastmath=True)  
def damage_factor_infrastructure(d):
    damage = min(0.28*d, 0.18*d +0.1, 1)
    return damage

# damage factor for industry, companies, goverment
@nb.njit(fastmath=True)  
def damage_factor_industry(d):
    if 0 <= d <= 1:
        damage = 0.1*d
    elif 1<= d <= 3:
        damage = 0.06*d + 0.04
    elif 3 <= d <= 5:
        damage = 0.38*d - 0.95
    elif 5 <= d:
        damage = 1
    return damage

# damage factor for residential
@nb.njit(fastmath=True)  
def damage_factor_residential(d):
    if 0 <= d <= 1:
        damage = 0.33*d
    elif 1<= d <= 2:
        damage = 0.05*d + 0.15
    elif 2 <= d <= 5:
        damage = 0.3*d -0.1
    elif 5 <= d:
        damage = 1
    return damage
"""
For home owners the average is 67 m2.
Tenants have more space: 92 m2. In the rest of the country,
the average area of usable space is 140 m2 for owners and 87 m2 for tenants.
Sixty-one percent (246,000) of Amsterdam households live in less than 75 m2 of usable space.

"""
@nb.njit(fastmath=True)
def max_damage_calc(category,i,j,inhabitants):
    #This is all per 100m2 or per vechile
    industry_conv = 0.3     #Conversion from per industry establishment to 100m2
    """
    category_to_max_dmg_dict = {
        #conversion is done from statistics above
        'infrastructure': 0,                                        # From DWW is per meter need to be converted to 100m2; roads/railsways are very different
        'residential': 241000*(100/((67+92)/2)),                    # From DWW is per house need to be converted to 100m2
        'industry': 350000*industry_conv,                           # From DWW is per person need to be converted to 100m2
        'government': 0.38,                                         # From DWW is per person need to be converted to 100m2
        'recreation': 1980,                                         # From the DWW
        'greenhouses': 4410,                                        # From the DWW
        'agriculture': 310,                                         # From the DWW
        'water': 0,                                                 # From the DWW, but different sort of damage
        'nature': 0,                                                # From the DWW, but different sort of damage
        'vehicles': 1070,                                           # From the DWW but is per vechile; need to research how many vechiles in flood area
        # Add more mappings here
    }
    """
    # infastructure
    if category == 1:
        #30% of the damage is to trunk roads,20% is motorways, 5% to railways, 55% to other roads
        return 100*((1450+650)*0.3+(980)*0.2+(270)*0.55+(25150+86+151)*0.05)/4
    # residential
    elif category == 2:
        return 241000*(100/((67+92)/2))
    # industry
    elif category == 3:
        return 350000*industry_conv
    # government
    elif category == 4:
        if inhabitants[i,j] > 10:
            people = inhabitants[i,j]
        else:
            people = 10
            return people/100*(60000+2200+9200)
    # recreation
    elif category == 5:
        return 1980
    # greenhouses
    elif category == 6:
        return 4410
    # agriculture
    elif category == 7:
        return 310
    # water
    elif category == 8:
        return 0
    # nature
    elif category == 9:
        return 0
    # vehicles
    elif category == 10:
        return 1070
    
    # Convert the input word to its corresponding number, if available
    #numeric_value = category_to_max_dmg_dict.get(category.lower())

    #if numeric_value is not None:
   #     return numeric_value
   # else:
        # Return None or any other suitable value for words not in the dictionary
   #     return None

# =============================================================================
#                                Damage per Tile calculation
# =============================================================================
@nb.njit(fastmath=True)
def tile_financial_dmg(water_level,land_class,i,j,inhabitants):
    damage = 0
    damage_factor = 0
    max_damage = max_damage_calc(land_class,i,j,inhabitants)
    
    #This elif list shows all the land classes that are in the model that can have financial damage
    if land_class == 1: #"infastructure":
        damage_factor = damage_factor_infrastructure(water_level)
        
    elif land_class == 2: # "residential":
        damage_factor = damage_factor_residential(water_level)
        
    elif land_class == 3: # "industry":
        damage_factor = damage_factor_industry(water_level)
        
    elif land_class == 4: # "government":
        damage_factor = damage_factor_industry(water_level)
        
    elif land_class ==5: # "recreation":
        damage_factor = damage_factor_agriculture(water_level)
        
    elif land_class == 6: # "greenhouses":
        damage_factor = damage_factor_agriculture(water_level)
        
    elif land_class == 7: # "agriculture":
        damage_factor = damage_factor_agriculture(water_level)
        
    elif land_class == 8: # "vehicles":
        damage_factor = damage_factor_vechicles(water_level)
    # Will catch anything that isn't recognized    
    else:
        damage_factor = 0
        
    damage = damage_factor*max_damage
    
    return damage
# =============================================================================
#                                Minutes to days/hours/minutes converter
# =============================================================================
@nb.njit(fastmath=True)
def min2dayhours(minutes,figures_ratio):
    minutes = minutes  * figures_ratio*10
    hours = 0
    days = 0
    # Loop runs while there are more then 60 minutes
    while minutes > 60:
        if minutes > 60:
            hours += 1
            minutes = minutes - 60
        if hours == 24:
            hours = hours - 24
            days += 1
            
    return minutes,hours,days
# =============================================================================
#                  Get the max water level over all figures per tile
# =============================================================================
@nb.njit(fastmath=True)
def max_water_level(water_height_figures,i,j):
    #print("Calculating max water level for",i,j)
    max_water_level = 0
    for step in range(len(water_height_figures)):
        max_water_level = max(max_water_level,water_height_figures[step][i,j])
    return max_water_level
   
 
# =============================================================================
#                                Full damage calculation
# =============================================================================

@nb.njit(fastmath=True,nogil=True)
def total_damage_calc(water_height_figures,landuse,inhabitants,run4all,figures_ratio,Financial_damage_map):
    #print("Calculating total damage")
    # Loop through all the water level figures
    total_damage_list = [0 for _ in range(len(water_height_figures))]
    for step in range(len(water_height_figures)):
        # Only run for the final result if run4all is false
        if run4all == False:
            if __name__ == '__main__':
                print("Only running for final value")
            step = len(water_height_figures)-1
        total_damage = 0
        #Loop through all the rows of the figure
        if __name__ == '__main__':
            print(f"Calculating total damage for step {step} in {len(water_height_figures)} steps")
        for i in nb.prange(landscape_shape[0]):
            #Loop through all of the columns of the figure
            for j in nb.prange(landscape_shape[1]):
                #skip the cell if it has nothing/water/nature
                if landuse[i,j] == -999 or landuse[i,j] == 8 or landuse[i,j] == 9:
                    continue
                #Determine landuse category then calculate its total damage
                #landuse_category = landuse_num_to_category(landuse[i,j])
                landuse_category = landuse[i,j]
                if run4all == False:
                    max_water_height = max_water_level(water_height_figures, i,j)
                else:
                    max_water_height = water_height_figures[step][i,j]
                tile_damage = tile_financial_dmg(max_water_height,landuse_category,i,j,inhabitants)
                Financial_damage_map[i,j] = tile_damage
                total_damage += tile_damage
        # Inflation calculation:
        total_damage = total_damage*1.8
        minutes,hours,days = min2dayhours(step,figures_ratio)
        
        print("Total damage Days:" ,days," Hours:", hours," Minutes:", minutes,":",total_damage,"€")

        total_damage_list[step] = total_damage
        
        # Get out of the loop if you only run the last one 
        if run4all == False:
            break
        
    return total_damage_list,Financial_damage_map
# This function will calculate the financial damage for the whole area
def run_flood_damage(water_height_figures,figures_ratio,run4all = False):
    Financial_damage_map = np.zeros(landscape_shape)  # You can set this as needed
    Financial_damage_map[np.isnan(AHN_gem)] = np.nan
    # Ignore all Numba warnings
    warnings.filterwarnings("ignore", category=nb.NumbaWarning)
    print(" ")
    print("Starting the financial Flood damage simulation:") 
    start_time = time.time()              
    lotta_damage,financial_damage_map_1 = total_damage_calc(water_height_figures,landuse,inhabitants,run4all,figures_ratio,Financial_damage_map)
    # make a matplot of the financial damage map where the color is based on the damage from green to red
    fig, ax = plt.subplots(figsize=(10,10))
    colors = [(0, 1, 0), (1, 0, 0)]  # Red to green
    colormap_landuse = LinearSegmentedColormap.from_list("my_list",colors)
    #ax.axis('scaled')
    ax.set_title('Financial damage map')
    ax.set_xlabel('X(100m)')
    ax.set_ylabel('Y(100m)')
    vmin = 0.1
    vmax = 100
    img = ax.imshow(np.flipud(financial_damage_map_1), cmap=colormap_landuse,  norm=SymLogNorm(linthresh=0.1, vmin=vmin, vmax=vmax))
    cax = fig.add_axes([0.92, 0.2, 0.02, 0.6])  # Adjust the position and size as needed
    cb = plt.colorbar(img, cax=cax)
    plt.show(img)
    
    final_damage = lotta_damage[-1]
    end_time = time.time()
    print("Done with the financial Flood damage simulation in",end_time-start_time,"seconds")
    return final_damage,img

 #%%               