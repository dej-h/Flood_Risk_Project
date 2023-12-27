# -*- coding: utf-8 -*-
#%%
"""
Data and plot Project Flood Risk Module 5


Data:
    Dike-ring-area: 100x100m raster where cell values 16, 43 denote dike ring 
    areas 16 and 43
    Land-use: 100x100m raster, cell value denotes the dominant land use
        Source: BBG2008 (CBS) and BRP2009 (Ministery of Economic Affairs)
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
    Inhabitants: 100x100m raster, cell value denotes number of inhabitants
        Source: CBS (2013)
    AHN: Actueel Hoogtebestand Nederland: digital elevation raster, 5x5m, 
    aggregated to 100x100m cells in two manners:
        AHN_max: Cell value is the maximum elevation in the 100x100m cell
        AHN_gem: Ceel value is the average elevation in the 100x100m cell

    Note that all rasters contain NaN (not a number) values outside the area 
    of dike rings 16 and 43. 
        - In making grid based calculations, operations with NaN input will 
        always result in NaN output.
        - For operations summarizing data (sum, max, min for example), Python has 
        specific NaN versions (nansum, nanmax, ...) avoiding NaN output
"""

#Initialize the model for letting the water flow
# =============================================================================
#                                Loading data
# =============================================================================
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import numba as nb
from numba import njit, prange
from numba import cuda, jit
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider
from matplotlib.colors import SymLogNorm
import concurrent.futures
import cProfile
import time
import tracemalloc
import psutil
import warnings
from memory_profiler import profile
dikeringarea = pd.read_csv('Dikeringarea.txt',header=None, delimiter = ';').to_numpy()
inhabitants = pd.read_csv('Inhabitants.txt',header=None, delimiter = ';').to_numpy()
landuse = pd.read_csv('landuse.txt',header=None, delimiter = ';').to_numpy()
AHN_max = pd.read_csv('AHN_max.txt', header = None, delimiter = ';').to_numpy()
AHN_gem = pd.read_csv('AHN_avg.txt', header = None, delimiter = ';').to_numpy()
#script_to_run = "My_packages.py"
#exec(open(script_to_run).read())

# Create a grid representing the landscape with separate attributes for land height and water height
landscape_shape = AHN_gem.shape
land_height = np.copy(AHN_max)  # Initialize with land heights
land_height[np.isnan(land_height)] = -999


# Initialize a 2D array for water heights with zeros or desired initial values
water_height_zero = np.zeros(landscape_shape)  # You can set this as needed
water_height_zero[np.isnan(AHN_gem)] = np.nan
water_height = np.copy(water_height_zero)  # Initialize with water heights
# Define the value to replace (e.g., 16) and the replacement value (e.g., 1)
landuse_water_value = 8
water_level_value = 1

# Define a function to calculate water levels based on AHN_max
def calculate_water_level(ahn_value):
    # Calculate the water level based on the AHN_gem value
    if ahn_value <= 0:
        water_level= -ahn_value
        
    elif 2 > ahn_value > 0:
        water_level = 0
        
    elif ahn_value > 2:
        water_level = 0
    
    return water_level
# Create a boolean mask to identify where the replace_value is in landuse
mask = (landuse == landuse_water_value)

# Use the mask to calculate water levels based on AHN_max
water_height[mask] = np.vectorize(calculate_water_level)(AHN_max[mask])
# velocity in x direction: u and velocity in y direction: v

# Getting the river water height against the return period in years
lek_data = pd.read_excel('Lek_data.xlsx')
waal_data = pd.read_excel('waal_data.xlsx')
# Selecting columns C and D, excluding the first row
lek_data_subset = lek_data.iloc[1:, [1, 2]]
waal_data_subset = waal_data.iloc[1:, [1, 2]]

# Convert DataFrame to NumPy array
lek_data_array = lek_data.iloc[1:, [1, 2]].to_numpy(dtype=np.float64)
waal_data_array = waal_data.iloc[1:, [1, 2]].to_numpy(dtype=np.float64)

@nb.njit(fastmath=True, nopython=True)
def closest_discharge(river_water_height, data_subset):
    distances = np.zeros(data_subset.shape[0], dtype=np.float64)
    for i in range(distances.shape[0]):
        distances[i] = (data_subset[i, 1] - river_water_height) ** 2

    closest_index = 0
    min_distance = distances[0]
    for i in range(1, distances.shape[0]):
        if distances[i] < min_distance:
            closest_index = i
            min_distance = distances[i]

    return data_subset[closest_index, 0]
#%%

# =============================================================================
#                    Water direction and amount calculation
# =============================================================================

# This function will check if the water can flow in a certain direction and calculate the slope
# It will return if the can flow in that direction and the slope
@nb.njit(fastmath=True)
def calc_slope_and_bound(land_height,water_height,current_cell, neighbour_cell):
    num_rows, num_cols = land_height.shape
    # only calculate slope if the neighbour is within the bounds of the grid and isn't nan
    if 0 < neighbour_cell[0] < num_rows and 0 < neighbour_cell[1] < num_cols and land_height[neighbour_cell] != -999:
        # Get the elevation of the current cell
        current_elevation = land_height[current_cell]+ water_height[current_cell]
        # Get the elevation of the neighbor
        neighbor_elevation = land_height[neighbour_cell]
        neighbor_elevation += water_height[neighbour_cell]
        # Calculate the slope to the neighbor
        slope = (current_elevation - neighbor_elevation)
        # Check if the water can flow in that direction
        
        return  slope
    else:
        return 0

@nb.njit(fastmath=True)
def distribute_water(slope_north, slope_south, slope_east, slope_west,water_available):
    # Calculate the total slope
    if slope_north < 0:
        slope_north = 0
    if slope_south < 0:
        slope_south = 0
    if slope_east < 0:
        slope_east = 0
    if slope_west < 0:
        slope_west = 0
    
    
    
    total_slope = slope_north + slope_south + slope_east + slope_west
    
    if total_slope == 0:
        return 0,0,0,0
    
    # Change the factor of how much water will flow in each direction
    spread_factor = 0.5
    total_slope = total_slope * (1/spread_factor)
    
    
    # Calculate the water distribution proportion based on slopes
    water_dist_north = slope_north / total_slope
    water_dist_south = slope_south / total_slope
    water_dist_east = slope_east / total_slope
    water_dist_west = slope_west / total_slope
    
    # Assign water level changes based on distribution
    water_level_north = water_dist_north * water_available
    water_level_south = water_dist_south * water_available
    water_level_east = water_dist_east * water_available
    water_level_west = water_dist_west * water_available
    
    
    # These are the changes in water level on the initial cell and the neighboring cells
    return water_level_north, water_level_south, water_level_east, water_level_west



# This function will calculate the flow direction for multiple tiles at the same time and calculate how much water will flow in that direction
@nb.njit(fastmath=True)
def calculate_flow_direction2(land_height,water_height, current_cell):
    # define flow directions (D4 flow algorithm):
    # Define the four possible neighboring cells as (row_offset, col_offset):
    
    # if there is no water return 0,0,0,0 (flow north, flow south, flow east, flow west)
    if water_height[current_cell] == 0:
        return 0,0,0,0
    
    north_neighbour = (-1,0)
    south_neighbour = (1,0)
    east_neighbour = (0,1)
    west_neighbour = (0,-1)
    
    ## Calculate the slope to north:
    row_offset,col_offset = north_neighbour
    neighbour = (current_cell[0] + row_offset,current_cell[1] + col_offset)
    north_slope = calc_slope_and_bound(land_height,water_height,current_cell, neighbour)

    
    ## Calculate the slope to south:
    row_offset,col_offset = south_neighbour
    neighbour = (current_cell[0] + row_offset,current_cell[1] + col_offset)
    south_slope = calc_slope_and_bound(land_height,water_height,current_cell, neighbour)
    
    ## Calculate the slope to east:
    row_offset,col_offset = east_neighbour
    neighbour = (current_cell[0] + row_offset,current_cell[1] + col_offset)
    east_slope = calc_slope_and_bound(land_height,water_height,current_cell, neighbour)
    
    ## Calculate the slope to west:
    row_offset,col_offset = west_neighbour
    neighbour = (current_cell[0] + row_offset,current_cell[1] + col_offset)
    west_slope = calc_slope_and_bound(land_height,water_height,current_cell, neighbour)
    
    ### Calculate how much water will flow in teach direction:
    # flow direcitons
    water_flow_north = 0
    water_flow_south = 0
    water_flow_east = 0
    water_flow_west = 0
    
    # water available:
    water_available = water_height[current_cell]
    
    # Use distribute function:
    water_flow_north, water_flow_south, water_flow_east, water_flow_west = distribute_water(north_slope, south_slope, east_slope, west_slope,water_available)
    return water_flow_north, water_flow_south, water_flow_east, water_flow_west
    
    
# =============================================================================
#                                Minutes to days/hours/minutes converter
# =============================================================================
# this function turns minutes into days hours and minutes
def min2dayhours(minutes):
    hours = 0
    days = 0
    # Loop runs while there are more then 60 minutes
    while minutes > 60:
        if minutes > 60:
            hours += 1
            minutes = minutes - 60
        if hours == 24:
            hours = 0
            days += 1
            
    return minutes,hours,days

# =============================================================================
#                                 Calculate the discharge
# =============================================================================
# this function will calculate the dyke discharge based on the Q formula
@nb.njit(fastmath = True)
def calc_discharge_dyke_breach(step,water_height_river,step_to_min):
    
 
        # time in minutes:
        start_time = step*step_to_min
        
        # discharge_time
        end_time = step_to_min
        # gravity constant
        g = 9.81
        # Flood Plain height
        fph = 0
        total_discharge_water_height = 0
        for current_time in range(start_time,end_time,1):
            #if __name__ == "__main__":
            #    print("time in minutes:",current_time)
            # N is the amount of hours the simulation has run
            n = current_time/60 
            # calculating h(n):
            h_n = water_height_river-fph-(4.6/300)*n
            if h_n < 0:
                h_n = 0
            #if __name__ == "__main__":
                #print("h_n:",h_n)
            # Calculating breach width Based on the ESTIMATING DIKE BREACH LENGTH FROM HISTORICAL DATA

            if n < 10:
                B = n*175/10
            else:
                B = 175
            # calculating the discharge:
            discharge = ((2/3)**(1.5))*np.sqrt(g)*B*h_n**(1.5) # m3/uur
            discharge_water_height = discharge/(100*100) # m/min*step_to_min
            total_discharge_water_height += discharge_water_height
        #if __name__ == "__main__":
            #print("discharge:",discharge_water_height)
        return total_discharge_water_height
# This function will calculate the discharge of the river based on the river type and its discharge data
@nb.njit(fastmath = True)
def calc_discharge_river(step,step_to_min,river_type,river_height):
    # time in hours:
    time_hours = step*step_to_min/60
    #percentage discharge of river to the lands
    discharge_percentage = 1
    # start_time
    start_time = step*step_to_min
    end_time = step*step_to_min + step_to_min
    # flood plain height
    fph = 0
    # net river height
    net_river_height = river_height - (4.6/300)*time_hours-fph
    # if the net river height is below 0 set it to 0
    if net_river_height<0:
        net_river_height = 0
    #
    river_total_discharge = 0
    for current_time in range(start_time,end_time,1):
        # 0: for running for Waal, 1: running for lek
        if river_type == 0:
            river_discharge= closest_discharge(net_river_height, waal_data_array)
        elif river_type == 1:
            river_discharge= closest_discharge(net_river_height, lek_data_array)
        river_discharge = (river_discharge*60)/(100*100) # m/min*step_to_min
        river_total_discharge += river_discharge
    return river_total_discharge*discharge_percentage
    
@nb.njit(fastmath = True)    
def calc_volume_dyke_breach(river_height,step_to_min,step_amount):
    
    
    # steps = step_amount Riverheight= river_height step_to_min: step_to_min, from step 0 to step_amount
    discharge = 0
    for i in range(0,step_amount,1):
        discharge +=calc_discharge_dyke_breach(i,river_height,step_to_min)   
    print("Step: 1-",step_amount," river water height:",river_height," step_to_min:", step_to_min," discharge:",discharge)
#test_calc_discharge_dyke_breach()
    
# =============================================================================
#                                 Setting up the figures
# =============================================================================

    
# Plot the 2D array with the water heights
def update_water_animation(frame): 
    ax.clear()
    vmin = 0.1
    vmax = 100
    im_normed = np.copy(short_water_height_figures[frame])
    img = ax.imshow(np.flipud(im_normed), cmap=colormap_landuse,  norm=SymLogNorm(linthresh=0.1, vmin=vmin, vmax=vmax))
    ax.axis('scaled')
    ax.set_xlabel('x (100m)')
    ax.set_ylabel('y (100m)')
    minutes,hours,days = min2dayhours(frame*steps_to_reduce*step_to_min*10)
    ax.set_title(f'Water Height Level (m) - {days} days, {hours} hours, {minutes} minutes, Step_to_min used: {step_to_min} ')#Discharge: {calc_discharge_river(frame*steps_to_reduce*step_to_min*10,step_to_min,river_type1,water_height_river1)} m/{step_to_min}min')
    
    cax = fig.add_axes([0.92, 0.2, 0.02, 0.6])  # Adjust the position and size as needed
    cb = plt.colorbar(img, cax=cax)
    #cb.set_label('Water Height (mm)')  # Set the label for the colorbar
    return img
# Update the silder
def update_slider(val):
    step = int(step_slider.val)
    img = update_water_animation(step)
    plt.draw()

# this function will take the water height figures and take snap shots every x steps
def snap_water_height(water_height_figures,steps_to_reduce):
    reduced_figures_amount = len(water_height_figures)//steps_to_reduce
    short_water_height_figures = [np.zeros((land_height.shape)) for _ in range(reduced_figures_amount)]
    for i in range(0, len(water_height_figures), steps_to_reduce):
        short_water_height_figures[i//steps_to_reduce] = water_height_figures[i]
        #print("snap step:",i)
    return short_water_height_figures

    
# This function will average the water height over a certain amount of steps per tile for the simplification of the model
@nb.njit(fastmath=True)
def average_water_height(water_height_figures, steps_to_average,land_height,water_height):
    # initiate the array to store the averaged figures
    averaged_figures = [water_height for _ in range(len(water_height_figures)//steps_to_average)]
    # loop over the figures in steps of the amount of steps to average
    for step in range(0, len(water_height_figures), steps_to_average):
        short_step = step//steps_to_average
        # loop over the tiles
        for i in range(landscape_shape[0]):
                for j in range(landscape_shape[1]):
                    # skip if it is nan
                    if land_height[i,j] == -999:
                        continue
                    average_height = 0
                    # loop over the iterations in the timestep to average
                    d = 0
                    for d in range(steps_to_average):
                        if step + d < len(water_height_figures):
                            average_height += water_height_figures[step + d][i,j]
                        else:
                            print("pain")
                    # divide by the amount of steps to average
                    average_height = average_height / min(steps_to_average, len(water_height_figures) - step)
                    #add the average height to the averaged figures
                    
                    averaged_figures[short_step][i,j] += average_height

        
        print("average step:",step)
        print("water height:",averaged_figures[short_step][150,410])


    return averaged_figures


        
# =============================================================================
#                                 Simulation
# =============================================================================

# This function is simulair to the simulation function but
# It will calculate the flow direction for multiple tiles at the same time and won't go diagonally
#@profile
@nb.njit(fastmath=True,  nogil=True)
def simulation2(num_steps, water_height, water_height_figures,w,water_height_river,step_to_min,river_type):
    # constants:
    
    # Simulate water flow for a certain number of time steps
    for step in range(num_steps):
        water_height_copy = np.copy(water_height)
        temp_w_copy = np.copy(w[len(w)-1])
        for i in nb.prange(landscape_shape[0]):
            for j in nb.prange(landscape_shape[1]):
                    
                    #Skip a cell information if it is nan or if water level is 0
                    if land_height[i,j] == -999 or water_height[i,j] == 0:
                        continue
                    
                    # Calculate flow in all directions for the current cell
                    north_flow, south_flow, east_flow, west_flow = calculate_flow_direction2(land_height, water_height, (i, j))
                    # Substract the outflowing water:
                    water_height_copy[i,j] = water_height_copy[i,j] - north_flow - south_flow - east_flow - west_flow
                    # Add the outflowing water to the neighbouring cells:
                    water_height_copy[i-1,j] = water_height_copy[i-1,j] + north_flow
                    water_height_copy[i+1,j] = water_height_copy[i+1,j] + south_flow
                    water_height_copy[i,j+1] = water_height_copy[i,j+1] + east_flow
                    water_height_copy[i,j-1] = water_height_copy[i,j-1] + west_flow
                    # update the rise rates m/per hour
                    temp_w_copy[i-1,j] += (north_flow/step_to_min)*60
                    temp_w_copy[i+1,j] += (south_flow/step_to_min)*60
                    temp_w_copy[i,j+1] += (east_flow/step_to_min)*60
                    temp_w_copy[i,j-1] += (west_flow/step_to_min)*60
                    
                    
        water_height = np.copy(water_height_copy)
        temp_w = np.copy(temp_w_copy)
        # Add the discharge from the dyke breaches          
        if step < num_steps:
            # 0: for running for Waal, 1: running for lek
            if river_type == 1:   
                water_height[48,410] += calc_discharge_river(step,step_to_min,river_type,water_height_river)
            elif river_type == 0:
                water_height[194,210] += calc_discharge_river(step,step_to_min,river_type,water_height_river)
        # Every 10 steps save a value to water_height_figures
        if step % 10 == 0:
            scaled_step = step//10
            water_height_figures[scaled_step] = np.copy(water_height)
            w[scaled_step] = np.copy(temp_w)
        if __name__ == '__main__':
            print("step:",step)
        # Get the memory usage in bytes
       # memory_usage = psutil.virtual_memory().used
       # print("memory usage:",memory_usage/(1024**2),"MB")
        
    return  water_height_figures,w


# ===================================================================================
#               Running the simulation and showing the water spread
# ==================================================================================

#cProfile.run('simulation(figures_amount, water_height, water_height_figures)')
#tracemalloc.start()
# Ignore all Numba warnings
#This function will run the simulation and show the water spread
# 0: for running for Waal, 1: running for lek
def water_spread_run(water_height_river,river_type):
    # =============================================================================
    #                   Plotting the water level
    # =============================================================================
    # Initialize how the watermap is structured
    # Define a custom colormap that goes from green to blue
    global river_type1
    river_type1 = river_type
    colours = [(0, 1, 0),(0, 0, 1)]  # Blue to green
    global colormap_landuse
    colormap_landuse = LinearSegmentedColormap.from_list("my_list",colours)
    cmap = colormap_landuse
    # Initialize an array to store water_height figures for each iteration
    standard_figures_amount = 18000 # equivalent to 300 hours
    #discharge = 1000 # m3/s
    #water_level_min = 60*discharge /(100*100) # m/min (discharge in water level/tile/sec)
    global figures_amount 
    figures_amount = 1800               # This will create 10x iterations
    data_save_amount = int(figures_amount//10) # This will save the data every 10x iterations
    # this ratio determines how time is calculated in the animation/slider
    global step_to_min
    # Run for every step_to_min *minutes*
    step_to_min= standard_figures_amount/figures_amount
    frame_amount = 100
    global steps_to_reduce 
    steps_to_reduce= data_save_amount//(frame_amount)
    reduced_figures_amount = data_save_amount//steps_to_reduce
    water_height_figures = [np.zeros((land_height.shape)) for _ in range(data_save_amount)]
    # w = Rise rate of water(m\hour)
    water_height_zero = np.zeros(landscape_shape)  # You can set this as needed
    water_height_zero[np.isnan(AHN_gem)] = np.nan
    w = [np.where(np.isnan(water_height_zero), np.nan, water_height_zero) for _ in range(data_save_amount)]

    global short_water_height_figures
    short_water_height_figures = [np.zeros((land_height.shape)) for _ in range(reduced_figures_amount)]

    global fig
    global ax
    fig,ax = plt.subplots()
    
    warnings.filterwarnings("ignore", category=nb.NumbaWarning)
    start_time = time.time()

    global water_height_river1
    water_height_river1 = water_height_river
    
    print("Starting  Water Spread simulation with:",water_height_river,"m water level in the river")
    # 0: for running for Waal, 1: running for lek
    if river_type == 0:
        water_height_figures,w = simulation2(figures_amount, water_height, water_height_figures,w,water_height_river,step_to_min,0)
    elif river_type == 1:
        water_height_figures,w = simulation2(figures_amount, water_height, water_height_figures,w,water_height_river,step_to_min,1)
    end_time = time.time()
    print("Done with Water Spread simulation, time taken:",end_time-start_time,"seconds")

    if __name__ == "__main__":
        #short_water_height_figures = average_water_height(water_height_figures, steps_to_reduce,land_height,water_height_zero)
        short_water_height_figures = snap_water_height(water_height_figures,steps_to_reduce)
        #short_water_height_figures = water_height_figures
        print("Done with averaging figures")
        # you can choose to show the animation or the slider
        #animation = FuncAnimation(fig, update_water_animation, frames=len(short_water_height_figures),  repeat=True, interval=10)
        # Add a subplot for the slider below the main plot
        ax_slider = plt.axes([0.1, 0.02, 0.65, 0.03], facecolor='lightgoldenrodyellow')

        # Create a slider widget
        global step_slider
        step_slider = Slider(ax_slider, 'Step', 0, len(short_water_height_figures) - 1, valinit=0, valstep=1)
        step_slider.on_changed(update_slider)
        plt.show()
        print("Done with animation/ slider")
    return water_height_figures,w,short_water_height_figures,steps_to_reduce,step_to_min





if __name__ == "__main__":
    water_height_river = 10
    water_height_figures,w,short_water_height_figures,steps_to_reduce,step_to_min = water_spread_run(water_height_river,1)
    discharge = 0
    for i in range(0,figures_amount,1):
        discharge +=calc_discharge_river(figures_amount,step_to_min,1,water_height_river)   
        
    print("Step: 1-",figures_amount," river water height:",water_height_river," step_to_min:", step_to_min," discharge:",discharge)
# %%
# =============================================================================
#                                 Plotting the land datas
# =============================================================================


if __name__ == "__main__":
    print('Population: \nTotal population: ' + str(np.nansum(inhabitants)))
    """ Determine land-use classes"""
    # Initialization of land-use classes matrix: 9x1
    # Filling of land-use classes matrix. For loop trough all boxes
    lu_class_area = np.zeros((9,1))
    xrange = np.arange(0,223,1)
    yrange = np.arange(0,983,1)
    for ix in xrange:
        for iy in yrange:
            if ~np.isnan(landuse[ix,iy]):
                lu_class_area[int(landuse[ix,iy])-1,0] = lu_class_area[int(landuse[ix,iy])-1,0] + 0.01
                # Each cell adds 1 ha, or 0.01 km2 to its land-use class

    print('\nLanduse:')    
    print(pd.DataFrame(data=lu_class_area,index=["Infrastructure","Residential","Industry/commercial","Governmental institutions/services","Recreation","Greenhouses","Agriculture","Water","Nature"],columns = ["Area (km2)"]))           

    # Change land-use matrix
    for ix in xrange:
        for iy in yrange:
            if np.isnan(landuse[ix,iy]):
                landuse[ix,iy] = 0
    landuse = landuse.astype(int)

    """ Plotting - Examples """ 
    """ Plotting of data allows a direct first visual verification!"""
    # Plot the land-use
    fig,ax = plt.subplots()
    plt.title('Land Use Classes')
    colors = [(1,1,1),(0.0,0.0,0.0),(1.0,0.0,0.0),(1.0,0.0,1.0),(0.6,0.6,0.0),(1.0,1.0,0.0),(0.0,0.0,1.0),(0.3,1.0,0.2),(0.0,1.0,1.0),(0.1,0.4,0.1)]
    colormap_landuse = LinearSegmentedColormap.from_list("my_list",colors)
    cmap = colormap_landuse
    bounds = [0,1,2,3,4,5,6,7,8,9,10]
    norm = mpl.colors.BoundaryNorm(bounds,cmap.N)
    p1 = plt.pcolor(np.flipud(landuse),cmap=cmap,norm=norm)
    cb = fig.colorbar(p1,spacing='uniform')

    #ax.axis('equal')
    ax.axis('scaled')
    ax.set_xlabel('x (100m)')
    ax.set_ylabel('y (100m)')
    #plt.show()

    # Plot the maximum elevation
    fig,ax = plt.subplots()
    plt.title('Maximum elevation [m+NAP]')
    p2 = plt.pcolor(np.flipud(AHN_max),cmap='jet')
    cb = fig.colorbar(p2)
    ax.axis('scaled')
    ax.set_xlabel('x (100m)')
    ax.set_ylabel('y (100m)')
    #plt.show()

    # Plot the average elevation
    fig,ax = plt.subplots()
    plt.title('Average elevation [m+NAP]')
    p3 = plt.pcolor(np.flipud(AHN_gem),cmap='jet')
    cb = fig.colorbar(p3)
    ax.axis('scaled')
    ax.set_xlabel('x (100m)')
    ax.set_ylabel('y (100m)')
    #plt.show()

    # Plot dike ring areas
    fig,ax = plt.subplots()
    plt.title('Dike ring area [m+NAP]')
    p3 = plt.pcolor(np.flipud(dikeringarea),cmap='jet')
    cb = fig.colorbar(p3)
    ax.axis('scaled')
    ax.set_xlabel('x (100m)')
    ax.set_ylabel('y (100m)')
    #plt.show()

    # Plot population per hectare
    fig,ax = plt.subplots()
    plt.title('Population per hectare (10log)')

    #Adapt to 10logscale
    inhabitants_new = np.empty((223,983))
    for ix  in xrange:
        for iy in yrange:
            if inhabitants[ix,iy] > 0:
                inhabitants_new[ix,iy] = np.log10(inhabitants[ix,iy])
            if inhabitants_new[ix,iy] == 0:
                inhabitants_new[ix,iy] = np.nan
    p4 = plt.pcolor(np.flipud(inhabitants_new),cmap='hsv')
    cb = fig.colorbar(p4)
    ax.axis('scaled')
    ax.set_xlabel('x (100m)')
    ax.set_ylabel('y (100m)')
    #plt.show()

# %%
