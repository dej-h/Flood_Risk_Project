# -*- coding: utf-8 -*-
#%%
# =============================================================================
# This file will take the river_water_height and the impact of casualties and financial damage and plot them
# =============================================================================

# =============================================================================
#                       Importing Data and packages
# =============================================================================
import pandas as pd
import numpy as np
import numba as nb
import time
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=nb.NumbaWarning)
import Water_spread as ws
import casualties as cs
import Flood_damage as fd
from matplotlib.colors import SymLogNorm

#==============================================================================
#                               Loading data
#==============================================================================
AHN_gem = pd.read_csv('AHN_avg.txt', header = None, delimiter = ';').to_numpy()

landscape_shape = AHN_gem.shape
# Getting the river water height against the return period in years
lek_data = pd.read_excel('Lek_data.xlsx')
waal_data = pd.read_excel('waal_data.xlsx')
# Selecting columns C and D, excluding the first row
lek_data_subset = lek_data.iloc[1:, [2, 3]]
waal_data_subset = waal_data.iloc[1:, [2, 3]]

# Plotting for Lek Data
plt.figure(figsize=(8, 6))
plt.scatter(lek_data_subset.iloc[:, 0], lek_data_subset.iloc[:, 1], label='Lek Data', color='blue')

# Plotting for Waal Data
plt.scatter(waal_data_subset.iloc[:, 0], waal_data_subset.iloc[:, 1], label='Waal Data', color='red')

# Adding a grey line at y_value of 10^5
plt.axhline(y=1e5, color='grey', linestyle='--', label='Cut-off return period')

# Adding labels and title
plt.xlabel('Water Height River (m)')
plt.ylabel('Return Period (years)')
plt.title('Scatter Plot of Water Height River against Return Period (years)')
plt.legend()  # Show legend

# Set y axes to logarithmic scale
plt.yscale('log')

# Show the plot
#plt.show()

# This function will find the closest return period for a given river water height
def closest_return_period(river_water_height, data_subset):
    # Assuming data_subset has two columns: Water Height River and Return Period
    # Calculate the Euclidean distance to find the closest point
    distances = np.sqrt((data_subset.iloc[:, 0] - river_water_height)**2 )#+ (data_subset.iloc[:, 1] - 1e5)**2)
    
    # Find the index of the minimum distance
    closest_index = np.argmin(distances)
    
    # Return the corresponding Return Period
    return data_subset.iloc[closest_index, 1]





# =============================================================================
#                       Plotting the impact of casualties and financial damage
# =============================================================================
# This function will plot the impact of casualties and financial damage for different river heights
# 0: for running for the Waal, 1: running for the Lek
def plotting_impact_river_height(river_type):
    # amount of river heights: 0-12:
    river_heights = np.arange(12, 12, 2)
    financial_damages = np.zeros(len(river_heights))
    casualties_low_fe = np.zeros(len(river_heights))
    casualties_high_fe = np.zeros(len(river_heights))
    return_periods = np.zeros(len(river_heights))
    risks_casualties = np.zeros(len(river_heights))
    risks_financial_damage = np.zeros(len(river_heights))
    
    print("  ")
    print("Start of the simulation")
    start_time = time.time()
    low_fe = 0.80
    high_fe = 0.98
    for i in range(len(river_heights)):
        water_height_figures, w, short_water_height_figures, steps_to_reduce, figures_ratio = ws.water_spread_run(river_heights[i],river_type)
        financial_damages[i] = fd.run_flood_damage(water_height_figures, figures_ratio)
        casualties_low_fe[i], casualties_high_fe[i], casualty_map = cs.run_casualties(water_height_figures, w, low_fe, high_fe)
        if river_type == 0:
            return_periods[i] = closest_return_period(river_heights[i], waal_data_subset)
        elif river_type == 1:
            return_periods[i] = closest_return_period(river_heights[i], lek_data_subset)
        risks_financial_damage[i] = financial_damages[i] * (1/return_periods[i])
        risks_casualties[i] = casualties_high_fe[i] * (1/return_periods[i])
        
    end_time = time.time()
    print("End of the simulation, Time taken: ",
          end_time - start_time, " seconds")
    # Plotting the impact of financial damage
    plt.close("all")
    # plotting financial damage for the river heights
    plt.figure()
    plt.plot(river_heights, financial_damages, label="Financial Damage")
    plt.xlabel("River height (m)")
    plt.ylabel("Financial damage (euro)")
    if river_type == 0:
        river_title = "The river Waal"
    elif river_type == 1:
        river_title = "The river Lek"
    plt.title(river_title + " Financial damage/river height")
    plt.legend()
    plt.grid(True)
    plt.show()

    

    
    # Plotting the impact of casualties for river height
    plt.figure()
    plt.plot(river_heights, casualties_low_fe,
             label="Low Evacuation Factor (f_e)")
    plt.plot(river_heights, casualties_high_fe,
             label="High Evacuation Factor (f_e)")
    plt.xlabel("River height (m)")
    plt.ylabel("Casualties")
    plt.title(river_title+" Casualties/river heights")
    plt.legend()
    plt.grid(True)
    plt.show()
    # plotting financial damage for the return periods
    plt.figure()
    plt.plot(return_periods, financial_damages, label="Financial Damage")
    plt.xlabel("Return Period (years)")
    plt.ylabel("Financial damage (euro)")
    plt.title(river_title+" Financial damage/return period")
    plt.legend()
    plt.grid(True)
    plt.xscale('log')
    plt.show()

    # Plotting the impact of casualties for return period
    plt.figure()
    plt.plot(return_periods, casualties_low_fe,
             label="Low Evacuation Factor (f_e)")
    plt.plot(return_periods, casualties_high_fe,
             label="High Evacuation Factor (f_e)")
    plt.xlabel("Return Period (years)")
    plt.ylabel("Casualties")
    plt.title(river_title+" Casualties /Return Period")
    plt.legend()
    plt.xscale('log')
    plt.grid(True)
    plt.show()
    
    print("Risk for Casualties:", risks_casualties)
    print("Risk for Financial Damage:", risks_financial_damage)


#Running for Waal: 0
#plotting_impact_river_height(0)
#Running for Lek: 1
#plotting_impact_river_height(1)

import numpy as np
import matplotlib.pyplot as plt

# Define data
river_heights = np.arange(0, 18, 2)
risk_casualties = np.array([0.00000000e+00, 2.00000000e+00, 2.38999531e+02, 5.66427757e+02,
                            1.02599189e+03, 1.64547526e+02, 6.25476193e+00, 1.28527100e-01,
                            1.07267113e-03])
risk_financial_damage = np.array([0.00000000e+00, 1.88119300e+06, 2.20541008e+07, 5.87243648e+07,
                                  8.17472149e+07, 1.36607407e+07, 4.13874159e+05, 9.27493822e+03,
                                  8.21730939e+01])

# Plot for Risk of Casualties
plt.figure(figsize=(8, 6))
plt.plot(river_heights, risk_casualties, marker='o', linestyle='-', color='b')
plt.title('Waal: Risk of Casualties vs. River Heights')
plt.xlabel('River Heights')
plt.ylabel('Risk for Casualties')
plt.grid(True)
#plt.show()

# Plot for Risk of Financial Damage
plt.figure(figsize=(8, 6))
plt.plot(river_heights, risk_financial_damage, marker='o', linestyle='-', color='r')
plt.title('Waal: Risk of Financial Damage vs. River Heights')
plt.xlabel('River Heights')
plt.ylabel('Risk for Financial Damage')
plt.grid(True)
#plt.show()

# Define data for The Lek
risk_casualties_lek = np.array([0.00000000e+00, 0.00000000e+00, 2.07963914e+02, 2.24582889e+02,
                                3.38959395e+01, 4.88491530e-01, 5.05448945e-03, 1.92815590e-05,
                                4.18119519e-08])
risk_financial_damage_lek = np.array([0.00000000e+00, 4.37638930e+05, 3.51198816e+06, 8.44972943e+06,
                                      4.80567000e+06, 1.02063841e+05, 9.68259668e+02, 3.53383842e+00,
                                      6.71329532e-03])

# Plot for Risk of Casualties for The Lek
river_heights = np.arange(0, 18, 2)
plt.figure(figsize=(8, 6))
plt.plot(river_heights, risk_casualties_lek, marker='o', linestyle='-', color='b')
plt.title('The Lek: Risk of Casualties vs. River Heights')
plt.xlabel('River Heights')
plt.ylabel('Risk for Casualties')
plt.grid(True)
#plt.show()

# Plot for Risk of Financial Damage for The Lek
plt.figure(figsize=(8, 6))
plt.plot(river_heights, risk_financial_damage_lek, marker='o', linestyle='-', color='r')
plt.title('The Lek: Risk of Financial Damage vs. River Heights')
plt.xlabel('River Heights')
plt.ylabel('Risk for Financial Damage')
plt.grid(True)
#plt.show()

#Risk of casualties and financial damage * prob that dyke breaches Define data for Waal
river_heights = np.arange(0, 18, 2)
multiplier_array = np.array([0,0,0,1.08737021e-07, 1.20188151e-04, 1.53531984e-02, 2.58035815e-01,
 8.05664880e-01, 9.91189220e-01])



# Apply the multiplier to Risk for Casualties and Risk for Financial Damage
risk_casualties_waal = np.array([0.00000000e+00, 2.00000000e+00, 2.38999531e+02, 5.66427757e+02,
                                 1.02599189e+03, 1.64547526e+02, 6.25476193e+00, 1.28527100e-01,
                                 1.07267113e-03]) * multiplier_array
risk_financial_damage_waal = np.array([0.00000000e+00, 1.88119300e+06, 2.20541008e+07, 5.87243648e+07,
                                       8.17472149e+07, 1.36607407e+07, 4.13874159e+05, 9.27493822e+03,
                                       8.21730939e+01]) * multiplier_array
print("total risk casualties:",risk_casualties_waal)
print("total risk financial damage:",risk_financial_damage_waal)

# Plot for Risk of Casualties for Waal
plt.figure(figsize=(8, 6))
plt.plot(river_heights, risk_casualties_waal, marker='o', linestyle='-', color='b')
plt.title('Waal: Risk of Casualties vs. River Heights for Piping')
plt.xlabel('River Heights')
plt.ylabel('Risk for Casualties')
plt.grid(True)
plt.show()

# Plot for Risk of Financial Damage for Waal
plt.figure(figsize=(8, 6))
plt.plot(river_heights, risk_financial_damage_waal, marker='o', linestyle='-', color='r')
plt.title('Waal: Risk of Financial Damage vs. River Heights for Piping')
plt.xlabel('River Heights')
plt.ylabel('Risk for Financial Damage')
plt.grid(True)
plt.show()

# For lek with multiplier of piping probabilty
multiplier_array = [0,5.41702110e-09, 1.19027462e-05, 3.12331653e-03, 1.06886756e-01,
 5.97946023e-01, 9.59011139e-01, 9.99382378e-01, 9.99998832e-01]
total_risk_casualties_lek = risk_casualties_lek * multiplier_array
total_risk_financial_damage_lek = risk_financial_damage_lek * multiplier_array
print("total risk casualties:",total_risk_casualties_lek)
print("total risk financial damage:",total_risk_financial_damage_lek)

# plot total risk of casualties for lek
plt.figure(figsize=(8, 6))
plt.plot(river_heights, total_risk_casualties_lek, marker='o', linestyle='-', color='b')
plt.title('Lek: Risk of Casualties vs. River Heights for Piping')
plt.xlabel('River Heights')
plt.ylabel('Risk for Casualties')
plt.grid(True)
plt.show()

# plot total risk of financial damage for lek
plt.figure(figsize=(8, 6))
plt.plot(river_heights, total_risk_financial_damage_lek, marker='o', linestyle='-', color='r')
plt.title('Lek: Risk of Financial Damage vs. River Heights for Piping')
plt.xlabel('River Heights')
plt.ylabel('Risk for Financial Damage')
plt.grid(True)
plt.show()
#%%
water_height_figures, w, short_water_height_figures, steps_to_reduce, figures_ratio = ws.water_spread_run(12,0)
fd.run_flood_damage(water_height_figures, figures_ratio)

# %%
