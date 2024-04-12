import numpy as np
import sympy as sym
import matplotlib.pyplot as plt
import pandas as pd
import pvlib
from pvlib.location import Location
import matplotlib.dates as mdates

figuresize = (15, 7)

timezone = "Europe/Copenhagen"
start_date = "2024-04-01"
end_date = "2024-04-30"
delta_time = "h"  # "Min", "H", 

# Definition of Location object. Coordinates and elevation of Amager, Copenhagen (Denmark)
site = Location(55.660439, 12.604980, timezone, 10, "Amager (DK)")  # latitude, longitude, time_zone, altitude, name

# Definition of a time range of simulation
times = pd.date_range(start_date + " 00:00:00", end_date + " 23:59:00", freq=delta_time, tz=timezone)
#Note - removed , closed="left"

t = np.linspace(0, 2 * np.pi, 1000)
f = np.cos(t)


plt.figure(figsize=figuresize)
plt.plot(t, f, color='b', linestyle='-')
plt.xlabel('t')
plt.ylabel('cos(t)')
#plt.show()


#print(t,f)  # uncomment to see the values of t and f

def solar_elevation_angle(theta):
    return (90 - theta)

def find_cordinates2(theta,phi,r):
    phi = np.deg2rad(phi)
    alpha = np.deg2rad(solar_elevation_angle(theta))
    z_cor = np.cos(alpha) * r
    b = np.sin(alpha) * r
    x_cor = np.cos(phi) * b
    y_cor = np.sin(phi) * b
    #print(np.sqrt(pow(x_cor, 2)+ pow(y_cor, 2)), b)
    #print(np.sqrt(pow(z_cor, 2)+ pow(b, 2)), r)
    return (x_cor, y_cor, z_cor)

def spherical_to_cartesian(theta, phi, r):
    #phi = np.deg2rad(phi)
    #theta = np.deg2rad(theta)
    # Radius r is 1 for unit vectors
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    return np.array([x, y, z])


def solar_panel_projection(theta_sun, phi_sun, theta_panel, phi_panel):
    us = spherical_to_cartesian(theta_sun, phi_sun, 1)
    up = spherical_to_cartesian(theta_panel, phi_panel, 1)
    return np.dot(us, up) if np.dot(us, up) > 0 else 0


#print(solar_panel_projection(np.pi / 4, np.pi, 0.0, np.pi))
# Estimate Solar Position with the 'Location' object

sunpos = site.get_solarposition(times)

# Visualize the resulting DataFrame
sunpos.head()

def plot_angles_day(day : str):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 10))
    fig.suptitle("Solar Position Estimation in " + site.name + day)
    ax1.plot(sunpos.loc[day].zenith)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%H"))

    ax2.plot(sunpos.loc[day].elevation)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%H"))
    
    ax3.plot(sunpos.loc[day].azimuth)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%H"))

    plt.show()

#plot_angles_day("2024-04-01")

def find_highest_angle(latitude, longitude, timezone, date):
    site = Location(latitude, longitude, timezone, 10)
    times = pd.date_range(date + " 00:00:00", date + " 23:59:00", freq='min', tz=timezone)
    sunpos = site.get_solarposition(times)
    return max(sunpos.loc[date].elevation)

#print(find_highest_angle(55.660439, 12.604980, timezone, "2024-04-01"))

def cor_sun_from_angles(theta_sun, phi_sun):
    #print(pvlib.solarposition.nrel_earthsun_distance(times) * 149597870700)
    distance_to_sun = 149597870700
    return spherical_to_cartesian(theta_sun, phi_sun, distance_to_sun)

def angles_from_cor_sun(cor):
    x, y, z = cor[0], cor[1], cor[2]
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r) 
    phi = np.arctan2(y, x)  
    return r, theta, phi

print(angles_from_cor_sun(cor_sun_from_angles(np.pi / 4, np.pi)))