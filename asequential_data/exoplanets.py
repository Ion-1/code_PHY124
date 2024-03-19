# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 19:40:41 2024.

@author: Ion-1
"""
import os
import csv
import matplotlib.pyplot as plt
import numpy as np
from railed import dictify
from BetterPlotting import plotter

np.seterr(divide='ignore')

filename = "exoplanet.eu_catalog.csv"

current_wdir = os.getcwd()

path = filename if os.path.isabs(
    filename) else os.path.join(current_wdir, filename)

if not os.path.exists(path):
    raise FileNotFoundError(f"path {path} does not exist")

with open(path, 'r', newline='') as csvfile:

    reader = csv.reader(csvfile)

    data = dictify(reader, 0, zero_value=None, key="# name", val1="star_radius",
                   val2="semi_major_axis", val3="star_teff", val4="mass")

similarity_s = {}
similarity_m = {}

for name, values in data.items():

    try:
        radius, axis, teff, mass = tuple(
            map(lambda x: float(x), values))
    except TypeError:
        continue

    similarity_s[name] = round(radius**2 * (teff/5800)**4 * axis**(-2), 3)

    similarity_m[name] = round(mass*317.8, 3)

limit = 2

weighting_s = 3
weighting_m = 1

# Weighted average percentage change: lose information on small scale
# similarity = {key: (weighting_s*abs(similarity_s[key]-1)+weighting_m*abs(similarity_m[key]-1))/(
#     weighting_m+weighting_s) for key in set(similarity_m.keys()).intersection(similarity_s.keys())}

# Weighted average logarithmic change
# similarity = {key: (weighting_s*abs(np.log(similarity_s[key])/np.log(limit))+weighting_m*abs(np.log(similarity_m[key])/np.log(limit)))/(
#     weighting_m+weighting_s) for key in set(similarity_m.keys()).intersection(similarity_s.keys())}

# Weighted tot logarithmic change
similarity = {key: (weighting_s*abs(np.log(similarity_s[key]))+weighting_m*abs(np.log(similarity_m[key])))/(
    weighting_m+weighting_s-1) for key in set(similarity_m.keys()).intersection(similarity_s.keys())}

counter = 0

for key, item in similarity.items():

    if item > abs(np.log(limit)):
        continue

    # if max((weighting_m*abs(np.log(similarity_m[key])), weighting_s*abs(np.log(similarity_s[key])))) > abs(np.log(limit)):
    #     continue

    # if max((weighting_m*abs(1-similarity_m[key]), weighting_s*abs(1-similarity_s[key]))) > abs(limit):
    #     continue

    counter += 1
    print(f"""Planet: {key}; \nTot. Change: {similarity[key]:.3f}; \nS.Sim: {similarity_s[key]:.3f}; Radius: {data[key][0]}; Axis: {
          data[key][1]}; Teff: {round(float(data[key][2])/5800, 3)}; \nMass: {round(320*float(data[key][3]), 3)};""", end="\n\n")

print(f"""{counter} total planets found within a factor of {limit} and weights of {
      weighting_s} and {weighting_m} for s and m, respectively.""", end="\n\n\n")

x, y = zip(*[(similarity_m[key], similarity_s[key])
           for key in similarity.keys()])

fig, ax = plt.subplots()
del ax

plotter(fig, x, y, y_lim=[0, 100], x_lim=[0, 60])

for key, item in {key: similarity[key] for key in ["GJ 667 C c", "Kepler-69 c", "LHS 1140 b", "TOI-700 d", "TRAPPIST-1 e", "GJ 273 b"]}.items():

    print(f"""Planet: {key}; \nTot. Change: {similarity[key]:.3f}; \nS.Sim: {similarity_s[key]:.3f}; Radius: {data[key][0]}; Axis: {
          data[key][1]}; Teff: {round(float(data[key][2])/5800, 3)}; \nMass: {round(320*float(data[key][3]), 3)};""", end="\n\n\n")
