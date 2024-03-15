# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 19:40:41 2024.

@author: Ion-1
"""
import os
import csv
import matplotlib.pyplot as plt
from railed import dictify
from BetterPlotting import plotter

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

    radius, axis, teff, mass = values

    if not (radius and axis and teff and mass):
        continue

    radius, axis, teff, mass = list(
        map(lambda x: float(x), [radius, axis, teff, mass]))

    similarity_s[name] = round(radius**2 * (teff/5800)**4 * axis**(-2), 3)

    similarity_m[name] = round(mass*320, 3)

weighting_s = 3
weighting_m = 0.5

similarity = {key: (weighting_s*similarity_s[key]+weighting_m*similarity_m[key])/(
    weighting_m+weighting_s) for key in set(similarity_m.keys()).intersection(similarity_s.keys())}

limit = 0.5
counter = 0

for key, item in similarity.items():

    if abs(item-1) > limit:
        continue

    counter += 1
    print(f"Planet: {key}; Tot. Sim.: {similarity[key]}; S.Sim: {similarity_s[key]}; Radius: {data[key][0]}; Axis: { \
          data[key][1]}; Teff: {round(float(data[key][2])/5800, 3)}; Mass: {round(320*float(data[key][3]), 3)};", end="\n\n")

print(f"{counter} total planets found within a limit of {limit*100}% and weights of {weighting_s} and {weighting_m} for s and m, respectively.")

# x, y = zip(*[(similarity_m[key], similarity_s[key])
#            for key in set(similarity_m.keys()).intersection(similarity_s.keys())])

# fig, ax = plt.subplots()
# del ax

# plotter(fig, x, y, y_lim=[0, 100], x_lim=[0, 100])

print("\n\n")
for key, item in {key: similarity[key] for key in ["GJ 667 C c","Kepler-69 c"]}.items():

    counter += 1
    print(f"Planet: {key}; Tot. Sim.: {similarity[key]}; S.Sim: {similarity_s[key]}; Radius: {data[key][0]}; Axis: {
          data[key][1]}; Teff: {round(float(data[key][2])/5800, 3)}; Mass: {round(320*float(data[key][3]), 3)};")
