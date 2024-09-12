# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 14:31:10 2024.

@author: Ion-1
"""
import os
import csv


def dictify(reader, header_index, zero_value='0', filter_=lambda index, row: True, **kwargs):

    indices = {}
    data = {}

    for index, row in enumerate(reader):

        if index < header_index:
            continue

        if index == header_index:
            for key, item in kwargs.items():
                indices[key] = row.index(item)
            continue
        
        if not filter_(index, row):
            continue

        data[row[indices['key']]] = [
            row[indices[f'val{i+1}']] if row[indices[f'val{i+1}']] else zero_value for i in range(len(kwargs)-1)]

    return data


def main(sorting_function, file_1, file_2, key, val1):

    current_wdir = os.getcwd()

    path_1 = file_1 if os.path.isabs(
        file_1) else os.path.join(current_wdir, file_1)
    path_2 = file_2 if os.path.isabs(
        file_2) else os.path.join(current_wdir, file_2)

    if not (os.path.exists(path_1) and os.path.exists(path_2)):
        raise FileNotFoundError('one of the two files does not exist')

    with open(path_1, 'r', newline='') as csvfile:

        reader = csv.reader(csvfile)

        pop_data = dictify(reader, 4, key=key, val1=val1)

    with open(path_2, 'r', newline='') as csvfile:

        reader = csv.reader(csvfile)

        rail_data = dictify(reader, 4, key=key, val1=val1)

    countries = set(pop_data.keys()).intersection(rail_data.keys())
    mod_countries = {}

    for country in countries:
        mod_countries[country] = sorting_function(
            *pop_data[country], *rail_data[country])

    return mod_countries


if __name__ == "__main__":

    pop_filename = "API_SP.POP.TOTL_DS2_en_csv_v2_3603750.csv"
    rail_filename = "API_IS.RRS.TOTL.KM_DS2_en_csv_v2_3479929.csv"

    year_of_inquiry = "2004"
    country_key = "Country Code"

    mod_countries = main(lambda x, y: round(1000*float(y)/float(x), 3)
                         if y != '0' and x != '0' else 0, pop_filename, rail_filename, country_key, year_of_inquiry)

    countries = list(mod_countries.keys())
    countries.sort(key=lambda x: mod_countries[x], reverse=True)

    print(countries[0], mod_countries[countries[0]])
