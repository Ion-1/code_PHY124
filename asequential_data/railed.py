# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 14:31:10 2024.

@author: Ion-1
"""
import os
import csv


def dictify(reader, header_index, zero_value='0', **kwargs):

    indices = {}
    data = {}

    for index, row in enumerate(reader):

        if index < header_index:
            continue

        if index == header_index:
            for key, item in kwargs.items():
                indices[key] = row.index(item)
            continue

        data[row[indices['key']]] = [
            row[indices[f'val{i+1}']] if row[indices[f'val{i+1}']] else zero_value for i in range(len(kwargs)-1)]

    return data


def main(sorting_function):

    pop_filename = "API_SP.POP.TOTL_DS2_en_csv_v2_3603750.csv"
    rail_filename = "API_IS.RRS.TOTL.KM_DS2_en_csv_v2_3479929.csv"

    current_wdir = os.getcwd()

    pop_path = pop_filename if os.path.isabs(
        pop_filename) else os.path.join(current_wdir, pop_filename)
    rail_path = rail_filename if os.path.isabs(
        rail_filename) else os.path.join(current_wdir, rail_filename)

    if not (os.path.exists(pop_path) and os.path.exists(rail_path)):
        raise FileNotFoundError('one of the two files does not exist')

    year_of_inquiry = "2004"
    country_key = "Country Code"

    with open(pop_path, 'r', newline='') as csvfile:

        reader = csv.reader(csvfile)

        pop_data = dictify(reader, 4, key=country_key, val1=year_of_inquiry)

    with open(rail_path, 'r', newline='') as csvfile:

        reader = csv.reader(csvfile)

        rail_data = dictify(reader, 4, key=country_key, val1=year_of_inquiry)

    countries = set(pop_data.keys()).intersection(rail_data.keys())
    mod_countries = {}

    for country in countries:
        mod_countries[country] = sorting_function(
            *pop_data[country], *rail_data[country])

    return mod_countries


if __name__ == "__main__":

    mod_countries = {key: round(item*1000, 3) for key, item in main(
        lambda x, y: float(y)/float(x) if y != '0' and x != '0' else 0).items()}

    highest = 0
    country_code = 'NO'

    for key, item in mod_countries.items():

        if item > highest:
            highest = item
            country_code = key

    print(country_code, highest)
