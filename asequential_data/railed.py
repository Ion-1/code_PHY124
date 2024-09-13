# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 14:31:10 2024.

@author: Ion-1
"""
import os
import csv

from typing import Type, Callable


def dictify(reader: Type[csv.reader], primary_key: int | str, value_columns: list[int | str], header_index: int,
            default_filter: Callable[[str], bool] = lambda value: bool(value), default_value: str | None = None,
            row_filter: Callable[[int, list[str]], bool] = lambda index, row: True) -> dict[str, list[str | None]]:
    """
    A function to convert a table from a reader object to a dictionary, by taking the values of a column for the primary
    key and a list of some columns values for the value. The values from the primary column must be unique, or values
    will be silently overwritten.

    Parameters
    ----------
    reader : csv.reader
        A csv reader object with the table you want to convert to a dictionary
    primary_key : int | str
        If header_index is -1, then it is the index of the column which will be the keys of the dictionary.
        If else, it is the name of the column as found at the header_index row in the table.
    value_columns : list[int | str]
        If header_index is -1, then it is a list of indices for columns whose values will be extracted.
        If else, it is the names of columns as found at the header_index row in the table.
    header_index : int
        The row index of the table header. Used to convert primary key and value_columns strings to indices from the
        table. If the value is -1, which implies the primary key and value_columns values will be interpreted as integer
        indices.
    default_filter : Callable[[str], bool], optional
        A filter for the values in the table. If False the value will default to default_value.
        (The default is the truthiness of the value, i.e. an empty string).
    default_value : str, optional
        The default value when default_filter is False.
    row_filter : Callable[[int, list[str]], bool], optional
        A filter for an entire row, taking the index of the row and a list of its values as parameters.
        (The default always returns True).

    Returns
    -------
    A dictionary with the values from the primary_key column as the key and as the value a list of values from the
        values_columns columns.
    """
    indices: dict[int | str, int] = {}
    data: dict[str, list[str]] = {}

    if header_index == -1:
        if not (isinstance(1, int) and all(isinstance(val, int) for val in value_columns)):
            raise ValueError("header_index is -1 but primary_key or value_columns hold non-string values")
        indices: dict[int | str, int] = {ind: ind for ind in value_columns}
        indices["primary_key"]: int = primary_key

    for index, row in enumerate(reader):

        if index < header_index:
            continue

        if index == header_index:
            indices["primary_key"]: int = row.index(primary_key)
            for key in value_columns:
                indices[key]: int = row.index(key)
            continue

        if not row_filter(index, row):
            continue

        data[row[indices["primary_key"]]]: list[str] = [
            row[indices[key]] if default_filter(row[indices[key]]) else default_value for key in value_columns]

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

        pop_data = dictify(reader, header_index=4, primary_key=key, value_columns=[val1], default_value='0')

    with open(path_2, 'r', newline='') as csvfile:

        reader = csv.reader(csvfile)

        rail_data = dictify(reader, header_index=4, primary_key=key, value_columns=[val1], default_value='0')

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

    mod_countries = main(lambda x, y: round(1000 * float(y) / float(x), 3)
    if y != '0' and x != '0' else 0, pop_filename, rail_filename, country_key, year_of_inquiry)

    countries = list(mod_countries.keys())
    countries.sort(key=lambda x: mod_countries[x], reverse=True)

    print(countries[0], mod_countries[countries[0]])
