# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 14:15:59 2024

@author: Ion-1
"""
# Functional Programming:


def filter_(seq, ignore):
    return list(filter(lambda x: x not in ignore, seq))


def find(seq, val):
    return list(
        filter(
            lambda x: x != None,
            map(lambda x, y: y if x == val else None, seq, range(len(seq))),
        )
    )


def common(seq1, seq2):
    return list(set(seq1).intersection(seq2))


# Non-functional and boring:


def filter_boring(seq, ignore):
    return [element for element in seq if element not in ignore]


def find_boring(seq, val):
    return [index for index in range(len(seq)) if seq[index] == val]


def common_boring(seq1, seq2):
    return list(set([element for element in seq1 if element in seq2]))
