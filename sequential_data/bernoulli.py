# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 15:32:43 2024

@author: Ion-1
"""
from fractions import Fraction as fr


def bernoulli_num_rec(n, full_list=False, called=False):

    if n == 0:
        return [fr(1)], [1, 1]

    b_list, factorials = bernoulli_num_rec(n-1, called=True)
    factorials.append((n+1)*factorials[n])

    sum_b = fr(0)

    for k in range(0, n):
        sum_b += fr(factorials[n]*b_list[k], factorials[k]*factorials[n-k+1])
    b_list.append(-sum_b)

    if called:
        return b_list, factorials

    return b_list if full_list else b_list[-1]


def bernoulli_num_loo(n, full_list=False):

    b_list, factorials = [fr(1)], [1, 1]
    counter = 0

    while counter < n:

        counter += 1

        factorials.append((counter+1)*factorials[counter])

        sum_b = fr(0)

        for k in range(0, counter):
            sum_b += fr(factorials[counter]*b_list[k],
                        factorials[k]*factorials[counter-k+1])

        b_list.append(-sum_b)

    return b_list if full_list else b_list[-1]


if __name__ == "__main__":
    print(bernoulli_num_rec(62, False))
    print(bernoulli_num_loo(62, False))
