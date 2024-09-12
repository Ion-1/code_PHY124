# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 15:32:43 2024.

@author: Ion-1
"""
from fractions import Fraction as fr


def bernoulli_num_rec(n, full_list=False, called=False):
    """
    Calculate Bernoulli numbers using recursion.

    Parameters
    ----------
    n : int
        Bernoulli number index.
    full_list : bool, optional
        Whether the full list of numbers should be returned in ascending order.
        The default is False.
    called : bool, optional
        Whether the list of factorials should be returned, for recursion.
        The default is False.

    Returns
    -------
    fractions.Fraction; list of Fractions; tuple of two lists
        If called: tuple with list of every bernoulli up to n,
                              list of factorial up to n+1.
        If full_list: list of every bernoulli to n.
        Else: fractions.Fraction representing bernoulli number n.
    """
    if n == 0:
        if called:
            return [fr(1)], [1, 1]
        return fr(1)

    b_list, factorials = bernoulli_num_rec(n - 1, called=True)
    factorials.append((n + 1) * factorials[n])

    sum_b = fr(0)

    for k in range(n):
        sum_b += fr(
            factorials[n] * b_list[k], factorials[k] * factorials[n - k + 1]
        )
    b_list.append(-sum_b)

    if called:
        return b_list, factorials

    return b_list if full_list else b_list[-1]


def bernoulli_num_loo(n, full_list=False):
    """
    Calculate Bernoulli numbers using recursion.

    Parameters
    ----------
    n : int
        Bernoulli number index.
    full_list : bool, optional
        Whether the full list of numbers should be returned in ascending order.
        The default is False.

    Returns
    -------
    fractions.Fraction; list of Fractions
        If full_list: list of every bernoulli to n.
        Else: fractions.Fraction representing bernoulli number n.
    """
    b_list, factorials = [fr(1)], [1, 1]

    for counter in range(1, n+1):

        factorials.append((counter + 1) * factorials[counter])

        sum_b = fr(0)

        for k in range(counter):
            sum_b += fr(
                factorials[counter] * b_list[k],
                factorials[k] * factorials[counter - k + 1],
            )

        b_list.append(-sum_b)

    return b_list if full_list else b_list[-1]


def bernoulli_num_fun(n, full_list=False):
    """
    Calculate Bernoulli numbers in a fun one-liner.

    Parameters
    ----------
    n : int
        Bernoulli number index.
    full_list : bool, optional
        Whether the full list of numbers should be returned in ascending order.
        The default is False.

    Returns
    -------
    fractions.Fraction; list of Fractions
        If full_list: list of every bernoulli to n.
        Else: fractions.Fraction representing bernoulli number n.
    """
    return (
        lambda n, x, y: (
            x
            if [
                y.append((counter + 1) * y[counter])
                or x.append(
                    -sum(
                        map(
                            lambda k: fr(
                                y[counter] * x[k], y[k] * y[counter - k + 1]
                            ),
                            range(counter),
                        )
                    )
                )
                for counter in range(1, n + 1)
            ]
            and full_list
            else x[-1]
        )
    )(n, [fr(1)], [1, 1])


if __name__ == "__main__":
    print(bernoulli_num_rec(62))
    print(bernoulli_num_loo(62))
    print(bernoulli_num_fun(62))
