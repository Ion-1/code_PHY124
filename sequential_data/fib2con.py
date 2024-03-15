# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 14:56:44 2024

@author: Ion-1
"""
import time


def lowest_prime_by_sieve(n, primes):

    prime_holder = [True for i in range(n+1)]

    for p in primes:
        for i in range(p * p, n+1, p):
            prime_holder[i] = False

    p = primes[-1]
    lowest = -1

    while (p * p <= n):

        p += 1

        if (prime_holder[p] == True):

            for i in range(p * p, n+1, p):
                prime_holder[i] = False

    for p in range(primes[-1], n+1):
        if prime_holder[p]:
            primes.append(p)

            if n % p == 0 and lowest == -1:
                lowest = p

    if n == lowest:
        return 1, primes

    return lowest, primes


def fib2con(s1, s2, n, lowest_prime, save_memory = False):

    res = [s1, s2]
    primes = [2]

    for i in range(2, n+1):
        
        sum_ = res[-1] + res[-2]

        for prime in primes:
            if sum_ % prime == 0:
                lowest = prime if prime != sum_ else 1
                break
        else:
            lowest, primes = lowest_prime(sum_, primes)

        res.append(int(sum_/lowest))
        if save_memory:
            res = res[-2:]

    return res


if __name__ == "__main__":
    start = time.time()
    print(fib2con(1, 1, 1000000000, lowest_prime_by_sieve, True)[-1])
    end = time.time()
    print(end-start)
