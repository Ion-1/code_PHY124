# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 14:56:44 2024

@author: Ion-1
"""


def fib2con(s1, s2, n):

    def lowest_prime(n, primes):

        for prime in primes:
            if n % prime == 0:
                return prime, primes
        else:
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
                    if n % p == 0:
                        lowest = p

            if n == lowest:
                return 1, primes
            else:
                return lowest, primes

    res = [s1, s2]
    primes = [2]

    for i in range(2, n+1):

        sum_ = res[i-1] + res[i-2]
        lowest, primes = lowest_prime(sum_, primes)

        res.append(int(sum_/lowest))

    return res


if __name__ == "__main__":
    print(fib2con(4, 3, 50)[24])
