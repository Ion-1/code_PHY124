# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 14:42:09 2024

@author: Ion-1
"""
import time

def base_recursive(n, base): return (base_recursive(n//base, base) + ['0','1','2','3','4','5','6','7','8','9','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','!','@','#','$','%','^','&','*','(',')','-','_','=','+','ö','ä','ü'][n%base] if n > 0 else "")

    
def base_looped(n, base):
    
    ans = ""
    
    while n > 0:
        
        ans = ['0','1','2','3','4','5','6','7','8','9','a','b','c','d','e','f',
               'g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v',
               'w','x','y','z','A','B','C','D','E','F','G','H','I','J','K','L',
               'M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','!','@',
               '#','$','%','^','&','*','(',')','-','_','=','+','ö','ä','ü'
               ][n%base] + ans
        
        n //= base
    
    return ans, len(ans)

if __name__ == "__main__":
    number = 2**2**2**2**2
    # number = 299792458 
    # number = 90094823
    base = 15
    try:
        start = time.time()
        print(base_recursive(number, base))
        end = time.time()
        print(end-start)
    except RecursionError:
        print("Lol")
    start = time.time()
    print(base_looped(number, base))
    end = time.time()
    print(end-start)