from math import comb
import numpy as np

# dynamic programming accumulators
rets = [[1], [1], [1, 1]]

max_n = 3

num = int(input())
for _ in range(num):
    s, n = [int(inp) for inp in input().split()]
    if max_n < n:
        for j in range(max_n, n):
            rets.append([1] + [rets[j-1][k] + rets[j-2][k-1] for k in range(1, (j+1)//2)])
            if j % 2 == 0:
                rets[j].append(1)
            max_n = n
    p = np.sum([comb(n-k, k) * 2.**(2*(k-n)) for k in range(n // 2 + 1)]) 
    if n == 1:
        p /= 1.
    else:
        p /= (1. - np.sum([2**(j-1-n) * rets[n-1][j-1] for j in range(1, (n+1)//2+1)]))**2
    res = round(19 * s * p)
    print(res)
