from itertools import combinations
from functools import reduce
import operator 

n, k = [int(c) for c in input().strip().split()]
l = [int(i) for i in input().strip().split()]



mk = 0.
N = 0
for j in range(n):
    for c in combinations(range(1,n), j):
        sub_l = [l[i] for i in c]
        if len(sub_l):
            mk += reduce(operator.xor, sub_l) ** k
        N += 1

s = str(mk / N)

ndec = len(s.split('.')[-1])
if ndec == 0:
    s += "00"
elif ndec == 1:
    s += "0"
else:
    s = s[:2-ndec]
    
print(s)