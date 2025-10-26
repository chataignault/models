N = int(input().strip())

kx = []
for _ in range(N):
    kx.append(int(input()))
    
for k in kx:
    if k % 2 == 1 or k <= 4:
        print(-1)
    else:
        p = 1
        while k % 2 ** p == 0: 
            p += 1
        p -= 1
        if k == 2**p:
            print(-1)
            continue
        p -= 1
        print(f"{k+2**p} {k // 2} {k // 2 - 2**p}")
