# n, k = [int(c) for c in input().strip().split(' ')]
# if n == 1:
#     l = [int(input().strip())]
# else:
#     l = [int(c) for c in input().strip().split(' ')]

# jx = []
# for _ in range(k):
#     a = input().strip().split()[0]
#     a = a.replace(',', '.')
#     jx.append(int(a))


# if k > 0:
#     for j in jx:
#         start = 0
#         end = n-1
#         m = (start + end) // 2
#         while end - start > 1 and l[m] != j:
#             m = (start + end) // 2
#             if j < l[m]:
#                 end = m
#             else:
#                 start = m
#         if l[m] != j:
#             if l[start] == j:
#                 m = start
#             elif l[end] == j:
#                 m = end
#         if l[m] != j:
#             print('-1 -1')
#         else:  
#             r, q = 1, 1
#             while m >= r and l[m-r] == j:
#                 r += 1
#             r -= 1
#             while m+q < n and l[m+q] == j:
#                 q += 1
#             q -= 1
#             print(f"{m-r+1} {m+q+1}")

# n, k = map(int, input().strip().split())

# if n == 1:
#     l = [int(input().strip())]
# else:
#     l = list(map(int, input().strip().split()))

# for _ in range(k):
#     j = int(input().strip())
    
#     # Binary search to find any occurrence
#     left, right = 0, n - 1
#     found_idx = -1
    
#     while left <= right:
#         mid = (left + right) // 2
#         if l[mid] == j:
#             found_idx = mid
#             break
#         elif j < l[mid]:
#             right = mid - 1
#         else:
#             left = mid + 1
    
#     if found_idx == -1:
#         print('-1 -1')
#     else:
#         # Find first occurrence
#         first = found_idx
#         while first > 0 and l[first - 1] == j:
#             first -= 1
        
#         # Find last occurrence
#         last = found_idx
#         while last < n - 1 and l[last + 1] == j:
#             last += 1
        
#         print(f"{first + 1} {last + 1}")

n, k = map(int, input().strip().split())

if n == 1:
    l = [int(input().strip())]
else:
    l = list(map(int, input().strip().split()))

for _ in range(k):
    j = int(input().strip().split()[0])  # Take first token like original
    
    # Binary search to find any occurrence
    left, right = 0, n - 1
    found_idx = -1
    
    while left <= right:
        mid = (left + right) // 2
        if l[mid] == j:
            found_idx = mid
            break
        elif j < l[mid]:
            right = mid - 1
        else:
            left = mid + 1
    
    if found_idx == -1:
        print('-1 -1')
    else:
        # Find first occurrence
        first = found_idx
        while first > 0 and l[first - 1] == j:
            first -= 1
        
        # Find last occurrence
        last = found_idx
        while last < n - 1 and l[last + 1] == j:
            last += 1
        
        print(f"{first + 1} {last + 1}")