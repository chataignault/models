def min_distance(pos_from, pos_to, n):
    """Calculate minimum distance on circular table"""
    clockwise = (pos_to - pos_from) % n
    counterclockwise = (pos_from - pos_to) % n
    return min(clockwise, counterclockwise)

def calculate_max_distance(initial, target, n):
    """Calculate max distance for a given target configuration"""
    max_dist = 0
    for i in range(n):
        friend = initial[i]
        # Find where this friend should go in target
        target_pos = target.index(friend)
        dist = min_distance(i, target_pos, n)
        max_dist = max(max_dist, dist)
    return max_dist

n = int(input())
initial = list(map(int, input().split()))

min_max_distance = float('inf')

# Try all clockwise configurations: [1,2,3,...,N] starting at each position
clockwise_order = list(range(1, n + 1))
for start in range(n):
    target = clockwise_order[start:] + clockwise_order[:start]
    max_dist = calculate_max_distance(initial, target, n)
    min_max_distance = min(min_max_distance, max_dist)

# Try all counterclockwise configurations: [N,N-1,...,1] starting at each position
counterclockwise_order = list(range(n, 0, -1))
for start in range(n):
    target = counterclockwise_order[start:] + counterclockwise_order[:start]
    max_dist = calculate_max_distance(initial, target, n)
    min_max_distance = min(min_max_distance, max_dist)

print(min_max_distance)