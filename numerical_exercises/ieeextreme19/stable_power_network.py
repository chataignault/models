import heapq
from collections import defaultdict

def solve():
    T = int(input())
    
    for _ in range(T):
        N, M = map(int, input().split())
        
        # Build adjacency list
        graph = defaultdict(list)
        
        for _ in range(M):
            u, v, W, R = map(int, input().split())
            graph[u].append((v, W, R))
            graph[v].append((u, W, R))
        
        # Modified Dijkstra: priority is (max_risk, total_time)
        # State: (max_risk, total_time, node)
        pq = [(0, 0, 1)]  # Start from node 1 with 0 risk and 0 time
        best = {}  # best[node] = (best_max_risk, best_total_time for that risk)
        
        result = None
        
        while pq:
            max_risk, total_time, node = heapq.heappop(pq)
            
            # If we reached node N
            if node == N:
                result = (max_risk, total_time)
                break
            
            # Skip if we've already processed this node with better values
            if node in best:
                continue
            
            best[node] = (max_risk, total_time)
            
            # Explore neighbors
            for neighbor, weight, risk in graph[node]:
                if neighbor not in best:
                    new_max_risk = max(max_risk, risk)
                    new_total_time = total_time + weight
                    heapq.heappush(pq, (new_max_risk, new_total_time, neighbor))
        
        if result:
            print(result[0], result[1])
        else:
            print(-1)

solve()