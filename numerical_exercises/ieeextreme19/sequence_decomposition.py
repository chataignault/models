n = int(input().strip())
next_letter = {
    0:'1',
    1:'1',
    2:'2',
    3:'0',
    4:'1',
    5:'2',
    6: 'end'
}

for _ in range(n):
    seq = input().strip()
    k = 1
    fortunes = []
    done = 0
    tot = len(seq) / 6
    for el in seq:
        if not len(fortunes):
            assert el == next_letter[0]
            fortunes.append([1, [k]])
        else:
            inserted = False
            
            # Find all matching fortunes
            matching_indices = []
            for j in range(len(fortunes)):
                if el == next_letter[fortunes[j][0]]:
                    matching_indices.append(j)
            
            if matching_indices:
                # Find the shortest fortune among matches
                shortest_idx = min(matching_indices, key=lambda j: fortunes[j][0])
                
                fortunes[shortest_idx][0] += 1
                fortunes[shortest_idx][1].append(k)
                inserted = True
                
                # test whether the sequence is complete
                if next_letter[fortunes[shortest_idx][0]] == 'end': 
                    print(" ".join([str(x) for x in fortunes[shortest_idx][1]]))
                    fortunes.pop(shortest_idx)
                    done += 1
            elif tot > len(fortunes) + done and el == next_letter[0]:
                # Create new fortune if no match found and we need more fortunes
                fortunes.append([1, [k]])
                inserted = True
        k += 1
