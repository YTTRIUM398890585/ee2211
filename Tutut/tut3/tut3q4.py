# outcome = (1, 2, 3, 4, 5, 6)
# pairs = {2: [], 3: [], 4: [], 5: [], 6: [],
#          7: [], 8: [], 9: [], 10: [], 11: [], 12: []}

# # part 1 and 2
# for i in outcome:
#     for j in outcome:
#         print(i, j, i+j)
#         if i+j in pairs:
#             pairs[i+j].append((i, j))

# print(pairs)

# # part 3
# total_count = 0
# event_count = 0

# for i in outcome:
#     for j in outcome:
#         for k in outcome:
#             total_count += 1
#             if i+j+k < 0.5*i*j*k:
#                 event_count += 1

# print("event_count = ", event_count)
# print("total_count = ", total_count)
# print("Pr(i+j+k < 0.5*i*j*k) = ", event_count/total_count)

# model answer way
from collections import defaultdict  # specialized container datatypes
from scipy import stats  # scientific computing and technical computing
import numpy as np

# part A

# Create a dictionary d, that maps a tuple (i, j) representing the two dice, to their sum
d = {(i, j): i+j for i in range(1, 7) for j in range(1, 7)}
print(d)

# Here we use defaultdict because it supports creating new keys arbitrarily
# We invert the dictionary d
# dinv maps integer values (the sums in d) to lists of tuples of how those sums can be obtained
dinv = defaultdict(list)
for i, j in d.items():
    dinv[j].append(i)
print(dinv)

# count the total number of ways to obtain a sum
total_ways = sum(len(i) for i in dinv.values())
print(total_ways)

# This maps the sums to the probability that those sums are obtained
# We count how many ways each sum can be obtained and divide by 36
X = {i: len(j)/total_ways for i, j in dinv.items()}
print(X)

# part B

# create dictionary, mapping tuple of 3 dice values to T/F value (if it meets the requirement)
d = {(i, j, k): ((i*j*k)/2 > i+j+k) for i in range(1, 7)
     for j in range(1, 7) for k in range(1, 7)}
print(d)

# invert the dictionary: turn keys into values and vice versa
dinv = defaultdict(list)
for i, j in d.items():
    dinv[j].append(i)
print(dinv)

# count the total number of ways to obtain a sum
total_ways = sum(len(i) for i in dinv.values())
print(total_ways)

# create a new dictionary, mapping keys T/F to values representing probability of that outcome
X={i:len(j)/total_ways for i,j in dinv.items() }
print(X)