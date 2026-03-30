# test_operators.py

import copy
from random import random
from deap import tools
from GA import make_individual, NUM_PRODUCTS

print("=" * 50)
print("GA OPERATOR TEST")
print("=" * 50)

# --- Create 2 individuals ---
ind1 = make_individual()
ind2 = make_individual()

print("\n-- INDIVIDUAL 1 (before crossover)")
print(f"  S0 (pallets):  {ind1[0]}")
print(f"  S1 (robots):   {ind1[1]}")
print(f"  S2 (sequence): {ind1[2]}")

print("\n-- INDIVIDUAL 2 (before crossover)")
print(f"  S0 (pallets):  {ind2[0]}")
print(f"  S1 (robots):   {ind2[1]}")
print(f"  S2 (sequence): {ind2[2]}")

# --- Crossover ---
def cx_unique(a, b):
    a, b = list(a), list(b)
    if set(a) & set(b):  # shared values exist, skip
        return a, b
    tools.cxOnePoint(a,b)
    return a, b

def custom_crossover(ind1, ind2):
    c1, c2 = copy.deepcopy(ind1), copy.deepcopy(ind2)
    # S[0] - 
    c1[0], c2[0] = cx_unique(c1[0], c2[0])

    # S[1]
    tools.cxOnePoint(c1[1], c2[1])

    # S[2]
    c1[2] = [x - 1 for x in c1[2]]
    c2[2] = [x - 1 for x in c2[2]]
    tools.cxOrdered(c1[2], c2[2])
    c1[2] = [x + 1 for x in c1[2]]
    c2[2] = [x + 1 for x in c2[2]]

    return c1, c2

child1, child2 = custom_crossover(copy.deepcopy(ind1), copy.deepcopy(ind2))

print("\n-- CHILD 1 (after crossover)")
print(f"  S0 (pallets):  {child1[0]}")
print(f"  S1 (robots):   {child1[1]}")
print(f"  S2 (sequence): {child1[2]}")

print("\n-- CHILD 2 (after crossover)")
print(f"  S0 (pallets):  {child2[0]}")
print(f"  S1 (robots):   {child2[1]}")
print(f"  S2 (sequence): {child2[2]}")

# --- S2 permutation check ---
def is_valid_permutation(seq, n):
    return sorted(seq) == list(range(1, n + 1))

print("\n-- PERMUTATION VALIDITY CHECK (S2)")
print(f"  Parent1 S2 valid: {is_valid_permutation(ind1[2], NUM_PRODUCTS)}")
print(f"  Parent2 S2 valid: {is_valid_permutation(ind2[2], NUM_PRODUCTS)}")
print(f"  Child1  S2 valid: {is_valid_permutation(child1[2], NUM_PRODUCTS)}")
print(f"  Child2  S2 valid: {is_valid_permutation(child2[2], NUM_PRODUCTS)}")

# --- S1 consistency check ---
print("\n-- S1 CONSISTENCY CHECK (robot assignments not None)")
print(f"  Child1 S1 has None: {None in child1[1]}")
print(f"  Child2 S1 has None: {None in child2[1]}")

print("\n" + "=" * 50)
print("TEST COMPLETE")
print("=" * 50)