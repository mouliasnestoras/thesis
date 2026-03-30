import random
import copy
import time
import matplotlib.pyplot as plt
from deap import base, creator, tools


from simulation_v2 import Simulator, load_environment
from simulation_v3 import Simulator as SimulatorV3  

start = time.perf_counter()
products, robot_starting_pos, orders = load_environment("instances/data_1.txt") # instances/data_1.txt 
end_time = time.perf_counter()

def objective(s0, s1, s2):
    solution = [s0, s1, s2]
    
    sim = SimulatorV3(solution=solution, starting_pos=robot_starting_pos, policy="priority") # default | priority
    sim.schedule_jobs(products)
    fitness , is_valid = sim.simulation2()
    
    return fitness, is_valid

# -- DEAP setup --------------------------------------------
creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))  # minimize both validity and makespan
creator.create("Individual", list, fitness=creator.FitnessMin)

MAX_PALLET   = robot_starting_pos[1] - 1
NUM_PRODUCTS = len(products)
NUM_ORDERS   = orders

def make_individual():
    s0 = random.sample(range(1, MAX_PALLET + 1), NUM_ORDERS)
    s1 = [random.randint(0, 1) for _ in range(NUM_PRODUCTS)]
    s2 = random.sample(range(1, NUM_PRODUCTS + 1), NUM_PRODUCTS)
    return creator.Individual([s0, s1, s2])

def evaluate(ind):
    fitness, is_valid = objective(ind[0], ind[1], ind[2])
    validity_score = 0 if is_valid else 1  # 0 = valid, 1 = invalid
    return (validity_score, fitness)  # DEAP minimizes both, validity wins first

# -- Crossover ------------------------------------------------
def cx_s0_pmx(a, b):
    """
    PMX crossover for S[0]: a subset-permutation of pallet locations.
    """
    a, b = list(a), list(b)
    size = len(a)

    # Pick two crossover points
    cx1, cx2 = sorted(random.sample(range(size), 2))

    def pmx_one(p1, p2):
        child = [None] * size
        # Copy the segment from p1
        child[cx1:cx2] = p1[cx1:cx2]
        segment_vals = set(p1[cx1:cx2])

        # Fill from p2, resolving conflicts via mapping
        for i in list(range(0, cx1)) + list(range(cx2, size)):
            candidate = p2[i]
            while candidate in segment_vals:
                # Follow the mapping: where does candidate appear in p1 segment?
                idx = p1[cx1:cx2].index(candidate)
                candidate = p2[cx1:cx2][idx]
            child[i] = candidate
        return child

    return pmx_one(a, b), pmx_one(b, a)

def custom_crossover(ind1, ind2):
    c1, c2 = ind1, ind2
    # S[0] - 
    c1[0], c2[0] = cx_s0_pmx(c1[0], c2[0])

    # S[1]
    #tools.cxUniform(c1[1], c2[1], 0.5)
    tools.cxTwoPoint(c1[1], c2[1])
    # S[2]
    # reduce by 1 for 0-based indexing, apply ordered crossover, then restore to 1-based
    c1[2] = [x - 1 for x in c1[2]]
    c2[2] = [x - 1 for x in c2[2]]
    tools.cxOrdered(c1[2], c2[2])
    c1[2] = [x + 1 for x in c1[2]]
    c2[2] = [x + 1 for x in c2[2]]

    return c1, c2

# -- Mutation ------------------------------------------------
def mutate_pallet_locations(locs, max_pallet, indpb=0.2):
    """Replace each location with a random unused one with probability indpb."""
    locs = list(locs)
    for i in range(len(locs)):
        if random.random() < indpb:
            used = set(locs) - {locs[i]}
            available = [x for x in range(1, max_pallet + 1) if x not in used]
            if available:
                locs[i] = random.choice(available)
    return locs

def custom_mutate(ind):
    m = ind

    # --- S[0]: pallet locations ---
    m[0] = mutate_pallet_locations(m[0], MAX_PALLET, indpb=0.6)

    # --- S[1]: robot assignment ---
    tools.mutUniformInt(m[1], low=0, up=1, indpb=0.2)

    # --- S[2]: pickup sequence ---
    tools.mutShuffleIndexes(m[2], indpb=0.4)

    return (m,)

toolbox = base.Toolbox()
toolbox.register("individual", make_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate",   evaluate)
toolbox.register("mate",       custom_crossover)
toolbox.register("mutate",     custom_mutate)
toolbox.register("select",     tools.selTournament, tournsize=3)

# -- Main loop ------------------------------------------------
def run_ga(pop_size=200, n_gen=200, cxpb=0.7, mutpb=0.2):

    gen_min = []
    gen_avg = []
    best_valid = None

    pop = toolbox.population(n=pop_size)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min_makespan", lambda x: min(v[1] for v in x if v[0] == 0) if any(v[0] == 0 for v in x) else float('inf'))
    stats.register("avg_makespan", lambda x: round(sum(v[1] for v in x) / len(x), 2))
    stats.register("valid_count",  lambda x: sum(1 for v in x if v[0] == 0))

    # --- Initial evaluation ---
    for ind in pop:
        ind.fitness.values = toolbox.evaluate(ind)

    print(f"{'Gen':>4} | {'Valid':>6} | {'Best':>10} | {'Avg':>10}")
    print("-" * 40)

    for gen in range(1, n_gen + 1):

        # --- 1. Selection ---
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(copy.deepcopy, offspring))

        # --- 2. Crossover ---
        for i in range(0, len(offspring) - 1, 2):
            if random.random() < cxpb:
                offspring[i], offspring[i+1] = toolbox.mate(offspring[i], offspring[i+1])
                del offspring[i].fitness.values
                del offspring[i+1].fitness.values

        # --- 3. Mutation ---
        current_mutpb = mutpb + (0.6 - mutpb) * (gen / n_gen)

        for ind in offspring:
            if random.random() < mutpb:
                (ind,) = toolbox.mutate(ind)
                del ind.fitness.values

        # --- 4. Evaluation ---
        invalid = [ind for ind in offspring if not ind.fitness.valid]
        for ind in invalid:
            ind.fitness.values = toolbox.evaluate(ind)

        # --- 5. Replace ---
        pop[:] = offspring

        # --- 6. Track best valid ---
        for ind in pop:
            if ind.fitness.values[0] == 0:
                if best_valid is None or ind.fitness.values[1] < best_valid.fitness.values[1]:
                    best_valid = copy.deepcopy(ind)

        # --- 7. Elitism ---
        if best_valid is not None:
            worst_idx = max(range(len(pop)), key=lambda i: pop[i].fitness.values)
            pop[worst_idx] = copy.deepcopy(best_valid)

        # --- Stats ---
        record = stats.compile(pop)
        gen_min.append(record['min_makespan'])
        gen_avg.append(record['avg_makespan'])

        print(f"{gen:>4} | {record['valid_count']:>6} | {record['min_makespan']:>10.4f} | {record['avg_makespan']:>10.4f}")

    # --- Final result ---
    print("\n-- Best Valid Individual --")
    if best_valid:
        print(f"  S0 (pallets):   {best_valid[0]}")
        print(f"  S1 (robots):    {best_valid[1]}")
        print(f"  S2 (sequence):  {best_valid[2]}")
        print(f"  Fitness:        {best_valid.fitness.values[1]:.4f}")
    else:
        print("  No valid individual found.")

    # --- Plot ---
    plt.figure(figsize=(10, 5))
    valid_gens = [(i, v) for i, v in enumerate(gen_min) if v != float('inf')]
    if valid_gens:
        xs, ys = zip(*valid_gens)
        plt.plot(xs, ys, label='Best valid fitness')
    plt.plot(gen_avg, label='Average fitness')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('GA convergence')
    plt.legend()
    plt.grid(True)
    plt.show()

    return best_valid, pop

if __name__ == "__main__":
    best, final_pop = run_ga()
    print(f"\nTotal execution time: {end_time - start:.2f} seconds")