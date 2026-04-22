import random
import copy
import time
import argparse
import matplotlib.pyplot as plt
from deap import base, creator, tools

from simulation_v2 import Simulator, load_environment
from simulation_v3 import Simulator as SimulatorV3

# -- DEAP creator classes (declared once at module level) -----------------
# Guarded so re-imports (e.g. from a runner script that also imports GA.py)
# do not raise the "class already created" warning/error.
if not hasattr(creator, "FitnessMin"):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMin)

# -- Environment globals ---------------------------------------------------
# Populated by run_ga() based on the instance path.
products = None
robot_starting_pos = None
orders = None
MAX_PALLET = None
NUM_PRODUCTS = None
NUM_ORDERS = None


def objective(s0, s1, s2):
    solution = [s0, s1, s2]
    sim = SimulatorV3(solution=solution, starting_pos=robot_starting_pos, policy="priority")  # default | priority
    sim.schedule_jobs(products)
    fitness, is_valid = sim.simulation2()
    return fitness, is_valid


def make_individual():
    s0 = random.sample(range(1, MAX_PALLET + 1), NUM_ORDERS)
    s1 = [random.randint(0, 1) for _ in range(NUM_PRODUCTS)]
    s2 = random.sample(range(1, NUM_PRODUCTS + 1), NUM_PRODUCTS)
    return creator.Individual([s0, s1, s2])


def evaluate(ind):
    fitness, is_valid = objective(ind[0], ind[1], ind[2])
    validity_score = 0 if is_valid else 1
    return (validity_score, fitness)


# -- Crossover ------------------------------------------------
def cx_s0_pmx(a, b):
    """PMX crossover for S[0]: a subset-permutation of pallet locations."""
    a, b = list(a), list(b)
    size = len(a)
    cx1, cx2 = sorted(random.sample(range(size), 2))

    def pmx_one(p1, p2):
        child = [None] * size
        child[cx1:cx2] = p1[cx1:cx2]
        segment_vals = set(p1[cx1:cx2])
        for i in list(range(0, cx1)) + list(range(cx2, size)):
            candidate = p2[i]
            while candidate in segment_vals:
                idx = p1[cx1:cx2].index(candidate)
                candidate = p2[cx1:cx2][idx]
            child[i] = candidate
        return child

    return pmx_one(a, b), pmx_one(b, a)


def custom_crossover(ind1, ind2):
    c1, c2 = ind1, ind2
    c1[0], c2[0] = cx_s0_pmx(c1[0], c2[0])

    tools.cxTwoPoint(c1[1], c2[1])

    c1[2] = [x - 1 for x in c1[2]]
    c2[2] = [x - 1 for x in c2[2]]
    tools.cxPartialyMatched(c1[2], c2[2])
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


def custom_mutate(ind, mutpb_s0=1.0, mutpb_s1=1.0, mutpb_s2=1.0):
    """
    Per-chromosome mutation with independent rates.

    Each of the three parts has its own probability of being mutated this
    call. Returns (ind, mutated_flag) so the caller can invalidate fitness
    only when something actually changed.
    """
    m = ind
    mutated = False

    if random.random() < mutpb_s0:
        m[0] = mutate_pallet_locations(m[0], MAX_PALLET, indpb=0.3)
        mutated = True

    if random.random() < mutpb_s1:
        tools.mutUniformInt(m[1], low=0, up=1, indpb=0.3)
        mutated = True

    if random.random() < mutpb_s2:
        tools.mutShuffleIndexes(m[2], indpb=0.3)
        mutated = True

    return m, mutated


# -- Diversity ---------------------------------------------------
def positional_distance(a, b):
    return sum(x != y for x, y in zip(a, b))


def hamming_distance(a, b):
    return sum(x != y for x, y in zip(a, b))


def individual_distance(ind_a, ind_b):
    """Normalized distance between two individuals [0=identical, 1=max different]."""
    d0 = positional_distance(ind_a[0], ind_b[0]) / len(ind_a[0])
    d1 = hamming_distance(ind_a[1], ind_b[1]) / len(ind_a[1])
    d2 = positional_distance(ind_a[2], ind_b[2]) / len(ind_a[2])
    return (d0 + d1 + d2) / 3.0


def count_duplicates(pop, distance_threshold=0.05):
    """Count near-duplicate pairs (adjacent in fitness-sorted order)."""
    sorted_pop = sorted(pop, key=lambda ind: ind.fitness.values)
    count = 0
    for k in range(len(sorted_pop) - 1):
        if individual_distance(sorted_pop[k], sorted_pop[k + 1]) < distance_threshold:
            count += 1
    return count


def dynamic_protect_frac(n_dupes, pop_size,
                         frac_high=0.8, frac_low=0.2,
                         dupe_ratio_trigger=0.40):
    """Linearly scale protect_top_frac down as duplicates rise."""
    if pop_size <= 0:
        return frac_high
    dupe_ratio = n_dupes / pop_size
    t = min(1.0, dupe_ratio / dupe_ratio_trigger) if dupe_ratio_trigger > 0 else 1.0
    return frac_high + (frac_low - frac_high) * t


def eliminate_duplicates(pop, distance_threshold=0.05, neighbor_threshold=0.15,
                         protect_top_frac=0.4):
    """
    Fitness-aware crowding elimination.

    When a near-duplicate pair is found, pick as target the one in the denser
    region (crowding principle), BUT only consider individuals in the bottom
    `protect_top_frac` of the population. This protects top individuals from
    being replaced while still targeting over-represented genotypes in the
    less-fit portion of the population.
    """
    n = len(pop)
    sorted_indices = sorted(range(n), key=lambda i: pop[i].fitness.values)

    protected_cutoff = int(n * protect_top_frac)
    protected = set(sorted_indices[:protected_cutoff])

    n_mutated = 0
    already_mutated = set()

    def count_neighbors(idx):
        c = 0
        for k in range(n):
            if k == idx:
                continue
            if individual_distance(pop[idx], pop[k]) < neighbor_threshold:
                c += 1
        return c

    for k in range(len(sorted_indices) - 1):
        i = sorted_indices[k]
        j = sorted_indices[k + 1]

        if i in already_mutated or j in already_mutated:
            continue

        dist = individual_distance(pop[i], pop[j])
        if dist < distance_threshold:
            candidates = [idx for idx in (i, j) if idx not in protected]

            if not candidates:
                worst_idx = sorted_indices[-1]
                if worst_idx in already_mutated:
                    continue
                target = worst_idx
            elif len(candidates) == 1:
                target = candidates[0]
            else:
                n_i = count_neighbors(i)
                n_j = count_neighbors(j)
                if n_i > n_j:
                    target = i
                elif n_j > n_i:
                    target = j
                else:
                    target = j  # tie-break: worse fitness

            # Heavy mutation on S[0] and S[1]
            pop[target][0] = mutate_pallet_locations(pop[target][0], MAX_PALLET, indpb=0.5)
            tools.mutUniformInt(pop[target][1], low=0, up=1, indpb=0.5)

            del pop[target].fitness.values
            already_mutated.add(target)
            n_mutated += 1

    return n_mutated


def population_diversity(pop, n_samples=50):
    """Normalized diversity per field (0=identical, 1=maximally diverse)."""
    n = len(pop)
    if n < 2:
        return 0.0, 0.0, 0.0
    pairs = [random.sample(range(n), 2) for _ in range(n_samples)]
    len_s0, len_s1, len_s2 = len(pop[0][0]), len(pop[0][1]), len(pop[0][2])
    div_s0 = sum(positional_distance(pop[i][0], pop[j][0]) for i, j in pairs) / (n_samples * len_s0)
    div_s1 = sum(hamming_distance(pop[i][1], pop[j][1]) for i, j in pairs) / (n_samples * len_s1)
    div_s2 = sum(positional_distance(pop[i][2], pop[j][2]) for i, j in pairs) / (n_samples * len_s2)
    return div_s0, div_s1, div_s2


# -- Toolbox (registered once) -----------------------------------
toolbox = base.Toolbox()
toolbox.register("individual", make_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate)
toolbox.register("mate", custom_crossover)
# Note: custom_mutate is called directly in the main loop (not via toolbox.mutate)
# because it takes three per-chromosome rates and returns (ind, mutated_flag).
toolbox.register("select", tools.selTournament, tournsize=2)


# -- Main loop ------------------------------------------------
def run_ga(instance_path="instances/data_1.txt",
           pop_size=200, n_gen=300, cxpb=0.9,
           mutpb_s0=0.10, mutpb_s1=0.20, mutpb_s2=0.05,
           mut_switch_frac=0.3,
           seed=None, verbose=True):
    """
    GA with dynamic fitness-aware crowding and per-chromosome mutation rates.

    Parameters
    ----------
    instance_path : str
        Path to the instance file to load via load_environment().
    pop_size, n_gen, cxpb : standard GA knobs.
    mutpb_s0 : probability of mutating S[0] (pallet locations) per individual.
    mutpb_s1 : probability of mutating S[1] (robot assignment) per individual.
    mutpb_s2 : probability of mutating S[2] (pickup sequence) per individual.
    mut_switch_frac : fraction of n_gen after which mutpb_s0/s1 are overridden
                      to 0.05 (late-stage schedule).
    seed : int or None — reproducibility.
    verbose : if True, print per-gen stats + show final plot; if False, silent
              (use verbose=False inside batch runners).

    Returns
    -------
    best_valid : creator.Individual or None
    pop : list (final population)
    history : dict with per-generation metrics (same keys as baseline GA.py
              plus "eliminated" and "protect" for diagnostics)
    """
    if seed is not None:
        random.seed(seed)

    # Load environment into the module-level globals.
    global products, robot_starting_pos, orders
    global MAX_PALLET, NUM_PRODUCTS, NUM_ORDERS

    products, robot_starting_pos, orders = load_environment(instance_path)
    MAX_PALLET = robot_starting_pos[1] - 1
    NUM_PRODUCTS = len(products)
    NUM_ORDERS = orders

    gen_min = []
    gen_avg = []
    gen_div_s0 = []
    gen_div_s1 = []
    gen_div_s2 = []
    gen_duplicates = []
    gen_valid_count = []
    gen_eliminated = []
    gen_protect = []
    best_valid = None

    pop = toolbox.population(n=pop_size)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min_makespan", lambda x: min(v[1] for v in x if v[0] == 0) if any(v[0] == 0 for v in x) else float('inf'))
    stats.register("avg_makespan", lambda x: round(sum(v[1] for v in x) / len(x), 2))
    stats.register("valid_count",  lambda x: sum(1 for v in x if v[0] == 0))

    # --- Initial evaluation ---
    for ind in pop:
        ind.fitness.values = toolbox.evaluate(ind)

    if verbose:
        print(f"{'Gen':>4} | {'Valid':>6} | {'Best':>10} | {'Avg':>10} | {'Div S0':>6} | {'Div S1':>6} | {'Div S2':>6} | {'Dupes':>5} | {'Elim':>5} | {'Prot':>5}")
        print("-" * 105)

    for gen in range(1, n_gen + 1):
        # --- 1. Selection ---
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(copy.deepcopy, offspring))

        # --- 2. Crossover ---
        for i in range(0, len(offspring) - 1, 2):
            if random.random() < cxpb:
                offspring[i], offspring[i + 1] = toolbox.mate(offspring[i], offspring[i + 1])
                del offspring[i].fitness.values
                del offspring[i + 1].fitness.values

        # --- 3. Mutation with independent per-chromosome rates ---
        # Optional: increase mutation rates in later generations
        if gen > mut_switch_frac * n_gen:
            mutpb_s0 = 0.05
            mutpb_s1 = 0.05
        for ind in offspring:
            _, mutated = custom_mutate(ind, mutpb_s0, mutpb_s1, mutpb_s2)
            if mutated:
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

        # --- 8. Duplicate elimination with dynamic protection ---
        pre_dupes = count_duplicates(pop, distance_threshold=0.05)
        current_protect = dynamic_protect_frac(pre_dupes, len(pop),
                                               frac_high=0.6, frac_low=0.2,
                                               dupe_ratio_trigger=0.40)
        n_eliminated = eliminate_duplicates(pop,
                                            distance_threshold=0.05,
                                            protect_top_frac=current_protect)

        # Re-evaluate anything touched by duplicate elimination
        for ind in pop:
            if not ind.fitness.valid:
                ind.fitness.values = toolbox.evaluate(ind)

        # --- Stats ---
        record = stats.compile(pop)
        gen_min.append(record['min_makespan'])
        gen_avg.append(record['avg_makespan'])
        gen_valid_count.append(record['valid_count'])

        div_s0, div_s1, div_s2 = population_diversity(pop)
        n_dupes = count_duplicates(pop)
        gen_div_s0.append(div_s0)
        gen_div_s1.append(div_s1)
        gen_div_s2.append(div_s2)
        gen_duplicates.append(n_dupes)
        gen_eliminated.append(n_eliminated)
        gen_protect.append(current_protect)

        if verbose:
            print(f"{gen:>4} | {record['valid_count']:>6} | {record['min_makespan']:>10.4f} | {record['avg_makespan']:>10.4f} | {div_s0:>6.3f} | {div_s1:>6.3f} | {div_s2:>6.3f} | {n_dupes:>5} | {n_eliminated:>5} | {current_protect:>5.2f}")

    # --- Final result ---
    if verbose:
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
        plt.title('GA convergence (crowding + fitness-aware)')
        plt.legend()
        plt.grid(True)
        plt.show()

    history = {
        "min": gen_min,
        "avg": gen_avg,
        "div_s0": gen_div_s0,
        "div_s1": gen_div_s1,
        "div_s2": gen_div_s2,
        "duplicates": gen_duplicates,
        "valid_count": gen_valid_count,
        "eliminated": gen_eliminated,
        "protect": gen_protect,
    }

    return best_valid, pop, history


# -- CLI entry point -----------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GA with dynamic fitness-aware crowding."
    )
    parser.add_argument("--instance", type=str, default="instances/data_1.txt",     # instances/data_1.txt | dataset/instance_l20_n30_1.txt
                        help="Path to the instance file (default: instances/data_1.txt)")
    parser.add_argument("--pop", type=int, default=200,
                        help="Population size (default: 200)")
    parser.add_argument("--gens", type=int, default=300,
                        help="Number of generations (default: 300)")
    parser.add_argument("--cxpb", type=float, default=0.9,
                        help="Crossover probability (default: 0.9)")
    parser.add_argument("--mutpb-s0", type=float, default=0.40,
                        help="Mutation rate for S[0] pallet locations (default: 0.10)")
    parser.add_argument("--mutpb-s1", type=float, default=0.40,
                        help="Mutation rate for S[1] robot assignment (default: 0.20)")
    parser.add_argument("--mutpb-s2", type=float, default=0.05,
                        help="Mutation rate for S[2] pickup sequence (default: 0.05)")
    parser.add_argument("--mut-switch-frac", type=float, default=0.0,
                        help="Fraction of n_gen after which the late-stage "
                             "mutation schedule kicks in (default: 0.3)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed (default: None = non-deterministic)")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress per-generation output and the final plot.")
    args = parser.parse_args()

    start = time.perf_counter()
    best, final_pop, history = run_ga(
        instance_path=args.instance,
        pop_size=args.pop,
        n_gen=args.gens,
        cxpb=args.cxpb,
        mutpb_s0=args.mutpb_s0,
        mutpb_s1=args.mutpb_s1,
        mutpb_s2=args.mutpb_s2,
        mut_switch_frac=args.mut_switch_frac,
        seed=args.seed,
        verbose=not args.quiet,
    )
    elapsed = time.perf_counter() - start
    print(f"\nTotal execution time: {elapsed:.2f} seconds")