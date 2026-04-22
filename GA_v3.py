import random
import copy
import time
import matplotlib.pyplot as plt
from deap import base, creator, tools


from simulation_v2 import load_environment
from simulation_v3 import Simulator as SimulatorV3  

start = time.perf_counter()
products, robot_starting_pos, orders = load_environment("dataset/instance_l20_n30_1.txt") # instances/data_1.txt | data.txt |
end_time = time.perf_counter()                                                # dataset/instance_l20_n30_1.txt

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

def custom_crossover(ind1, ind2, cxpb_s0=0.9, cxpb_s1=0.9, cxpb_s2=0.9):
    c1, c2 = ind1, ind2

    # S[0] - PMX crossover
    if random.random() < cxpb_s0:
        c1[0], c2[0] = cx_s0_pmx(c1[0], c2[0])

    # S[1] - Two-point crossover
    if random.random() < cxpb_s1:
        tools.cxTwoPoint(c1[1], c2[1])

    # S[2] - Ordered crossover
    if random.random() < cxpb_s2:
        c1[2] = [x - 1 for x in c1[2]]
        c2[2] = [x - 1 for x in c2[2]]
        tools.cxPartialyMatched(c1[2], c2[2])
        c1[2] = [x + 1 for x in c1[2]]
        c2[2] = [x + 1 for x in c2[2]]
        #tools.cxPartialyMatched(c1[2], c2[2])  # apply ordered crossover

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

def custom_mutate(ind, mutpb_s0=1.0, mutpb_s1=1.0, mutpb_s2=1.0,
                  indpb_s0=0.2, indpb_s1=0.3, indpb_s2=0.1):
    """
    Per-field mutation. Each field is mutated INDEPENDENTLY with its own
    probability, instead of v2's all-or-nothing approach where triggering
    mutation at all meant touching all three fields.

    Args:
        mutpb_s0/s1/s2: probability that this field is mutated at all
        indpb_s0/s1/s2: per-gene mutation rate INSIDE each field

    Note: v2 used indpb_s1=0.7 which was very aggressive. Lowered default
    to 0.3 to prevent overwhelming structural damage when S[1] does mutate.
    """
    m = ind

    if random.random() < mutpb_s0:
        m[0] = mutate_pallet_locations(m[0], MAX_PALLET, indpb=indpb_s0)

    if random.random() < mutpb_s1:
        tools.mutUniformInt(m[1], low=0, up=1, indpb=indpb_s1)

    if random.random() < mutpb_s2:
        tools.mutShuffleIndexes(m[2], indpb=indpb_s2)

    return (m,)

# -- Diversity ------------------------------------------------
def positional_distance(a, b):
    """Count of positions where two sequences differ."""
    return sum(x != y for x, y in zip(a, b))

def hamming_distance(a, b):
    """Hamming distance for binary lists."""
    return sum(x != y for x, y in zip(a, b))

def individual_distance(ind_a, ind_b):
    """
    Normalized distance between two individuals across all fields.
    Returns value in [0, 1] where 0 = identical, 1 = maximally different.
    """
    len_s0 = len(ind_a[0])
    len_s1 = len(ind_a[1])
    len_s2 = len(ind_a[2])

    d0 = positional_distance(ind_a[0], ind_b[0]) / len_s0
    d1 = hamming_distance(ind_a[1], ind_b[1]) / len_s1
    d2 = positional_distance(ind_a[2], ind_b[2]) / len_s2

    return (d0 + d1 + d2) / 3.0

def population_diversity(pop, n_samples=50, baseline=None):
    """
    Compute diversity for each field, rescaled against a random-population baseline.

    CRITICAL FIX from v2: The previous version normalized by array length, which
    is wrong for binary fields. Two random binary arrays have expected Hamming
    distance = 0.5 per bit, so that normalization caps out at ~0.5, never 1.0.
    The adaptive rates treated 0.5 as "healthy middle" when it actually means
    "indistinguishable from random search".

    Fix: We rescale each field's raw diversity by the diversity of a REFERENCE
    random population. Result:
      - 1.0 = as diverse as a fresh random population (no convergence)
      - 0.5 = population has partially converged
      - 0.0 = full collapse (all individuals identical in this field)

    Args:
        pop:      current population
        n_samples: number of random pairs to sample
        baseline: (base_s0, base_s1, base_s2) raw diversities from a random
                  reference population. If None, returns UNSCALED raw values
                  (used when computing the baseline itself).
    """
    n = len(pop)
    if n < 2:
        return 0.0, 0.0, 0.0

    pairs = [random.sample(range(n), 2) for _ in range(n_samples)]

    len_s0 = len(pop[0][0])
    len_s1 = len(pop[0][1])
    len_s2 = len(pop[0][2])

    raw_s0 = sum(positional_distance(pop[i][0], pop[j][0]) for i, j in pairs) / (n_samples * len_s0)
    raw_s1 = sum(hamming_distance(pop[i][1], pop[j][1]) for i, j in pairs) / (n_samples * len_s1)
    raw_s2 = sum(positional_distance(pop[i][2], pop[j][2]) for i, j in pairs) / (n_samples * len_s2)

    if baseline is None:
        return raw_s0, raw_s1, raw_s2

    b0, b1, b2 = baseline
    # Avoid division by zero; if baseline is 0 the field is fundamentally constant
    # and diversity tracking is meaningless.
    div_s0 = min(1.0, raw_s0 / b0) if b0 > 1e-9 else 0.0
    div_s1 = min(1.0, raw_s1 / b1) if b1 > 1e-9 else 0.0
    div_s2 = min(1.0, raw_s2 / b2) if b2 > 1e-9 else 0.0

    return div_s0, div_s1, div_s2


def compute_diversity_baseline(pop_size, n_reference_pops=5, n_samples=200):
    """
    Estimate the raw diversity of a fresh random population per field.
    This is what "maximum achievable diversity" looks like in practice.

    We average over multiple random populations to get a stable baseline.
    Called ONCE at the start of run_ga, before the main loop.
    """
    b0_list = []
    b1_list = []
    b2_list = []
    for _ in range(n_reference_pops):
        ref_pop = [make_individual() for _ in range(pop_size)]
        r0, r1, r2 = population_diversity(ref_pop, n_samples=n_samples, baseline=None)
        b0_list.append(r0)
        b1_list.append(r1)
        b2_list.append(r2)

    baseline = (sum(b0_list) / len(b0_list),
                sum(b1_list) / len(b1_list),
                sum(b2_list) / len(b2_list))
    return baseline

def eliminate_duplicates(pop, distance_threshold=0.05, worst_prob=0.3):
    """
    Find near-duplicate individuals and force-mutate one to restore diversity.

    With probability (1 - worst_prob): mutate the worse of the duplicate pair
        → directly breaks up the clone
    With probability worst_prob: mutate the worst-fitness individual in the population
        → spreads diversity more broadly, avoids always punishing duplicates

    Returns the number of individuals mutated.
    """
    sorted_indices = sorted(range(len(pop)), key=lambda i: pop[i].fitness.values)
    n_mutated = 0
    already_mutated = set()

    # Pre-find worst individual (skip if already mutated)
    worst_idx = sorted_indices[-1]

    for k in range(len(sorted_indices) - 1):
        i = sorted_indices[k]
        j = sorted_indices[k + 1]

        if i in already_mutated or j in already_mutated:
            continue

        dist = individual_distance(pop[i], pop[j])
        if dist < distance_threshold:
            # Choose target: duplicate or worst individual
            if random.random() < worst_prob and worst_idx not in already_mutated:
                target = worst_idx
            else:
                target = j  # worse of the pair (sorted ascending)

            # Heavy mutation
            pop[target][0] = mutate_pallet_locations(pop[target][0], MAX_PALLET, indpb=0.5)
            tools.mutUniformInt(pop[target][1], low=0, up=1, indpb=0.5)
            
            del pop[target].fitness.values
            already_mutated.add(target)
            n_mutated += 1

    return n_mutated

def adaptive_rates(div_s0, div_s1, div_s2,
                   cx_range=(0.5, 0.9), mut_range=(0.05, 0.30),
                   div_high=0.75):
    """
    Adaptive rates v3: each field gets its OWN crossover and mutation rate.

    Change vs v2:
      - Returns mutpb_s0, mutpb_s1, mutpb_s2 separately (v2 returned a single
        averaged mutpb, which washed out per-field signals).
      - Mutation range upped from (0.02, 0.25) to (0.05, 0.60) because with
        per-field control, low-diversity fields need aggressive rescue.

    Logic per field:
      - In healthy range (0 -> div_high): crossover scales with diversity,
        mutation scales inversely.
      - Above div_high (destructive):    reduce crossover, keep mutation low.
    """
    cx_lo, cx_hi = cx_range
    mut_lo, mut_hi = mut_range

    def cx_for_field(div):
        if div > div_high:
            return cx_lo
        t = div / div_high
        return cx_lo + (cx_hi - cx_lo) * t

    def mut_for_field(div):
        if div > div_high:
            return mut_lo
        t = div / div_high
        return mut_hi - (mut_hi - mut_lo) * t

    cxpb_s0 = cx_for_field(div_s0)
    cxpb_s1 = cx_for_field(div_s1)
    cxpb_s2 = cx_for_field(div_s2)

    mutpb_s0 = mut_for_field(div_s0)
    mutpb_s1 = mut_for_field(div_s1)
    mutpb_s2 = mut_for_field(div_s2)

    return cxpb_s0, cxpb_s1, cxpb_s2, mutpb_s0, mutpb_s1, mutpb_s2


def reactive_field_boost(pop, div_s0, div_s1, div_s2,
                         collapse_threshold=0.10, boost_fraction=0.30):
    """
    Emergency diversity injection when a specific field collapses.

    When any field's diversity drops below collapse_threshold, heavily mutate
    that field in `boost_fraction` of the population (targeting worst
    fitness individuals to preserve good solutions).

    This is different from eliminate_duplicates because:
      - It targets a SPECIFIC field that has collapsed
      - It applies aggressive (indpb=0.5) mutation, not standard rates
      - It triggers on global field diversity, not pairwise similarity

    Returns (n_boosted_s0, n_boosted_s1, n_boosted_s2).
    """
    n_to_boost = max(1, int(len(pop) * boost_fraction))

    # Target the WORST individuals (sort descending so we pick highest fitness values)
    worst_indices = sorted(range(len(pop)),
                          key=lambda i: pop[i].fitness.values,
                          reverse=True)[:n_to_boost]

    n_s0 = n_s1 = n_s2 = 0

    for idx in worst_indices:
        if div_s0 < collapse_threshold:
            pop[idx][0] = mutate_pallet_locations(pop[idx][0], MAX_PALLET, indpb=0.5)
            n_s0 += 1
        if div_s1 < collapse_threshold:
            tools.mutUniformInt(pop[idx][1], low=0, up=1, indpb=0.5)
            n_s1 += 1
        if div_s2 < collapse_threshold:
            tools.mutShuffleIndexes(pop[idx][2], indpb=0.3)
            n_s2 += 1

        if n_s0 + n_s1 + n_s2 > 0:
            del pop[idx].fitness.values

    return n_s0, n_s1, n_s2


toolbox = base.Toolbox()
toolbox.register("individual", make_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate",   evaluate)
toolbox.register("mate",       custom_crossover)
toolbox.register("mutate",     custom_mutate)
toolbox.register("select",     tools.selTournament, tournsize=2)

# -- Main loop ------------------------------------------------
def run_ga(pop_size=300, n_gen=500, adaptive=True,
           cxpb_s0=0.9, cxpb_s1=0.9, cxpb_s2=0.9,
           mutpb_s0=0.3, mutpb_s1=0.3, mutpb_s2=0.3,
           seed=None,
           tournsize=2,
           use_reactive_boost=True,
           use_duplicate_elim=True,
           collapse_threshold=0.10,
           boost_fraction=0.30):
    """
    v3 improvements over v2:
      - Decoupled per-field mutation rates (not one global mutpb)
      - Per-field adaptive rates (adaptive_rates returns 6 values now)
      - Reactive boost when a specific field's diversity collapses
      - Tunable tournament size for selection pressure
      - Ablation flags to test which improvement actually helps

    Args:
        adaptive:            if True, use diversity-driven rates. If False, use fixed rates.
        tournsize:           selection pressure (lower = more diverse)
        use_reactive_boost:  toggle field-specific emergency mutation
        use_duplicate_elim:  toggle near-clone removal (from v2)
        collapse_threshold:  trigger boost when field diversity drops below this
        boost_fraction:      fraction of worst individuals to heavily mutate on collapse
    """
    if seed is not None:
        random.seed(seed)

    # Re-register selection with the chosen tournsize (v2 hard-coded 2)
    toolbox.unregister("select")
    toolbox.register("select", tools.selTournament, tournsize=tournsize)

    gen_min = []
    gen_avg = []
    gen_diversity = []
    gen_rates = []
    gen_duplicates = []
    gen_boosts = []  # (n_s0, n_s1, n_s2) boost counts per generation
    best_valid = None

    pop = toolbox.population(n=pop_size)

    # --- Compute diversity baseline from random reference populations ---
    # This gives us the "what does maximum diversity look like" reference for THIS problem,
    # which matters because binary fields cap at raw 0.5, not 1.0.
    diversity_baseline = compute_diversity_baseline(pop_size)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min_makespan",
                   lambda x: min(v[1] for v in x if v[0] == 0) if any(v[0] == 0 for v in x) else float('inf'))
    stats.register("avg_makespan", lambda x: round(sum(v[1] for v in x) / len(x), 2))
    stats.register("valid_count",  lambda x: sum(1 for v in x if v[0] == 0))

    # --- Initial evaluation ---
    for ind in pop:
        ind.fitness.values = toolbox.evaluate(ind)

    print(f"=== GA v3 | adaptive={adaptive} | tourn={tournsize} | "
          f"boost={use_reactive_boost} | dupelim={use_duplicate_elim} ===")
    print(f"Diversity baseline (raw, from random reference pops): "
          f"S0={diversity_baseline[0]:.3f}, S1={diversity_baseline[1]:.3f}, "
          f"S2={diversity_baseline[2]:.3f}")
    print(f"(All displayed diversities are rescaled: 1.0 = as diverse as random init)")
    print(f"{'Gen':>4} | {'Val':>4} | {'Best':>9} | {'Avg':>9} | "
          f"{'D0':>4} | {'D1':>4} | {'D2':>4} | "
          f"{'Cx0':>4} | {'Cx1':>4} | {'Cx2':>4} | "
          f"{'M0':>4} | {'M1':>4} | {'M2':>4} | "
          f"{'Dup':>3} | {'B0':>3} | {'B1':>3} | {'B2':>3}")
    print("-" * 150)

    for gen in range(1, n_gen + 1):

        # --- 0. Compute diversity & adapt rates ---
        div_s0, div_s1, div_s2 = population_diversity(pop, baseline=diversity_baseline)
        gen_diversity.append((div_s0, div_s1, div_s2))

        if adaptive:
            (cxpb_s0, cxpb_s1, cxpb_s2,
             mutpb_s0, mutpb_s1, mutpb_s2) = adaptive_rates(div_s0, div_s1, div_s2)

        gen_rates.append((cxpb_s0, cxpb_s1, cxpb_s2, mutpb_s0, mutpb_s1, mutpb_s2))

        # --- 1. Selection ---
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(copy.deepcopy, offspring))

        # --- 2. Crossover ---
        for i in range(0, len(offspring) - 1, 2):
            offspring[i], offspring[i+1] = custom_crossover(
                offspring[i], offspring[i+1], cxpb_s0, cxpb_s1, cxpb_s2)
            del offspring[i].fitness.values
            del offspring[i+1].fitness.values

        # --- 3. Mutation (decoupled per field) ---
        for ind in offspring:
            (ind,) = custom_mutate(ind,
                                   mutpb_s0=mutpb_s0,
                                   mutpb_s1=mutpb_s1,
                                   mutpb_s2=mutpb_s2)
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

        # --- 8. Reactive boost (v3 NEW) ---
        n_b0 = n_b1 = n_b2 = 0
        if use_reactive_boost:
            # Recompute diversity in case elitism/crossover shifted things
            cur_d0, cur_d1, cur_d2 = population_diversity(pop, baseline=diversity_baseline)
            n_b0, n_b1, n_b2 = reactive_field_boost(
                pop, cur_d0, cur_d1, cur_d2,
                collapse_threshold=collapse_threshold,
                boost_fraction=boost_fraction)
        gen_boosts.append((n_b0, n_b1, n_b2))

        # --- 9. Duplicate elimination (from v2) ---
        n_dupes = 0
        if use_duplicate_elim:
            n_dupes = eliminate_duplicates(pop, distance_threshold=0.05)
        gen_duplicates.append(n_dupes)

        # --- Re-evaluate anything touched by boost or dupe-elim ---
        for ind in pop:
            if not ind.fitness.valid:
                ind.fitness.values = toolbox.evaluate(ind)

        # --- Stats ---
        record = stats.compile(pop)
        gen_min.append(record['min_makespan'])
        gen_avg.append(record['avg_makespan'])

        print(f"{gen:>4} | {record['valid_count']:>4} | "
              f"{record['min_makespan']:>9.2f} | {record['avg_makespan']:>9.2f} | "
              f"{div_s0:>4.2f} | {div_s1:>4.2f} | {div_s2:>4.2f} | "
              f"{cxpb_s0:>4.2f} | {cxpb_s1:>4.2f} | {cxpb_s2:>4.2f} | "
              f"{mutpb_s0:>4.2f} | {mutpb_s1:>4.2f} | {mutpb_s2:>4.2f} | "
              f"{n_dupes:>3} | {n_b0:>3} | {n_b1:>3} | {n_b2:>3}")

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
    fig, axes = plt.subplots(4, 1, figsize=(11, 13), sharex=True)
    ax1, ax2, ax3, ax4 = axes

    # Fitness plot
    valid_gens = [(i, v) for i, v in enumerate(gen_min) if v != float('inf')]
    if valid_gens:
        xs, ys = zip(*valid_gens)
        ax1.plot(xs, ys, label='Best valid fitness')
    ax1.plot(gen_avg, label='Average fitness', alpha=0.7)
    ax1.set_ylabel('Fitness')
    ax1.set_title('GA v3 Convergence')
    ax1.legend()
    ax1.grid(True)

    # Diversity plot
    d0, d1, d2 = zip(*gen_diversity)
    ax2.plot(d0, label='S0 (pallets)')
    ax2.plot(d1, label='S1 (robots)')
    ax2.plot(d2, label='S2 (sequence)')
    ax2.axhline(y=collapse_threshold, linestyle='--', color='red', alpha=0.4,
                label=f'Collapse threshold ({collapse_threshold})')
    ax2.set_ylabel('Diversity (0-1)')
    ax2.set_title('Population Diversity')
    ax2.legend()
    ax2.grid(True)

    # Per-field mutation rates (shows WHICH field is being rescued)
    _, _, _, m0s, m1s, m2s = zip(*gen_rates)
    ax3.plot(m0s, label='MutP S0')
    ax3.plot(m1s, label='MutP S1')
    ax3.plot(m2s, label='MutP S2')
    ax3.set_ylabel('Mutation probability')
    ax3.set_title('Per-Field Mutation Rates')
    ax3.legend()
    ax3.grid(True)

    # Boost events per field
    b0, b1, b2 = zip(*gen_boosts) if gen_boosts else ([], [], [])
    ax4.plot(b0, label='Boosts S0', alpha=0.7)
    ax4.plot(b1, label='Boosts S1', alpha=0.7)
    ax4.plot(b2, label='Boosts S2', alpha=0.7)
    ax4.plot(gen_duplicates, label='Dupes eliminated', color='black', alpha=0.5, linestyle=':')
    ax4.set_xlabel('Generation')
    ax4.set_ylabel('Count')
    ax4.set_title('Reactive Boost & Duplicate-Elim Events')
    ax4.legend()
    ax4.grid(True)

    plt.tight_layout()
    plt.show()

    return best_valid, pop, {
        "min": gen_min,
        "avg": gen_avg,
        "diversity": gen_diversity,
        "rates": gen_rates,
        "duplicates": gen_duplicates,
        "boosts": gen_boosts,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="GA v3: decoupled mutation + reactive boost.")
    parser.add_argument("--seed",      type=int, default=None)
    parser.add_argument("--pop",       type=int, default=200)
    parser.add_argument("--gens",      type=int, default=300)
    parser.add_argument("--tournsize", type=int, default=2)
    parser.add_argument("--no-boost",  action="store_true",
                        help="Disable reactive field boost (for ablation).")
    parser.add_argument("--no-dupelim", action="store_true",
                        help="Disable duplicate elimination (for ablation).")
    parser.add_argument("--no-adaptive", action="store_true",
                        help="Use fixed rates instead of adaptive.")
    args = parser.parse_args()

    best, final_pop, history = run_ga(
        pop_size=args.pop,
        n_gen=args.gens,
        seed=args.seed,
        tournsize=args.tournsize,
        adaptive=not args.no_adaptive,
        use_reactive_boost=not args.no_boost,
        use_duplicate_elim=not args.no_dupelim,
    )
    print(f"\nTotal execution time: {end_time - start:.2f} seconds")