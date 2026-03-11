import random
import copy
from deap import base, creator, tools

from simulation_v2 import Simulator, load_environment

products, robot_starting_pos = load_environment("data.txt")

def objective(s0, s1, s2):
    solution = [s0, s1, s2]
    
    sim = Simulator(solution=solution, starting_pos=robot_starting_pos)
    sim.schedule_jobs(products)
    fitness = sim.simulation()
    
    return fitness

# ── DEAP setup ───────────────────────────────────────────────────────────────
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

MAX_PALLET   = 12
NUM_ROBOTS   = 2
NUM_PRODUCTS = 12

def make_individual():
    s0 = [random.randint(1, MAX_PALLET) for _ in range(3)]
    s1 = [random.randint(0, NUM_ROBOTS - 1) for _ in range(NUM_PRODUCTS)]
    s2 = random.sample(range(1, NUM_PRODUCTS + 1), NUM_PRODUCTS)
    return creator.Individual([s0, s1, s2])

def evaluate(ind):
    return (objective(ind[0], ind[1], ind[2]),)  # must return tuple

def custom_crossover(ind1, ind2):
    c1, c2 = copy.deepcopy(ind1), copy.deepcopy(ind2)

    tools.cxTwoPoint(c1[0], c2[0])
    tools.cxUniform(c1[1], c2[1], 0.5)

    # cxOrdered requires 0-indexed — shift down, cross, shift back up
    c1[2] = [x - 1 for x in c1[2]]
    c2[2] = [x - 1 for x in c2[2]]
    tools.cxOrdered(c1[2], c2[2])
    c1[2] = [x + 1 for x in c1[2]]
    c2[2] = [x + 1 for x in c2[2]]

    return c1, c2

def custom_mutate(ind):
    m = copy.deepcopy(ind)
    tools.mutUniformInt(m[0], low=1, up=MAX_PALLET, indpb=0.3)
    tools.mutUniformInt(m[1], low=0, up=NUM_ROBOTS - 1, indpb=0.2)
    tools.mutShuffleIndexes(m[2], indpb=0.2)
    return (m,)

toolbox = base.Toolbox()
toolbox.register("individual", make_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate",   evaluate)
toolbox.register("mate",       custom_crossover)
toolbox.register("mutate",     custom_mutate)
toolbox.register("select",     tools.selTournament, tournsize=3)

# ── main loop ────────────────────────────────────────────────────────────────
def run_ga(pop_size=200, n_gen=100, cxpb=0.7, mutpb=0.3):

    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)          # tracks best individual ever seen

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", lambda x: min(v[0] for v in x))
    stats.register("avg", lambda x: round(sum(v[0] for v in x) / len(x), 2))

    # evaluate the initial population (all fitness invalid at this point)
    fitnesses = map(toolbox.evaluate, pop)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    print(f"{'Gen':>4} | {'Min':>10} | {'Avg':>10}")
    print("-" * 32)

    for gen in range(1, n_gen + 1):

        # 1. SELECTION — pick parents
        offspring = toolbox.select(pop, len(pop))
        offspring = [copy.deepcopy(ind) for ind in offspring]

        # 2. CROSSOVER — pair up and mate
        for i in range(0, len(offspring) - 1, 2):
            if random.random() < cxpb:
                offspring[i], offspring[i+1] = toolbox.mate(offspring[i], offspring[i+1])
                # invalidate fitness — these individuals changed
                del offspring[i].fitness.values
                del offspring[i+1].fitness.values

        # 3. MUTATION
        for ind in offspring:
            if random.random() < mutpb:
                (ind,) = toolbox.mutate(ind)
                del ind.fitness.values   # invalidate fitness

        # 4. EVALUATE — only individuals with invalid fitness (new/mutated ones)
        invalid = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid)
        for ind, fit in zip(invalid, fitnesses):
            ind.fitness.values = fit

        # 5. REPLACE population
        pop[:] = offspring

        # 6. UPDATE hall of fame & stats
        hof.update(pop)
        record = stats.compile(pop)
        print(f"{gen:>4} | {record['min']:>10.4f} | {record['avg']:>10.4f}")

    print("\n-- Best Individual --")
    print(f"  S0 (pallets):   {hof[0][0]}")
    print(f"  S1 (robots):    {hof[0][1]}")
    print(f"  S2 (sequence):  {hof[0][2]}")
    print(f"  Fitness:        {hof[0].fitness.values[0]:.4f}")

    return hof[0], pop

best, final_pop = run_ga()
