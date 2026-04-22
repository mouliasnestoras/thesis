import random
import os
from dataclasses import dataclass
from typing import List


# ------------------ Parameter classes ------------------ #

@dataclass
class NormalIntParam:
    mean: float
    std: float
    min_val: int
    max_val: int

    def sample(self, rng: random.Random) -> int:
        """
        Sample from a normal distribution N(mean, std), round to int,
        then clip to [min_val, max_val].
        """
        val = self.mean + self.std * rng.gauss(0, 1)
        iv = int(round(val))
        if iv < self.min_val:
            iv = self.min_val
        if iv > self.max_val:
            iv = self.max_val
        return iv


# ------------------ Problem instance ------------------ #

@dataclass
class ProblemInstance:
    num_locations: int
    belts: List[int]               # positions of belts along the line (2..N-1)
    orders: List[List[int]]        # each order is a sequence of product types (1..num_belts)

    def to_text(self) -> str:
        """Return the instance as a formatted string, matching the desired template."""
        lines = []
        lines.append(f"Number of locations:   {self.num_locations}")
        belt_labels = [f"P{i+1}" for i in range(len(self.belts))]
        belts_header = ", ".join(belt_labels)
        belts_positions = ", ".join(str(b) for b in self.belts)
        lines.append(f"Belts {belts_header}:   {belts_positions}")
        for i, order in enumerate(self.orders, start=1):
            order_str = ", ".join(str(p) for p in order)
            lines.append(f"Order O{i}:   {order_str}")
        return "\n".join(lines)


# ------------------ Normal-distribution parameter factories ------------------ #

def get_num_belts_param(l: int) -> NormalIntParam:
    """Normal distribution for the number of belts (product types), depending on l."""
    if l <= 10:
        return NormalIntParam(mean=4.0, std=0.8, min_val=3, max_val=min(6, l - 2))
    elif l <= 20:
        return NormalIntParam(mean=6.0, std=1.0, min_val=4, max_val=8)
    else:
        return NormalIntParam(mean=6.0, std=1.5, min_val=3, max_val=min(10, l - 2))


def get_num_orders_param(l: int, n: int, num_belts: int) -> NormalIntParam:
    """Normal distribution for the number of orders, respecting hard constraints."""
    # hard cap: must fit in remaining pallet positions AND not exceed n
    max_orders = max(1, min(l - num_belts, n))
    if l <= 10:
        mean = min(4.0, float(max_orders))
        std = 1.5
    elif l <= 20:
        mean = min(8.0, float(max_orders))
        std = 2.5
    else:
        mean = min(12.0, float(max_orders))
        std = 3.0
    min_orders = min(2, max_orders)
    return NormalIntParam(mean=mean, std=std, min_val=min_orders, max_val=max_orders)


# ------------------ Generator helpers ------------------ #

def sample_distinct_positions(num_locations: int, num_belts: int, rng: random.Random) -> List[int]:
    """Choose num_belts distinct internal positions (not the two extremes), sorted."""
    if num_locations <= 2:
        raise ValueError("num_locations must be > 2 to have internal positions.")
    internal_positions = list(range(1, num_locations - 1))
    if num_belts > len(internal_positions):
        raise ValueError("num_belts cannot exceed number of internal positions.")
    rng.shuffle(internal_positions)
    return sorted(internal_positions[:num_belts])


def partition_jobs_normal(n: int, num_orders: int, rng: random.Random,
                          std_ratio: float = 0.35) -> List[int]:
    """
    Partition n jobs into num_orders parts using a Normal distribution.
    Each part is sampled from N(n/num_orders, std_ratio * n/num_orders),
    clipped to >= 1, then the sizes are adjusted so that the total equals n exactly.
    """
    if num_orders > n:
        raise ValueError("Cannot have more orders than jobs.")

    mean = n / num_orders
    std = max(0.5, std_ratio * mean)  # avoid degenerate std

    # Initial normal samples (each >= 1)
    sizes = []
    for _ in range(num_orders):
        val = rng.gauss(mean, std)
        iv = max(1, int(round(val)))
        sizes.append(iv)

    # Adjust the total to be exactly n, while keeping every size >= 1
    diff = n - sum(sizes)
    while diff != 0:
        if diff > 0:
            idx = rng.randint(0, num_orders - 1)
            sizes[idx] += 1
            diff -= 1
        else:
            candidates = [i for i, s in enumerate(sizes) if s > 1]
            if not candidates:
                break
            idx = rng.choice(candidates)
            sizes[idx] -= 1
            diff += 1

    return sizes


def generate_instance_fixed(l: int, n: int, rng: random.Random) -> ProblemInstance:
    """
    Generate a TRPASP instance with fixed l rail locations and n total jobs.
    All random quantities (num_belts, num_orders, order sizes) are drawn
    from Normal distributions, then clipped to feasible ranges.
    """
    num_locations = l

    # --- Number of belts ~ Normal ---
    num_belts_param = get_num_belts_param(l)
    num_belts = num_belts_param.sample(rng)
    max_internal = l - 2
    if num_belts > max_internal:
        num_belts = max_internal

    # --- Belt positions (uniform over internal rail positions) ---
    belts = sample_distinct_positions(l, num_belts, rng)

    # --- Number of orders ~ Normal ---
    num_orders_param = get_num_orders_param(l, n, num_belts)
    num_orders = num_orders_param.sample(rng)

    # --- Order sizes ~ Normal, corrected to sum = n ---
    order_sizes = partition_jobs_normal(n, num_orders, rng)

    # --- Product types per job: uniform over {1..num_belts}
    # (categorical label; normal distribution is not meaningful here) ---
    orders: List[List[int]] = []
    for size in order_sizes:
        order = [rng.randint(1, num_belts) for _ in range(size)]
        orders.append(order)

    return ProblemInstance(num_locations=num_locations, belts=belts, orders=orders)


# ------------------ File I/O helpers ------------------ #

def write_instance_to_file(instance: ProblemInstance, filepath: str) -> None:
    """Write a single instance to a .txt file."""
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(instance.to_text())


def clear_all(directory: str = "dataset") -> None:
    """Delete all .txt files from the given directory."""
    if not os.path.isdir(directory):
        return
    for name in os.listdir(directory):
        if name.lower().endswith(".txt"):
            full_path = os.path.join(directory, name)
            if os.path.isfile(full_path):
                os.remove(full_path)


# ------------------ Generate the test suite ------------------ #

def main():
    # Fixed seed for reproducibility (set to None for fully random)
    rng = random.Random()

    output_dir = "dataset"
    os.makedirs(output_dir, exist_ok=True)

    # --- Test suite parameters (matching the paper) ---
    l_values = [10, 20]                 # rail locations
    n_values = [20, 60, 100]            # total number of jobs
    instances_per_pair = 5              # 5 instances per (l, n) pair -> 30 total

    total_count = 0
    for l in l_values:
        for n in n_values:
            for k in range(1, instances_per_pair + 1):
                inst = generate_instance_fixed(l, n, rng)
                filename = f"instance_l{l}_n{n}_{k}.txt"
                filepath = os.path.join(output_dir, filename)
                write_instance_to_file(inst, filepath)
                total_count += 1
                print(f"Wrote {filepath}  "
                      f"(orders={len(inst.orders)}, belts={len(inst.belts)}, "
                      f"total_jobs={sum(len(o) for o in inst.orders)})")

    print(f"\nTotal instances generated: {total_count}")


if __name__ == "__main__":
    main()
    # Uncomment the following line to delete all generated instances:
    #clear_all()