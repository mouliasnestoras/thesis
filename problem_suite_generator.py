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


@dataclass
class InstanceParams:
    num_locations_param: NormalIntParam
    num_belts_param: NormalIntParam
    num_orders_param: NormalIntParam
    order_length_param: NormalIntParam


# ------------------ Problem instance ------------------ #

@dataclass
class ProblemInstance:
    num_locations: int
    belts: List[int]               # positions of belts along the line (2..N-1)
    orders: List[List[int]]        # each order is a sequence of product types (1..num_belts)

    def to_text(self) -> str:
        """Return the instance as a formatted string, matching the desired template."""
        lines = []

        # Number of locations
        lines.append(f"Number of locations:   {self.num_locations}")

        # Belts line: show product types P1, P2, ... and their locations
        belt_labels = [f"P{i+1}" for i in range(len(self.belts))]
        belts_header = ", ".join(belt_labels)
        belts_positions = ", ".join(str(b) for b in self.belts)
        lines.append(f"Belts {belts_header}:   {belts_positions}")

        # Orders: product types as plain integers (1, 2, 3, ...)
        for i, order in enumerate(self.orders, start=1):
            order_str = ", ".join(str(p) for p in order)
            lines.append(f"Order O{i}:   {order_str}")

        return "\n".join(lines)


# ------------------ Generator functions ------------------ #

def sample_distinct_positions(num_locations: int, num_belts: int, rng: random.Random) -> List[int]:
    """
    Choose num_belts distinct positions from internal positions {2, ..., num_locations-1}
    and return them sorted. Positions 1 and num_locations are forbidden.
    """
    if num_locations <= 2:
        raise ValueError("num_locations must be > 2 to have internal positions.")
    internal_positions = list(range(1, num_locations-1))  # 2..num_locations-1

    if num_belts > len(internal_positions):
        raise ValueError("num_belts cannot exceed number of internal positions.")

    rng.shuffle(internal_positions)
    chosen = sorted(internal_positions[:num_belts])
    return chosen


def generate_instance(params: InstanceParams, rng: random.Random) -> ProblemInstance:
    """
    Generate a single problem instance using the given parameters
    and random number generator.
    """

    # 1. Number of locations
    num_locations = params.num_locations_param.sample(rng)

    # 2. Max number of belts allowed (internal positions only)
    max_internal_positions = max(0, num_locations - 2)
    if max_internal_positions == 0:
        raise ValueError("No internal positions available for belts.")

    # 3. Number of belts (cannot exceed internal positions)
    num_belts = params.num_belts_param.sample(rng)
    if num_belts > max_internal_positions:
        num_belts = max_internal_positions

    # 4. Belt positions along the line (2..num_locations-1)
    belts = sample_distinct_positions(num_locations, num_belts, rng)

    # 5. Number of orders
    num_orders = params.num_orders_param.sample(rng)

    # 6. Orders: each order is a sequence of product types (1..num_belts)
    orders: List[List[int]] = []
    for _ in range(num_orders):
        length = params.order_length_param.sample(rng)
        order = [rng.randint(1, num_belts) for _ in range(length)]
        orders.append(order)

    return ProblemInstance(num_locations=num_locations, belts=belts, orders=orders)


# ------------------ File I/O helpers ------------------ #

def write_instance_to_file(instance: ProblemInstance, filepath: str) -> None:
    """Write a single instance to a .txt file."""
    text = instance.to_text()
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(text)


def clear_all(directory: str = "instances") -> None:
    """
    Delete all .txt files from the given directory.
    If the directory does not exist, nothing happens.
    """
    if not os.path.isdir(directory):
        return
    for name in os.listdir(directory):
        if name.lower().endswith(".txt"):
            full_path = os.path.join(directory, name)
            if os.path.isfile(full_path):
                os.remove(full_path)


# ------------------ Example: generate a suite ------------------ #

def main():
    rng = random.Random()  

    # Output folder (will be created if it does not exist)
    output_dir = "instances"
    os.makedirs(output_dir, exist_ok=True)


    # Choose instance parameters 
    inst_params = InstanceParams(
        num_locations_param=NormalIntParam(mean=10, std=2, min_val=10, max_val=20),
        num_belts_param=NormalIntParam(mean=3, std=0.5, min_val=1, max_val=6),
        num_orders_param=NormalIntParam(mean=3, std=1, min_val=1, max_val=5),
        order_length_param=NormalIntParam(mean=4, std=0.5, min_val=1, max_val=5),
    )


    num_easy = 5

    # Generate instances
    for i in range(1, num_easy + 1):
        inst = generate_instance(inst_params, rng)
        filename = f"data_{i}.txt"
        filepath = os.path.join(output_dir, filename)
        write_instance_to_file(inst, filepath)
        print(f"Wrote {filepath}")



if __name__ == "__main__":
    #main()
    # Uncomment the following line to delete all generated instances:
    clear_all()
