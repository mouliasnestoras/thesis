from typing import Generator, List, Tuple

from simulation import Product


def parse_input_file(path: str):
    number_of_locations = None
    belts = []
    orders = {}

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith("Number of locations"):
                number_of_locations = int(line.split(":")[1])

            elif line.startswith("Belts"):
                parts = line.split(":")[1]
                belts = [int(x.strip()) for x in parts.split(",")]

            elif line.startswith("Order"):
                key_part, nums_part = line.split(":")
                order_name = key_part.split()[1]
                nums = [int(x.strip()) for x in nums_part.split(",")]
                orders[order_name] = nums

    return {
        "number_of_locations": number_of_locations,
        "belts": belts,
        "orders": orders,
    }


def iter_problems(path: str) -> Generator[Tuple[List[Product], List[int]], None, None]:
    """Yield one parsed problem at a time from a combined data file.

    The parser consumes lines until it finishes a single problem definition
    (terminated by a blank line or end-of-file), immediately yields the
    corresponding products list and starting positions, and then continues
    to the next problem on the next iteration.
    """

    with open(path, "r") as f:
        while True:
            belts: List[int] = []
            products: List[Product] = []
            product_id_counter = 1
            order_index = 0
            starting_positions: List[int] = []

            for raw_line in f:
                line = raw_line.strip()

                if not line:
                    if products or starting_positions:
                        yield products, starting_positions
                        break
                    continue

                if line.startswith("Number of locations"):
                    _, value = line.split(":")
                    starting_positions = [0, int(value.strip()) - 1]
                    continue

                if line.startswith("Belts"):
                    _, values = line.split(":")
                    belts = [int(x.strip()) for x in values.split(",")]
                    continue

                if line.startswith("Order"):
                    order_index += 1
                    _, values = line.split(":")
                    product_types = [int(x.strip()) for x in values.split(",")]

                    product_index = 0
                    for pt in product_types:
                        product_index += 1
                        belt_location = belts[pt - 1]
                        products.append(
                            Product(
                                product_id=product_id_counter,
                                order_id=order_index,
                                belt_location=belt_location,
                                product_in_order=product_index,
                                product_type=pt,
                            )
                        )
                        product_id_counter += 1
                    continue

            else:
                if products or starting_positions:
                    yield products, starting_positions
                break


def _demo_main():
    """Pseudo main showing how to consume problems one by one.

    This demonstrates how to iterate over multiple problem definitions in
    ``data.txt`` (or any file with the same format), schedule the jobs using
    a provided solution vector, and run the simulation. Replace the example
    solution ``S`` with your algorithm's output when integrating.
    """

    from simulation import S, Simulator

    for idx, (products, starting_positions) in enumerate(iter_problems("data.txt"), start=1):
        print(f"=== Problem {idx} ===")
        sim = Simulator(solution=S, starting_pos=starting_positions)
        sim.schedule_jobs(products)
        fitness = sim.simulation()
        print(f"Problem {idx} completed with fitness {fitness}\n")


if __name__ == "__main__":
    _demo_main()
