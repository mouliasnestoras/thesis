from dataclasses import dataclass
from typing import List

S = [
    [4, 2, 8],                             # Pallet location 
    [0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0],   # Field 2
    [7, 8, 4, 1, 11, 10, 9, 5, 2, 6, 12, 3] # Field 3
]

@dataclass(frozen=True)
class Product:
    product_id : int
    order_id   : int
    belt_location : int
    # additional attributes for print messages
    product_in_order : int

@dataclass
class Job:
    job_id  : int
    robot_id    : int
    destination : List[int]
    # additional attributes for print messages
    order_id : int
    product_in_order : int


def parse(filename: str) -> List[Product]:
    """
    Parse the data.txt and return a flat list of Product objects.
    """

    products: List[Product] = []
    belts: List[int] = []
    product_id_counter = 1  # global increasing product ID
    order_index = 0         # O1=1, O2=2,...

    with open(filename, "r") as f:
        for line in f:
            line = line.strip()

            # ----------------------------------------
            # Parse belt line: "Belts P1, P2, P3: 2, 5, 9"
            # ----------------------------------------
            if line.startswith("Belts"):
                # Split at ":" and take the right side "2, 5, 9"
                _, values = line.split(":")
                belts = [int(x.strip()) for x in values.split(",")]
                continue

            # ----------------------------------------
            # Parse each order: "Order O1: 1, 1, 3, 3, 2"
            # ----------------------------------------
            if line.startswith("Order"):
                order_index += 1  # O1 -> 1, O2 -> 2, ...

                # Extract product type numbers after ':'
                _, values = line.split(":")
                product_types = [int(x.strip()) for x in values.split(",")]

                # Create Product objects for every product in this order
                product_index = 0
                for pt in product_types:
                    product_index += 1

                    belt_location = belts[pt - 1]  # type→belt index
                    products.append(Product(
                        product_id=product_id_counter,
                        order_id=order_index,
                        belt_location=belt_location,
                        product_in_order=product_index
                    ))
                    product_id_counter += 1

    return products



def schedule_jobs(solution: List[List[int]], products: List["Product"]) -> List[Job]:
    """
    Create a list of Job objects from a solution vector and a list of products.

    solution[0]: list of pallet locations per order (index = order_id - 1)
    solution[1]: list of robot ids per product (index = product_id - 1)
    solution[2]: pickup sequence = list of product_ids in pickup order

    Each Job has:
      - product_id
      - robot_id  (from solution[1])
      - destination = [belt_location, pallet_location]

    """

    pallet_locations = solution[0]
    robot_ids = solution[1]
    pickup_sequence = solution[2]

    jobs: List[Job] = []

    for job in pickup_sequence:
        product = products[job-1]

        job_id = product.product_id

        # robot id is indexed by (product_id - 1)
        robot_id = robot_ids[job-1]

        # destination[0] = belt location (from Product)
        belt_location = product.belt_location

        # destination[1] = pallet location (from solution[0])
        pallet_location = pallet_locations[product.order_id - 1]

        job = Job(
            job_id= job_id,
            robot_id=robot_id,
            destination=[belt_location, pallet_location],
            order_id=product.order_id,
            product_in_order=product.product_in_order
        )
        jobs.append(job)

    return jobs

    

def simulation():
    pass

if __name__ == "__main__":
    products = parse("data.txt")
    
    jobs = schedule_jobs(S, products)
    for job in jobs:
        print(job)