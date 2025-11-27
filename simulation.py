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

@dataclass
class Robot:
    id : int
    current_pos : int
    loaded : int = 0
    isassigned : bool = False
    current_job : Job = None

def parse(filename: str) -> List[Product]:
    """
    Parse the data.txt and return a flat list of Product objects.
    """
    current_pos: List[int]
    products: List[Product] = []
    belts: List[int] = []
    product_id_counter = 1  # global increasing product ID
    order_index = 0         # O1=1, O2=2,...

    with open(filename, "r") as f:
        for line in f:
            line = line.strip()

            if line.startswith("Number of locations"):
                _, value = line.split(":")
                current_pos = [0, int(value.strip())-1]
            
            if line.startswith("Belts"):
                # Split at ":" and take the right side "2, 5, 9"
                _, values = line.split(":")
                belts = [int(x.strip()) for x in values.split(",")]
                continue

            if line.startswith("Order"):
                order_index += 1

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

    return products , current_pos



def schedule_jobs(solution: List[List[int]], products: List["Product"]) -> List[Job]:
    """
    solution[0]: list of pallet locations per order (index = order_id - 1)
    solution[1]: list of robot ids per product (index = product_id - 1)
    solution[2]: pickup sequence = list of product_ids in pickup order

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


def assign_job(robot_id: int)-> Job:
    print(len(jobs))
    for job in jobs:
        if job.robot_id == robot_id:
            return job
        



def robot_event(robot: Robot):
    job = robot.current_job
    
    if job is None:
        # go to starting position
        return 

    if job.destination[robot.loaded] > robot.current_pos:
        robot.current_pos += 1
        print(f"Robot {robot.id}: moves right to position {robot.current_pos}")
    elif job.destination[robot.loaded] < robot.current_pos:
        robot.current_pos -= 1
        print(f"Robot {robot.id}: moves left to position {robot.current_pos}")
    else: # at destination
        if robot.loaded == 0:
            robot.loaded = 1
            print(f"Robot {robot.id}: picks up product")
        else:
            jobs.remove(job)
            robot.isassigned = False
            print(f"Robot {robot.id}: delivers product for Order")



def simulation(starting_pos: List[int] ,jobs: List["Job"]) -> int:

    r0 = Robot(id=0, current_pos=starting_pos[0])
    r1 = Robot(id=1, current_pos=starting_pos[1])
    
    makespan = 0

    while len(jobs) > 0:
        print(f"--- Time step {makespan} ---")
        if r0.isassigned == False:
            r0.current_job = assign_job(r0.id)
            r0.loaded = 0
            r0.isassigned = True

        if r1.isassigned == False:
            r1.current_job = assign_job(r1.id)
            r1.loaded = 0
            r1.isassigned = True

        
        robot_event(r0)
        robot_event(r1)

        # Check for collision
        if r0.current_pos == r1.current_pos:
            print("Collision detected between Robot 0 and Robot 1")
            #return 
        

        makespan += 1
   
    return makespan


if __name__ == "__main__":
    products, robot_starting_pos = parse("data.txt")
    jobs = schedule_jobs(S, products)

    makespan = simulation(robot_starting_pos, jobs)
    
    for job in jobs:
        print(job)