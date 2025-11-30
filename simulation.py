from dataclasses import dataclass
from typing import List

# Example solution S works as an initial solution
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
    product_type : int

@dataclass
class Job:
    job_id  : int
    robot_id    : int
    destination : List[int]
    # additional attributes for print messages
    order_id : int
    product_type : int
    product_in_order : int
    
@dataclass
class Robot:
    id : int
    current_pos : int
    loaded : int = 0
    priority : bool = True
    current_job : Job = None
    going_home : bool = False
#--------------- Parsing function ----------------#
def load_environment(filename: str) -> List[Product]:
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
                        product_in_order=product_index,
                        product_type=pt
                    ))
                    product_id_counter += 1

    return products , current_pos

#--------------- Simulation functions ----------------#

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
            product_in_order=product.product_in_order,
            product_type=product.product_type
        )
        jobs.append(job)

        
    return jobs


def assign_job(robot):
    for job in jobs:
        if job.robot_id == robot.id:
            robot.current_job = job
            robot.loaded = 0
            #robot.isassigned = True
            robot.going_home = False

            print(
                f"Robot {robot.id} assigned job to "
                f"pickup product #{job.product_in_order} of order {job.order_id} "
                f"(type {job.product_type}) from belt {job.destination[0]} "
                f"and deliver to pallet {job.destination[1]}"
            )
            return

    # No matching job found:
    robot.current_job = None
    #robot.isassigned = True
    robot.going_home = True
    print(f"Robot {robot.id} has no more jobs, returning home")

def robot_event(robot: Robot):
    job = robot.current_job
    
    if job is None:
        return 

    if robot.priority == False:
        if robot.id == 0:
            robot.current_pos -= 1
            print(f"Robot {robot.id} moves left to position {robot.current_pos}")
        else:
            robot.current_pos += 1
            print(f"Robot {robot.id} moves right to position {robot.current_pos}")
    elif job.destination[robot.loaded] > robot.current_pos:
        robot.current_pos += 1
        print(f"Robot {robot.id} moves right to position {robot.current_pos}")
    elif job.destination[robot.loaded] < robot.current_pos:
        robot.current_pos -= 1
        print(f"Robot {robot.id} moves left to position {robot.current_pos}")
    else:
        if robot.loaded == 0:
            robot.loaded = 1
            print(
                f"Robot {robot.id} picks up product type {job.product_type} "
                f"from belt position {job.destination[0]}"
            )
        else:
            print(
                f"Robot {robot.id} delivers product type {job.product_type} "
                f"to pallet position {job.destination[1]}"
            )
            jobs.remove(job)
            assign_job(robot)

def go_home(robot: Robot, starting_pos: int):
    if robot.current_pos < starting_pos:
        robot.current_pos += 1
        print(f"Robot {robot.id}: moves right to position {robot.current_pos} (going home)")
    elif robot.current_pos > starting_pos:
        robot.current_pos -= 1
        print(f"Robot {robot.id}: moves left to position {robot.current_pos} (going home)")
    else:
        robot.going_home = False
        print(f"Robot {robot.id}: reached home position {robot.current_pos}")


def check_distance(r0: Robot, r1: Robot, policy) -> int:
    if policy == "default":
        return abs(r1.current_pos - r0.current_pos)
    else: #For now. There shall be other policies later
        if abs(r1.current_pos - r0.current_pos) == 1:
            # here will probably be a dicide_priority function
            r1.priority = False
        return abs(r1.current_pos - r0.current_pos)

def evaluate_penalty(total_jobs: int, finished_jobs: int, total_tu: int, L: int) -> int:
    """
    Calculate penalty for collision scenarios.
    
    F(S) = big_penalty + penalty_per_job*(|O|-finished_jobs) + penalty_per_tu*total_tu
    
    Args:
        total_jobs: Total number of jobs |O|
        finished_jobs: Number of completed jobs
        total_tu: Total time units elapsed until collision
        L: Number of locations (for calculating big_penalty)
    
    Returns:
        Penalty value
    """
    big_penalty = L * total_jobs
    penalty_per_job = 10
    penalty_per_tu = 1
    
    unfinished_jobs = total_jobs - finished_jobs
    
    penalty = big_penalty + (penalty_per_job * unfinished_jobs) + (penalty_per_tu * total_tu)
    
    return penalty

def simulation(starting_pos: List[int] ,jobs: List["Job"], policy = "default") -> int:

    r0 = Robot(id=0, current_pos=starting_pos[0])
    r1 = Robot(id=1, current_pos=starting_pos[1])
    
    total_jobs = len(jobs)
    L = starting_pos[1] - 1  # Number of locations
    makespan = 0
    print(f"--- Time step {makespan} ---")
    print(f"Robots are at starting positions: {starting_pos[0]}, {starting_pos[1]}")
    assign_job(r0)
    assign_job(r1)
    while len(jobs) > 0 or r0.going_home or r1.going_home:
        
        makespan += 1
        print(f"--- Time step {makespan} ---")

        if check_distance(r0, r1, policy) == 0:
            finished_jobs = total_jobs - len(jobs)
            return evaluate_penalty(total_jobs, finished_jobs, makespan, L)
        if r0.going_home:
            go_home(r0, starting_pos[0])
        else:
            robot_event(r0)
        
        if check_distance(r0, r1, policy) == 0:
            finished_jobs = total_jobs - len(jobs)
            return evaluate_penalty(total_jobs, finished_jobs, makespan,L)
        if r1.going_home:
            go_home(r1, starting_pos[1])
        else:
            robot_event(r1)

    return makespan


if __name__ == "__main__":
    products, robot_starting_pos = load_environment("data.txt")
    jobs = schedule_jobs(S, products)

    fitness = simulation(robot_starting_pos, jobs)
    
    print(f"Simulation completed the fitness score is: {fitness}")