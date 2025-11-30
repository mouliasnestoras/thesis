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
#--------------- Parsing functions ----------------#
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

#--------------- Simulation functions ----------------#
class simulator:
    def __init__(self, policy: str = "default"):
        self.policy = policy
        self.jobs: List[Job] = []

    def assign_job(self, robot: Robot):
        for job in self.jobs:
            if job.robot_id == robot.id:
                robot.current_job = job
                robot.loaded = 0
                robot.going_home = False

                print(
                    f"Robot {robot.id} assigned job to "
                    f"pickup product #{job.product_in_order} of order {job.order_id} "
                    f"(type {job.product_type}) from belt {job.destination[0]} "
                    f"and deliver to pallet {job.destination[1]}"
                )
                return

        robot.current_job = None
        robot.going_home = True
        print(f"Robot {robot.id} has no more jobs, returning home")

    def robot_event(self, robot: Robot):
        job = robot.current_job

        if job is None:
            return

        if robot.priority is False:
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
                self.jobs.remove(job)
                self.assign_job(robot)

    def go_home(self, robot: Robot, starting_pos: int):
        if robot.current_pos < starting_pos:
            robot.current_pos += 1
            print(f"Robot {robot.id}: moves right to position {robot.current_pos} (going home)")
        elif robot.current_pos > starting_pos:
            robot.current_pos -= 1
            print(f"Robot {robot.id}: moves left to position {robot.current_pos} (going home)")
        else:
            robot.going_home = False
            print(f"Robot {robot.id}: reached home position {robot.current_pos}")

    def check_distance(self, r0: Robot, r1: Robot) -> int:
        if self.policy == "default":
            return abs(r1.current_pos - r0.current_pos)

        if abs(r1.current_pos - r0.current_pos) == 1:
            r1.priority = False
        return abs(r1.current_pos - r0.current_pos)

    def evaluate_penalty(self) -> int:
        return 9999

    def simulation(self, starting_pos: List[int], jobs: List["Job"]) -> int:
        self.jobs = jobs

        r0 = Robot(id=0, current_pos=starting_pos[0])
        r1 = Robot(id=1, current_pos=starting_pos[1])

        makespan = 0
        print(f"--- Time step {makespan} ---")
        print(f"Robots are at starting positions: {r0.current_pos}, {r1.current_pos}")
        self.assign_job(r0)
        self.assign_job(r1)
        while len(self.jobs) > 0 or r0.going_home or r1.going_home:
            makespan += 1
            print(f"--- Time step {makespan} ---")

            if self.check_distance(r0, r1) == 0:
                return self.evaluate_penalty()
            if r0.going_home:
                self.go_home(r0, starting_pos[0])
            else:
                self.robot_event(r0)

            if self.check_distance(r0, r1) == 0:
                return self.evaluate_penalty()
            if r1.going_home:
                self.go_home(r1, starting_pos[1])
            else:
                self.robot_event(r1)

        return makespan


if __name__ == "__main__":
    products, robot_starting_pos = parse("data.txt")
    jobs = schedule_jobs(S, products)

    sim = simulator()
    fitness = sim.simulation(robot_starting_pos, jobs)

    print(f"Simulation completed in {fitness} time steps.")
    for job in jobs:
        print(job)
