from dataclasses import dataclass
from typing import List, Optional
import time
S = [
    [4, 2, 8],                             # pallet locations 
    [0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0],   # index = product_id - 1, value = robot_id
    [7, 8, 4, 1, 11, 10, 9, 5, 2, 6, 12, 3] # index = pickup sequence, value = product_id
]
S3 = [
    [6, 3, 9],
    [0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
    [4, 11, 9, 8, 3, 1, 7, 2, 12, 10, 5, 6]
]

S4 = [
    [2, 3, 4, 5, 1, 16, 6],# pallet locations
    [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1],# index = product_id - 1, value = robot_id
    [7, 17, 16, 8, 2, 3, 4, 5, 25, 11, 14, 13, 26, 12, 18, 6, 1, 10, 27, 23, 28, 15, 20, 9, 22, 21, 24, 19]# index = pickup sequence, value = product_id
]

S8 = [[1, 2, 3, 14, 5, 16, 7], 
      [0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1], 
      [4, 25, 11, 26, 9, 8, 19, 23, 15, 16, 5, 27, 22, 10, 7, 20, 12, 24, 14, 1, 17, 2, 28, 3, 6, 21, 18, 13]]

@dataclass(frozen=True)
class Product:
    product_id: int
    order_id: int
    belt_location: int
    # additional attributes for print messages
    product_in_order: int
    product_type: int


@dataclass
class Job:
    job_id: int
    robot_id: int
    destination: List[int]
    # additional attributes for print messages
    order_id: int
    product_type: int
    product_in_order: int


@dataclass
class Robot:
    id: int
    current_pos: int
    loaded: int = 0
    priority: bool = True
    memic: int =0
    current_job: Optional[Job] = None
    going_home: bool = False


#--------------- Parsing function ----------------#
def load_environment(filename: str) -> List[Product]:
    """
    Parse the data.txt and return a flat list of Product objects.
    """
    current_pos: List[int]
    products: List[Product] = []
    belts: List[int] = []
    product_id_counter = 1  # global increasing product ID
    order_index = 0  # O1=1, O2=2,...

    with open(filename, "r") as f:
        for line in f:
            line = line.strip()

            if line.startswith("Number of locations"):
                _, value = line.split(":")
                current_pos = [0, int(value.strip()) - 1]

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

    return products, current_pos, order_index

# when verbose is False, we don't want to print anything, so we can replace print with a no-op function
def noop(*args, **kwargs):
    pass

#--------------- Simulation class ----------------#

class Simulator:
    def __init__(self, solution: List[List[int]], starting_pos: List[int], policy: str = "default", verbose=False):
        self.policy = policy
        self.solution = solution
        self.starting_pos = starting_pos
        self.jobs: List[Job] = []

        self.log = print if verbose else noop

    def schedule_jobs(self, products: List["Product"]) -> List[Job]:
        """
        solution[0]: list of pallet locations per order (index = order_id - 1)
        solution[1]: list of robot ids per product (index = product_id - 1)
        solution[2]: pickup sequence = list of product_ids in pickup order
        """

        pallet_locations = self.solution[0]
        robot_ids = self.solution[1]
        pickup_sequence = self.solution[2]

        jobs: List[Job] = []

        for job in pickup_sequence:
            product = products[job - 1]

            job_id = product.product_id

            robot_id = robot_ids[job - 1]

            belt_location = product.belt_location
            pallet_location = pallet_locations[product.order_id - 1]

            job = Job(
                job_id=job_id,
                robot_id=robot_id,
                destination=[belt_location, pallet_location],
                order_id=product.order_id,
                product_in_order=product.product_in_order,
                product_type=product.product_type,
            )
            jobs.append(job)

        self.jobs = jobs

        return jobs

    def assign_job(self, robot: Robot):
        for job in self.jobs:
            if job.robot_id == robot.id:
                robot.current_job = job
                robot.loaded = 0
                robot.going_home = False

                self.log(
                    f"Robot {robot.id} assigned job to "
                    f"pickup product #{job.product_in_order} of order {job.order_id} "
                    f"(type {job.product_type}) from belt {job.destination[0]} "
                    f"and deliver to pallet {job.destination[1]}"
                )
                return

        robot.current_job = None
        robot.going_home = True
        self.log(f"Robot {robot.id} has no more jobs, returning home")

    def robot_event(self, robot: Robot):
        job = robot.current_job

        #if job is None:
        #    return

        if robot.priority is False:
            robot.current_pos += robot.memic # direction of other robot -> Positive(+1), Negative(-1), Neutral(0)
            self.log(f"Robot {robot.id} yields to position {robot.current_pos}")

        elif job is None:
            return
        
        elif job.destination[robot.loaded] > robot.current_pos:
            robot.current_pos += 1
            self.log(f"Robot {robot.id} moves right to position {robot.current_pos}")
        elif job.destination[robot.loaded] < robot.current_pos:
            robot.current_pos -= 1
            self.log(f"Robot {robot.id} moves left to position {robot.current_pos}")
        else:
            if robot.loaded == 0:
                robot.loaded = 1
                self.log(
                    f"Robot {robot.id} picks up product type {job.product_type} "
                    f"from belt position {job.destination[0]}"
                )
            else:
                self.log(
                    f"Robot {robot.id} delivers product type {job.product_type} "
                    f"to pallet position {job.destination[1]}"
                )
                self.jobs.remove(job)
                self.assign_job(robot)

    def go_home(self, robot: Robot):
        starting_pos = self.starting_pos[robot.id]

        if robot.current_pos < starting_pos:
            robot.current_pos += 1
            self.log(f"Robot {robot.id}: moves right to position {robot.current_pos} (going home)")
            if robot.current_pos == starting_pos:
                robot.going_home = False
                self.log(f"Robot {robot.id}: reached home position {robot.current_pos}")
        elif robot.current_pos > starting_pos:
            robot.current_pos -= 1
            self.log(f"Robot {robot.id}: moves left to position {robot.current_pos} (going home)")
            if robot.current_pos == starting_pos:
                robot.going_home = False
                self.log(f"Robot {robot.id}: reached home position {robot.current_pos}")
        else:
            robot.going_home = False
            self.log(f"Robot {robot.id}: already at home position {robot.current_pos}")

    def check_distance(self, r0: Robot, r1: Robot) -> int:
        return r1.current_pos - r0.current_pos
        
    def evaluate_penalty(self, total_jobs: int, finished_jobs: int, total_tu: int, L: int) -> int:
        big_penalty = L * total_jobs
        penalty_per_job = 10
        penalty_per_tu = 1

        unfinished_jobs = total_jobs - finished_jobs
        #penalty = big_penalty + (penalty_per_job * unfinished_jobs) + (penalty_per_tu * total_tu)
        penalty = (penalty_per_job * unfinished_jobs) + (penalty_per_tu * total_tu)
        self.log("Colision detected! Evaluating penalty:")
        self.log(f"penalty = {L}*{total_jobs} + ({penalty_per_job} * {unfinished_jobs}) + ({penalty_per_tu} * {total_tu}) = {penalty}")
        return penalty

    def direction(self, robot: Robot) -> int:
        job = robot.current_job

        if job is None:
            return -1 if robot.id == 0 else 1 # Robot 0 goes right, Robot 1 goes left when idle 
        elif job.destination[robot.loaded] > robot.current_pos:
            return 1
        elif job.destination[robot.loaded] < robot.current_pos:
            return -1
        else:
            return 0

    def approach(self, r0: Robot, r1: Robot) -> int:
        r0_dir = self.direction(r0)
        r1_dir = self.direction(r1)

        danger_case = {
            (1, -1): 1, # -> <-
            (1, 0): 2,  #-> _
            (0, -1): 3  # _ <-
        }
        return danger_case.get((r0_dir, r1_dir), 4) # 4 means no danger

    def manage_colision(self, r0: Robot, r1: Robot) -> bool:
        distance = r1.current_pos - r0.current_pos
        scenario = self.approach(r0, r1)
        if distance == 2 and scenario == 1:
            self.calculate_priority(r0, r1, True)
            self.log("Scenario 1: -> <-, distance = 2")
            return True
        elif distance == 1 and scenario == 1:
            self.calculate_priority(r0, r1, False)
            self.log("Scenario 2: -> <-, distance = 1")
            return True
        elif distance == 1 and scenario == 2:
            r0.priority = False
            r0.memic = self.direction(r1)
            r1.priority = True
            self.log("Scenario 3: -> _, distance = 1")
            return True
        elif distance == 1 and scenario == 3:
            r0.priority = True
            r1.priority = False
            r1.memic = self.direction(r0)
            self.log("Scenario 4: _ <-, distance = 1")
            return True
        else:
            r0.priority = True
            r1.priority = True
            return False
            
    def calculate_priority(self, r0: Robot, r1: Robot, stand: bool = False):
        if r0.loaded == 1 and r1.loaded == 0:
            r0.priority = True
            r1.priority = False
            r1.memic = self.direction(r0) if not stand else 0
            self.log("Priority: R0 loaded, R1 empty")
        elif r0.loaded == 0 and r1.loaded == 1:
            r1.priority = True
            r0.priority = False
            r0.memic = self.direction(r1) if not stand else 0
            self.log("Priority: R1 loaded, R0 empty")
        else:
            if abs(r0.current_job.destination[r0.loaded] - r0.current_pos) < abs(r1.current_job.destination[r1.loaded] - r1.current_pos):
                r0.priority = True
                r1.priority = False
                r1.memic = self.direction(r0) if not stand else 0
                self.log("Priority: R0 closer to destination")
            elif abs(r0.current_job.destination[r0.loaded] - r0.current_pos) > abs(r1.current_job.destination[r1.loaded] - r1.current_pos):
                r1.priority = True
                r0.priority = False
                r0.memic = self.direction(r1) if not stand else 0
                self.log("Priority: R1 closer to destination")
                
            else: # if both are equally close, we can choose to let the robot with the lower ID go first
                r0.priority = True
                r1.priority = False
                r1.memic = self.direction(r0) if not stand else 0
                self.log("Priority: Both robots equally close, Robot 0 goes first")
        

    def simulation2(self) -> int:
        r0 = Robot(id=0, current_pos=self.starting_pos[0])
        r1 = Robot(id=1, current_pos=self.starting_pos[1])

        total_jobs = len(self.jobs)
        L = self.starting_pos[1] - 2  # Number of locations - 2 parking spots
        makespan = 0
        colision = False

        self.log(f"--- Time step {makespan} ---")
        self.log(
            f"Robots are at starting positions: {self.starting_pos[0]}, {self.starting_pos[1]}"
        )
        self.assign_job(r0)
        self.assign_job(r1)
        while len(self.jobs) > 0 or r0.going_home or r1.going_home:                  
            colision = self.manage_colision(r0, r1)
            if self.policy =="default" and colision == True:
                finished_jobs = total_jobs - len(self.jobs)
                return self.evaluate_penalty(total_jobs, finished_jobs, makespan, L) , False

            makespan += 1
            self.log(f"--- Time step {makespan} ---")

            # Robot 0 event
            if r0.going_home:
                self.go_home(r0)
            else:
                self.robot_event(r0)
            # Robot 1 event
            if r1.going_home:
                self.go_home(r1)
            else:
                self.robot_event(r1)
                
            sanity_distance = self.check_distance(r0, r1)
            if not sanity_distance > 0:
                self.log("Error: Robots collided at position {r0.current_pos}!")
                print(self.solution)
                return 1000000000000000000 , False
                
        return makespan , True

if __name__ == "__main__":
    products, robot_starting_pos, orders = load_environment("instances/data_1.txt") # instances/data_1.txt | instances/data_2.txt | dataset/instance_l20_n30_1.txt

    sim = Simulator(solution=S8, starting_pos=robot_starting_pos, verbose=True, policy="priority") # default | priority
    sim.schedule_jobs(products)

    start = time.perf_counter()
    fitness , is_valid = sim.simulation2()
    end_time = time.perf_counter()

    print(end_time - start, "seconds")
    print(f"Simulation completed in {fitness} time steps.")