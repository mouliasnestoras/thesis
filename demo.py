"""
Twin Robot Warehouse Simulation — Visual Demo
---------------------------------------------
Runs the simulation directly (no log file needed).
State snapshots are captured in-memory during the run.

Usage:
    python robot_demo.py                        # uses DATA_FILE and SOLUTION below
    python robot_demo.py --data instances/data_1.txt

Edit SOLUTION and DATA_FILE at the top to switch scenarios.
"""

import tkinter as tk
from tkinter import font as tkfont
import argparse
import sys
from dataclasses import dataclass
from typing import List, Optional

# ═══════════════════════════════════════════════════════════════════
#  CONFIG — edit these to change scenario
# ═══════════════════════════════════════════════════════════════════

DATA_FILE = "instances/data_1.txt" # instances/data_1.txt | instances/data_2.txt
# makespan = 97
S4 = [
    [4, 3, 7, 1, 5, 14, 2],
    [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1],
    [15, 17, 25, 27, 19, 21, 8, 7, 2, 22, 18, 6, 4, 1, 10, 11, 24, 13, 14, 20, 16, 28, 3, 12, 9, 26, 23, 5],
]
S5 = [
    [3, 16, 14, 11],
    [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1],
    [15, 8, 11, 14, 7, 3, 5, 1, 16, 9, 12, 13, 4, 10, 2, 6]
]

SOLUTION = S4
POLICY   = "priority"   # "default" | "priority"


@dataclass(frozen=True)
class Product:
    product_id: int
    order_id: int
    belt_location: int
    product_in_order: int
    product_type: int


@dataclass
class Job:
    job_id: int
    robot_id: int
    destination: List[int]
    order_id: int
    product_type: int
    product_in_order: int


@dataclass
class Robot:
    id: int
    current_pos: int
    loaded: int = 0
    priority: bool = True
    memic: int = 0
    current_job: Optional[Job] = None
    going_home: bool = False


def load_environment(filename: str):
    products: List[Product] = []
    belts: List[int] = []
    product_id_counter = 1
    order_index = 0
    current_pos = [0, 0]

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
                product_index = 0
                for pt in product_types:
                    product_index += 1
                    belt_location = belts[pt - 1]
                    products.append(Product(
                        product_id=product_id_counter,
                        order_id=order_index,
                        belt_location=belt_location,
                        product_in_order=product_index,
                        product_type=pt,
                    ))
                    product_id_counter += 1

    return products, current_pos, order_index, belts


class Simulator:
    def __init__(self, solution, starting_pos, policy="default"):
        self.policy = policy
        self.solution = solution
        self.starting_pos = starting_pos
        self.jobs: List[Job] = []
        self._snapshots: list = []
        self._current_log: list = []

    def _log(self, msg: str):
        self._current_log.append(msg)

    def _robot_action(self, r: Robot) -> str:
        if r.going_home:
            return "home"
        if r.current_job is None:
            return "done"
        return "active"

    def _save_snapshot(self, step: int, r0: Robot, r1: Robot):
        self._snapshots.append({
            "step": step,
            "robots": {
                0: {
                    "pos": r0.current_pos,
                    "carrying": r0.current_job.product_type if (r0.current_job and r0.loaded == 1) else None,
                    "action": self._robot_action(r0),
                },
                1: {
                    "pos": r1.current_pos,
                    "carrying": r1.current_job.product_type if (r1.current_job and r1.loaded == 1) else None,
                    "action": self._robot_action(r1),
                },
            },
            "log": list(self._current_log),
        })
        self._current_log = []

    def schedule_jobs(self, products):
        pallet_locations = self.solution[0]
        robot_ids        = self.solution[1]
        pickup_sequence  = self.solution[2]
        jobs = []
        for job_pid in pickup_sequence:
            product = products[job_pid - 1]
            jobs.append(Job(
                job_id=product.product_id,
                robot_id=robot_ids[job_pid - 1],
                destination=[product.belt_location, pallet_locations[product.order_id - 1]],
                order_id=product.order_id,
                product_in_order=product.product_in_order,
                product_type=product.product_type,
            ))
        self.jobs = jobs
        return jobs

    def assign_job(self, robot: Robot):
        for job in self.jobs:
            if job.robot_id == robot.id:
                robot.current_job = job
                robot.loaded = 0
                robot.going_home = False
                self._log(
                    f"Robot {robot.id} assigned job to "
                    f"pickup product #{job.product_in_order} of order {job.order_id} "
                    f"(type {job.product_type}) from belt {job.destination[0]} "
                    f"and deliver to pallet {job.destination[1]}"
                )
                return
        robot.current_job = None
        robot.going_home = True
        self._log(f"Robot {robot.id} has no more jobs, returning home")

    def robot_event(self, robot: Robot):
        job = robot.current_job
        if job is None:
            return
        if not robot.priority:
            robot.current_pos += robot.memic
            self._log(f"Robot {robot.id} yields to position {robot.current_pos}")
        elif job.destination[robot.loaded] > robot.current_pos:
            robot.current_pos += 1
            self._log(f"Robot {robot.id} moves right to position {robot.current_pos}")
        elif job.destination[robot.loaded] < robot.current_pos:
            robot.current_pos -= 1
            self._log(f"Robot {robot.id} moves left to position {robot.current_pos}")
        else:
            if robot.loaded == 0:
                robot.loaded = 1
                self._log(
                    f"Robot {robot.id} picks up product type {job.product_type} "
                    f"from belt position {job.destination[0]}"
                )
            else:
                self._log(
                    f"Robot {robot.id} delivers product type {job.product_type} "
                    f"to pallet position {job.destination[1]}"
                )
                self.jobs.remove(job)
                self.assign_job(robot)

    def go_home(self, robot: Robot):
        sp = self.starting_pos[robot.id]
        if robot.current_pos < sp:
            robot.current_pos += 1
            self._log(f"Robot {robot.id}: moves right to position {robot.current_pos} (going home)")
            if robot.current_pos == sp:
                robot.going_home = False
                self._log(f"Robot {robot.id}: reached home position {robot.current_pos}")
        elif robot.current_pos > sp:
            robot.current_pos -= 1
            self._log(f"Robot {robot.id}: moves left to position {robot.current_pos} (going home)")
            if robot.current_pos == sp:
                robot.going_home = False
                self._log(f"Robot {robot.id}: reached home position {robot.current_pos}")

    def direction(self, robot: Robot) -> int:
        job = robot.current_job
        if job is None:
            return 0
        elif job.destination[robot.loaded] > robot.current_pos:
            return 1
        elif job.destination[robot.loaded] < robot.current_pos:
            return -1
        return 0

    def approach(self, r0, r1) -> int:
        r0d, r1d = self.direction(r0), self.direction(r1)
        return {(1, -1): 1, (1, 0): 2, (0, -1): 3}.get((r0d, r1d), 4)

    def calculate_priority(self, r0, r1, stand=False):
        if r0.loaded == 1 and r1.loaded == 0:
            r0.priority = True; r1.priority = False
            r1.memic = self.direction(r0) if not stand else 0
            self._log("Priority: R0 loaded, R1 empty")
        elif r0.loaded == 0 and r1.loaded == 1:
            r1.priority = True; r0.priority = False
            r0.memic = self.direction(r1) if not stand else 0
            self._log("Priority: R1 loaded, R0 empty")
        else:
            d0 = abs(r0.current_job.destination[r0.loaded] - r0.current_pos)
            d1 = abs(r1.current_job.destination[r1.loaded] - r1.current_pos)
            if d0 < d1:
                r0.priority = True; r1.priority = False
                r1.memic = self.direction(r0) if not stand else 0
                self._log("Priority: R0 closer to destination")
            elif d0 > d1:
                r1.priority = True; r0.priority = False
                r0.memic = self.direction(r1) if not stand else 0
                self._log("Priority: R1 closer to destination")

    def manage_collision(self, r0, r1) -> bool:
        distance = r1.current_pos - r0.current_pos
        scenario = self.approach(r0, r1)
        if distance == 2 and scenario == 1:
            self.calculate_priority(r0, r1, True)
            self._log("Scenario 1: -> <-, distance = 2")
            return True
        elif distance == 1 and scenario == 1:
            self.calculate_priority(r0, r1, False)
            self._log("Scenario 2: -> <-, distance = 1")
            return True
        elif distance == 1 and scenario == 2:
            r0.priority = False; r0.memic = self.direction(r1); r1.priority = True
            self._log("Scenario 3: -> _, distance = 1")
            return True
        elif distance == 1 and scenario == 3:
            r0.priority = True; r1.priority = False; r1.memic = self.direction(r0)
            self._log("Scenario 4: _ <-, distance = 1")
            return True
        else:
            r0.priority = True; r1.priority = True
            return False

    def evaluate_penalty(self, total_jobs, finished_jobs, total_tu, L) -> int:
        return L * total_jobs + 10 * (total_jobs - finished_jobs) + total_tu

    def run(self):
        """Run full simulation; return (snapshots, makespan_or_penalty, is_valid)."""
        self._snapshots = []
        self._current_log = []

        r0 = Robot(id=0, current_pos=self.starting_pos[0])
        r1 = Robot(id=1, current_pos=self.starting_pos[1])

        total_jobs = len(self.jobs)
        L = self.starting_pos[1] - 2
        makespan = 0

        self._log(f"Robots are at starting positions: {self.starting_pos[0]}, {self.starting_pos[1]}")
        self.assign_job(r0)
        self.assign_job(r1)
        self._save_snapshot(makespan, r0, r1)

        while len(self.jobs) > 0 or r0.going_home or r1.going_home:
            collision = self.manage_collision(r0, r1)
            if self.policy == "default" and collision:
                finished = total_jobs - len(self.jobs)
                penalty = self.evaluate_penalty(total_jobs, finished, makespan, L)
                self._save_snapshot(makespan, r0, r1)
                return self._snapshots, penalty, False

            makespan += 1

            if r0.going_home:
                self.go_home(r0)
            else:
                self.robot_event(r0)

            if r1.going_home:
                self.go_home(r1)
            else:
                self.robot_event(r1)

            self._save_snapshot(makespan, r0, r1)

        return self._snapshots, makespan, True


# ═══════════════════════════════════════════════════════════════════
#  THEME
# ═══════════════════════════════════════════════════════════════════

BG          = "#0d1117"
GRID_LINE   = "#1e2730"
BELT_COL    = "#1f3a5f"
BELT_ACCENT = "#2979ff"
PALLET_COL  = "#1a3328"
PALLET_ACC  = "#00e676"
ROBOT0_COL  = "#ff6d00"
ROBOT1_COL  = "#d500f9"
TEXT_COL    = "#e6edf3"
DIM_COL     = "#586069"
LOG_BG      = "#010409"
YIELD_COL   = "#ffd600"
HOME_COL    = "#37474f"

CELL_W   = 54
CELL_H   = 90
TOP_PAD  = 110
LEFT_PAD = 30
BOT_PAD  = 10


# ═══════════════════════════════════════════════════════════════════
#  GUI
# ═══════════════════════════════════════════════════════════════════

class SimApp(tk.Tk):
    def __init__(self, n_locations, belts, pallet_locations, states,
                 starting_pos, makespan, is_valid):
        super().__init__()
        self.title("Twin Robot Warehouse Simulator")
        self.configure(bg=BG)
        self.resizable(False, False)

        self.n            = n_locations
        self.belts        = belts
        self.pallet_locs  = pallet_locations
        self.states       = states
        self.starting_pos = starting_pos
        self.makespan     = makespan
        self.is_valid     = is_valid

        self.idx      = 0
        self._playing = False
        self._speed   = 300

        self.cw = LEFT_PAD * 2 + self.n * CELL_W
        self.ch = TOP_PAD + CELL_H + BOT_PAD

        self._build_ui()
        self._draw_state()

    # ── layout ─────────────────────────────────────────────────────

    def _build_ui(self):
        self.f_mono  = tkfont.Font(family="Courier New", size=10)
        self.f_label = tkfont.Font(family="Courier New", size=8)
        self.f_title = tkfont.Font(family="Courier New", size=13, weight="bold")
        self.f_step  = tkfont.Font(family="Courier New", size=18, weight="bold")
        self.f_badge = tkfont.Font(family="Courier New", size=7,  weight="bold")

        # header
        hdr = tk.Frame(self, bg=BG)
        hdr.pack(fill="x", padx=16, pady=(12, 0))
        tk.Label(hdr, text="⬡ TWIN ROBOT WAREHOUSE", font=self.f_title,
                 bg=BG, fg=TEXT_COL).pack(side="left")
        status_col = PALLET_ACC if self.is_valid else "#ff1744"
        status_txt = (f"✓ VALID  makespan={self.makespan}" if self.is_valid
                      else f"✗ COLLISION  penalty={self.makespan}")
        tk.Label(hdr, text=status_txt, font=self.f_label,
                 bg=BG, fg=status_col).pack(side="left", padx=16)
        self.lbl_step = tk.Label(hdr, text="STEP 000", font=self.f_step,
                                  bg=BG, fg=BELT_ACCENT)
        self.lbl_step.pack(side="right")

        # canvas
        self.canvas = tk.Canvas(self, width=self.cw, height=self.ch,
                                bg=BG, highlightthickness=0)
        self.canvas.pack(padx=16, pady=6)

        # robot info row
        info = tk.Frame(self, bg=BG)
        info.pack(fill="x", padx=16)
        self.r0_lbl = tk.Label(info, text="", font=self.f_mono,
                                bg=BG, fg=ROBOT0_COL, anchor="w", width=42)
        self.r0_lbl.pack(side="left")
        self.r1_lbl = tk.Label(info, text="", font=self.f_mono,
                                bg=BG, fg=ROBOT1_COL, anchor="e", width=42)
        self.r1_lbl.pack(side="right")

        # log box
        self.log_text = tk.Text(self, height=7, bg=LOG_BG, fg=DIM_COL,
                                font=self.f_label, relief="flat",
                                insertbackground=BG, wrap="word",
                                highlightthickness=1,
                                highlightbackground=GRID_LINE)
        self.log_text.pack(fill="x", padx=16, pady=(4, 4))
        self.log_text.configure(state="disabled")

        # controls
        ctrl = tk.Frame(self, bg=BG)
        ctrl.pack(pady=(0, 8))
        btn_cfg = dict(bg="#1c2128", fg=TEXT_COL, font=self.f_mono,
                       relief="flat", padx=12, pady=5, cursor="hand2",
                       activebackground="#2d333b", activeforeground=TEXT_COL, bd=0)
        tk.Button(ctrl, text="◀ PREV",  command=self._prev,         **btn_cfg).grid(row=0, column=0, padx=4)
        self.btn_play = tk.Button(ctrl, text="▶ PLAY", command=self._toggle_play, **btn_cfg)
        self.btn_play.grid(row=0, column=1, padx=4)
        tk.Button(ctrl, text="NEXT ▶",  command=self._next,         **btn_cfg).grid(row=0, column=2, padx=4)
        tk.Label(ctrl, text="SPEED", bg=BG, fg=DIM_COL,
                 font=self.f_label).grid(row=0, column=3, padx=(20, 4))
        self.speed_var = tk.IntVar(value=300)
        tk.Scale(ctrl, from_=50, to=1000, orient="horizontal",
                 variable=self.speed_var, bg=BG, fg=TEXT_COL,
                 troughcolor=GRID_LINE, highlightthickness=0,
                 sliderrelief="flat", length=120,
                 command=lambda v: setattr(self, "_speed", int(v))
                 ).grid(row=0, column=4, padx=4)

        # scrubber
        self.scrub_var = tk.IntVar(value=0)
        tk.Scale(self, from_=0, to=max(0, len(self.states) - 1),
                 orient="horizontal", variable=self.scrub_var,
                 bg=BG, fg=DIM_COL, troughcolor=GRID_LINE,
                 highlightthickness=0, sliderrelief="flat",
                 length=self.cw, showvalue=False,
                 command=self._on_scrub).pack(padx=16, pady=(0, 10))

    # ── drawing ────────────────────────────────────────────────────

    def _draw_state(self):
        c = self.canvas
        c.delete("all")
        state  = self.states[self.idx]
        step   = state["step"]
        robots = state["robots"]

        # Grid
        for i in range(self.n):
            x = LEFT_PAD + i * CELL_W
            c.create_rectangle(x, TOP_PAD, x + CELL_W - 2, TOP_PAD + CELL_H,
                                fill=GRID_LINE, outline="")
            c.create_text(x + CELL_W // 2, TOP_PAD + CELL_H + 12,
                          text=str(i), fill=DIM_COL, font=self.f_label)

        # Belts
        for pos in self.belts:
            if 0 <= pos < self.n:
                x = LEFT_PAD + pos * CELL_W
                c.create_rectangle(x + 2, TOP_PAD + 2, x + CELL_W - 4, TOP_PAD + CELL_H - 2,
                                   fill=BELT_COL, outline=BELT_ACCENT, width=1)
                c.create_text(x + CELL_W // 2, TOP_PAD + 14,
                              text="BELT", fill=BELT_ACCENT, font=self.f_label)
                c.create_text(x + CELL_W // 2, TOP_PAD + 26,
                              text=f"@{pos}", fill=BELT_ACCENT, font=self.f_label)

        # Pallets
        for i, pos in enumerate(self.pallet_locs):
            if 0 <= pos < self.n:
                x = LEFT_PAD + pos * CELL_W
                c.create_rectangle(x + 4, TOP_PAD + CELL_H - 34,
                                   x + CELL_W - 6, TOP_PAD + CELL_H - 6,
                                   fill=PALLET_COL, outline=PALLET_ACC, width=1)
                c.create_text(x + CELL_W // 2, TOP_PAD + CELL_H - 24,
                              text=f"O{i + 1}", fill=PALLET_ACC, font=self.f_label)
                c.create_text(x + CELL_W // 2, TOP_PAD + CELL_H - 12,
                              text=f"@{pos}", fill=PALLET_ACC, font=self.f_label)

        # Home markers
        for rid, sp in enumerate(self.starting_pos):
            col = ROBOT0_COL if rid == 0 else ROBOT1_COL
            x = LEFT_PAD + sp * CELL_W
            c.create_text(x + CELL_W // 2, TOP_PAD - 16,
                          text=f"HOME{rid}", fill=col, font=self.f_label)

        # Robots
        for rid, rdata in robots.items():
            pos      = rdata["pos"]
            action   = rdata["action"]
            carrying = rdata["carrying"]

            col = ROBOT0_COL if rid == 0 else ROBOT1_COL
            if action == "yield": col = YIELD_COL
            elif action == "home": col = HOME_COL
            elif action == "done": col = DIM_COL

            x  = LEFT_PAD + pos * CELL_W
            yo = 8 if rid == 0 else 36

            c.create_oval(x + 5, TOP_PAD + yo,
                          x + CELL_W - 7, TOP_PAD + yo + 26,
                          fill=col, outline="white", width=1)
            c.create_text(x + CELL_W // 2, TOP_PAD + yo + 13,
                          text=f"R{rid}", fill="white", font=self.f_badge)

            if carrying is not None:
                bx = x + CELL_W - 15
                by = TOP_PAD + yo - 5
                c.create_rectangle(bx, by, bx + 13, by + 12, fill="#ffd600", outline="")
                c.create_text(bx + 6, by + 6, text=str(carrying), fill="black", font=self.f_badge)

        # Widgets update
        self.lbl_step.config(text=f"STEP {step:03d}")
        self.scrub_var.set(self.idx)

        r0, r1 = robots[0], robots[1]
        c0 = f"type {r0['carrying']}" if r0["carrying"] is not None else "empty"
        c1 = f"type {r1['carrying']}" if r1["carrying"] is not None else "empty"
        self.r0_lbl.config(text=f"● R0  pos={r0['pos']:>2}  {c0:<12}  [{r0['action']}]")
        self.r1_lbl.config(text=f"[{r1['action']}]  {c1:<12}  pos={r1['pos']:>2}  R1 ●")

        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        for line in state["log"]:
            if "Robot 0" in line:
                tag = "r0"
            elif "Robot 1" in line:
                tag = "r1"
            elif "Priority" in line or "Scenario" in line:
                tag = "warn"
            else:
                tag = "info"
            self.log_text.insert("end", "  " + line + "\n", tag)
        self.log_text.tag_configure("r0",   foreground=ROBOT0_COL)
        self.log_text.tag_configure("r1",   foreground=ROBOT1_COL)
        self.log_text.tag_configure("warn", foreground=YIELD_COL)
        self.log_text.tag_configure("info", foreground=DIM_COL)
        self.log_text.configure(state="disabled")

    # ── controls ───────────────────────────────────────────────────

    def _next(self):
        if self.idx < len(self.states) - 1:
            self.idx += 1
            self._draw_state()

    def _prev(self):
        if self.idx > 0:
            self.idx -= 1
            self._draw_state()

    def _on_scrub(self, val):
        self.idx = int(val)
        self._draw_state()

    def _toggle_play(self):
        self._playing = not self._playing
        self.btn_play.config(text="⏸ PAUSE" if self._playing else "▶ PLAY")
        if self._playing:
            self._auto_step()

    def _auto_step(self):
        if not self._playing:
            return
        if self.idx < len(self.states) - 1:
            self.idx += 1
            self._draw_state()
            self.after(self._speed, self._auto_step)
        else:
            self._playing = False
            self.btn_play.config(text="▶ PLAY")


# ═══════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Robot Simulation Visual Demo")
    parser.add_argument("--data",   default=DATA_FILE, help="Path to environment data file")
    parser.add_argument("--policy", default=POLICY,    help="Collision policy: default | priority")
    args = parser.parse_args()

    try:
        products, robot_starting_pos, n_orders, belts = load_environment(args.data)
    except FileNotFoundError:
        print(f"ERROR: data file not found: '{args.data}'")
        sys.exit(1)

    sim = Simulator(solution=SOLUTION, starting_pos=robot_starting_pos, policy=args.policy)
    sim.schedule_jobs(products)
    states, makespan, is_valid = sim.run()

    n_locations  = robot_starting_pos[1] + 1
    pallet_locs  = SOLUTION[0]

    app = SimApp(
        n_locations      = n_locations,
        belts            = belts,
        pallet_locations = pallet_locs,
        states           = states,
        starting_pos     = robot_starting_pos,
        makespan         = makespan,
        is_valid         = is_valid,
    )
    app.mainloop()


if __name__ == "__main__":
    main()