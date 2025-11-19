# ================================================================
# MIT License
#
# Copyright (c) 2025 Robert Vrabel
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
# ================================================================

import numpy as np
import pandas as pd
import time
import random
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp
from mealpy.swarm_based.PSO import OriginalPSO
from mealpy.evolutionary_based.DE import OriginalDE
from mealpy.utils.space import FloatVar
from mealpy import Problem


# ============================================================
# GLOBAL SETTINGS
# ============================================================
GLOBAL_SEED = 123
np.random.seed(GLOBAL_SEED)
random.seed(GLOBAL_SEED)

T_END = 25.0
N_T = 1001
TIME_GRID = np.linspace(0, T_END, N_T)
alpha = beta = 1.0

PID_BOUNDS = (0.0, 10.0)
PIDD_BOUNDS = (0.0, 10.0)

EPOCHS = 25 
POP_SIZE = 30
N_RUNS = 50 


# ============================================================
# SPROTT SYSTEM
# ============================================================
def disturbance(t):
    return 0.5 * np.sin(2.5 * t)

def sprott_pid_ode(t, s, Kp, Ki, Kd):
    x, y, z, I = s
    e, de = -x, -y
    u = Kp * e + Ki * I + Kd * de
    d = disturbance(t)
    return [y, z,
            -x - 0.6*y - 2*z + z*z - 0.4*x*y + d + u,
            e]

def sprott_pidd_ode(t, s, Kp, Ki, Kd, Kdd):
    x, y, z, I = s
    e, de, dde = -x, -y, -z
    u = Kp*e + Ki*I + Kd*de + Kdd*dde
    d = disturbance(t)
    return [y, z,
            -x - 0.6*y - 2*z + z*z - 0.4*x*y + d + u,
            e]


# ============================================================
# SIMULATION + COST
# ============================================================
def simulate_system(controller, gains):

    if controller == "PID":
        Kp, Ki, Kd = gains
        dyn = lambda t, s: sprott_pid_ode(t, s, Kp, Ki, Kd)
    else:
        Kp, Ki, Kd, Kdd = gains
        dyn = lambda t, s: sprott_pidd_ode(t, s, Kp, Ki, Kd, Kdd)

    try:
        sol = solve_ivp(
            dyn, (0, T_END), [0, 0, 0, 0],
            method="Radau", t_eval=TIME_GRID,
            rtol=1e-6, atol=1e-8
        )
        if not sol.success or np.isnan(sol.y).any():
            raise ValueError

        x, y, z, I = sol.y
        e = -x
        de = -y

        if controller == "PID":
            u = gains[0]*e + gains[1]*I + gains[2]*de
        else:
            u = gains[0]*e + gains[1]*I + gains[2]*de + gains[3]*(-z)

        return x, u, sol.t

    except Exception:
        return None, None, None


def cost_function(controller, g):
    x, u, t = simulate_system(controller, g)
    if x is None:
        return 1e9
    J = alpha*np.trapz(t*np.abs(x), t) + beta*np.trapz(u*u, t)
    return np.inf if np.isnan(J) else J


def make_objective(controller):
    def obj(x):
        return cost_function(controller, x)
    return obj


# ============================================================
# OPTIMIZATION FOR ONE RUN
# ============================================================
def optimize_single_run(controller, algo, run_id):

    np.random.seed(GLOBAL_SEED + run_id)
    random.seed(GLOBAL_SEED + run_id)

    if controller == "PID":
        dim = 3
        lb = [PID_BOUNDS[0]] * dim
        ub = [PID_BOUNDS[1]] * dim
    else:
        dim = 4
        lb = [PIDD_BOUNDS[0]] * dim
        ub = [PIDD_BOUNDS[1]] * dim

    bounds = [FloatVar(lb[i], ub[i]) for i in range(dim)]

    problem = Problem(bounds=bounds, minmax="min",
                      obj_func=make_objective(controller))

    if algo == "PSO":
        model = OriginalPSO(epoch=EPOCHS, pop_size=POP_SIZE)
    else:
        model = OriginalDE(epoch=EPOCHS, pop_size=POP_SIZE)

    t0 = time.time()

    try:
        result = model.solve(problem)
        J = result.target.fitness
        best_vec = result.solution  # <<< PRIDANE
    except Exception:
        J = 1e9
        best_vec = None

    elapsed = time.time() - t0
    return J, elapsed, best_vec


# ============================================================
# PLOTTING 
# ============================================================
def plot_system(filename, title, uncontrolled, ps, de):
    x_u, t = uncontrolled
    x_pso, _ = ps
    x_de, _ = de

    plt.figure(figsize=(10, 3))
    plt.plot(t, x_u, 'k', label="Uncontrolled")
    plt.plot(t, x_pso, 'b', label="PSO best")
    plt.plot(t, x_de, 'r', label="DE best")
    plt.title(title)
    plt.xlabel("Time [s]")
    plt.ylabel("y(t)")
    plt.grid(True, linewidth=0.5, alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, format='eps', dpi=300)
    plt.close()


# ============================================================
# MAIN SCRIPT
# ============================================================
if __name__ == "__main__":

    controllers = ["PID", "PIDD"]
    algorithms = ["PSO", "DE"]

    results = []
    best_gains = {
        ("PID","PSO"): (1e9, None),
        ("PID","DE"):  (1e9, None),
        ("PIDD","PSO"): (1e9, None),
        ("PIDD","DE"):  (1e9, None),
    }

    total_runs = len(controllers) * len(algorithms) * N_RUNS
    print(f"Total runs: {total_runs}")

    run_counter = 0

    for controller in controllers:
        for algo in algorithms:
            for i in range(N_RUNS):

                run_counter += 1
                J, elapsed, g = optimize_single_run(controller, algo, i)

                results.append((controller, algo, i, J, elapsed))

                # store best gains
                key = (controller, algo)
                if J < best_gains[key][0] and g is not None:
                    best_gains[key] = (J, g)

                print(f"[{run_counter:03d}/{total_runs}] "
                      f"{controller}/{algo}, run {i+1:02d} — "
                      f"J={J:.5g}, time={elapsed:.4f}s")

    # SAVE RUN RESULTS
    df = pd.DataFrame(results,
                      columns=["controller","algo","run","J","time_sec"])
    df.to_csv("all_runs.csv", index=False)
    print("\nSaved all_runs.csv")

    # SAVE BEST GAINS
    rows = []
    for (c,a),(J,g) in best_gains.items():
        rows.append([c, a] + list(g))
    bg = pd.DataFrame(rows)
    bg.to_csv("best_gains.csv", index=False, header=False)
    print("Saved best_gains.csv")

    # ====================================================
    # CREATE EPS PLOTS
    # ====================================================
    print("Generating EPS plots...")

    # uncontrolled sim
    xu, _, t = simulate_system("PID", [0,0,0])  # zero control → same for PID/PIDD

    # PID plots
    Jp, gp_pso = best_gains[("PID","PSO")]
    Jd, gp_de  = best_gains[("PID","DE")]

    x_pso, _, _ = simulate_system("PID", gp_pso)
    x_de, _, _  = simulate_system("PID", gp_de)

    plot_system("plot_PID.eps",
                "PID – Uncontrolled vs PSO vs DE",
                (xu, t), (x_pso, t), (x_de, t))

    # PIDD plots
    Jp2, gp_pso2 = best_gains[("PIDD","PSO")]
    Jd2, gp_de2  = best_gains[("PIDD","DE")]

    x_pso2, _, _ = simulate_system("PIDD", gp_pso2)
    x_de2, _, _  = simulate_system("PIDD", gp_de2)

    plot_system("plot_PIDD.eps",
                "PIDD – Uncontrolled vs PSO vs DE",
                (xu, t), (x_pso2, t), (x_de2, t))

    print("Generated: plot_PID.eps, plot_PIDD.eps")
    print("\nDONE.")
