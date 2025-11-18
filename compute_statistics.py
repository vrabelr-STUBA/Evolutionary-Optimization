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


import pandas as pd
import numpy as np

# -----------------------------
# Local bootstrap function 
# -----------------------------
def bootstrap_ci(data, n_bootstrap=2000, ci=0.95):
    """Compute bootstrap confidence interval for the mean."""
    data = np.array(data)
    boot_means = []
    n = len(data)

    for _ in range(n_bootstrap):
        sample = np.random.choice(data, n, replace=True)
        boot_means.append(sample.mean())

    lower = np.percentile(boot_means, (1 - ci) / 2 * 100)
    upper = np.percentile(boot_means, (1 + ci) / 2 * 100)
    return lower, upper


# ============================================================
# Load experimental results
# ============================================================
df = pd.read_csv("all_runs.csv")

controllers = df["controller"].unique()
algorithms = df["algo"].unique()

summary_rows = []
psi_rows = []

# ============================================================
# Compute J*, target thresholds
# ============================================================
J_star = {
    c: df[df["controller"] == c]["J"].min()
    for c in controllers
}

J_target = {c: J_star[c] * 1.01 for c in controllers}


# ============================================================
# Compute summary statistics and robustness
# ============================================================
for c in controllers:
    for a in algorithms:
        sub = df[(df["controller"] == c) & (df["algo"] == a)]
        Js = sub["J"].values
        times = sub["time_sec"].values

        RS = np.mean(Js <= J_target[c])

        summary_rows.append({
            "controller": c,
            "algorithm": a,
            "mean_J": np.mean(Js),
            "median_J": np.median(Js),
            "std_J": np.std(Js),
            "mean_time": np.mean(times),
            "std_time": np.std(times),
            "robustness": RS,
        })


# ============================================================
# Pareto Superiority Index (PSI)
# ============================================================
def dominates(j1, t1, j2, t2):
    """Check if (j1,t1) Pareto dominates (j2,t2)."""
    return (j1 <= j2 and t1 <= t2) and (j1 < j2 or t1 < t2)

for c in controllers:
    dfP = df[df["controller"] == c]

    pso = dfP[dfP["algo"] == "PSO"]
    de = dfP[dfP["algo"] == "DE"]

    count_pso_dom = 0
    count_de_dom = 0
    total = len(pso) * len(de)

    for _, r1 in pso.iterrows():
        for _, r2 in de.iterrows():
            if dominates(r1["J"], r1["time_sec"], r2["J"], r2["time_sec"]):
                count_pso_dom += 1
            if dominates(r2["J"], r2["time_sec"], r1["J"], r1["time_sec"]):
                count_de_dom += 1

    psi_rows.append({
        "controller": c,
        "PSI_PSO_over_DE": count_pso_dom / total,
        "PSI_DE_over_PSO": count_de_dom / total,
    })


# ============================================================
# Save outputs
# ============================================================
pd.DataFrame(summary_rows).to_csv("summary_statistics.csv", index=False)
pd.DataFrame(psi_rows).to_csv("psi_statistics.csv", index=False)

print("Statistics computed and saved.")
