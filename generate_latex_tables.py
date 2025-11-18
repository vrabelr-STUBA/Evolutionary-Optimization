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

# Load statistics
summary = pd.read_csv("summary_statistics.csv")
psi = pd.read_csv("psi_statistics.csv")

# ============================================================
#   LATEX SUMMARY TABLE (Table 2)
# ============================================================
latex_summary = r"""
\begin{table}[t]
\centering
\caption{Summary statistics over $R=50$ runs: final objective values, runtime, and robustness score.}
\label{tab:summary}
\begin{tabular}{l l r r r r r r}
\hline
Controller & Algorithm &
Mean $J$ & Median $J$ & Std $J$ &
Mean time [s] & Std time [s] & RS \\
\hline
"""

for _, row in summary.iterrows():
    latex_summary += (
        f"{row['controller']} & {row['algorithm']} & "
        f"{row['mean_J']:.4g} & {row['median_J']:.4g} & {row['std_J']:.4g} & "
        f"{row['mean_time']:.4g} & {row['std_time']:.4g} & {row['robustness']:.4g} \\\\\n"
    )

latex_summary += r"\hline" + "\n" + r"\end{tabular}\end{table}"


# ============================================================
#   LATEX PSI TABLE (Table 3)
# ============================================================
latex_psi = r"""
\begin{table}[t]
\centering
\caption{Empirical Pareto Superiority Index (PSI) comparing bivariate performance $(J,\tilde{E})$.}
\label{tab:psi}
\begin{tabular}{l r r}
\hline
Controller & PSO${}\succ{}$DE & DE${}\succ{}$PSO \\
\hline
"""

for _, row in psi.iterrows():
    latex_psi += (
        f"{row['controller']} & "
        f"{row['PSI_PSO_over_DE']:.4g} & "
        f"{row['PSI_DE_over_PSO']:.4g} \\\\\n"
    )

latex_psi += r"\hline" + "\n" + r"\end{tabular}\end{table}"


# Save output
with open("latex_summary_table.tex", "w") as f:
    f.write(latex_summary)

with open("latex_psi_table.tex", "w") as f:
    f.write(latex_psi)

print("Generated LaTeX files:")
print(" - latex_summary_table.tex")
print(" - latex_psi_table.tex")
