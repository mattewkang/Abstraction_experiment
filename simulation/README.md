# MATLAB simulations behind Figures 3 and 4 and data S1

Numerical validation of the eigen-direction (neighborhood) partition and of the
approximately bisimilar finite abstraction reported in the manuscript. Tested with
MATLAB R2024b; only base MATLAB is required (`polyshape` and `voronoin` are built in).

| Script | Role |
| --- | --- |
| `bisim_setup.m` | Builds the 3-D example system (one real eigenvalue, one complex pair, `rho(A) = 0.6`, non-normal); the `shear` argument controls the eigenvector conditioning `M`. |
| `bisim_grid.m` | Eigen-separable seed set for a covering radius `eps`, with its covering-radius bookkeeping. |
| `bisim_project.m` | Projection `P_xi` of a continuous state onto the current layer of the abstraction. |
| `neighborhood_figs.m` | Figure 3 (`nb_A.eps` ... `nb_F.eps`): partition, seeds and sampled states for three 2-D dynamics at `k = 0` and `k = 1`; appends the `Fig3*` sheets to `data_S1.xlsx`. Random seed `rng(1)`. |
| `sr_error_figs.m` | Figure 4 (`bisim_partition.eps`, `bisim_truncation.eps`, `bisim_tradeoff.eps`, `bisim_conditioning.eps`): partition error, truncation error, precision/complexity trade-off and conditioning sweep; creates `data_S1.xlsx` with its `README`, `System_parameters`, `Text_index_preservation` and `Fig4*` sheets. Random seed `rng(0)`. |

## Reproducing the figures and data S1

Both figure scripts write into `outdir = fullfile('..','Science Robotics submission')`,
a sibling directory of `simulation/`. Create it (or edit `outdir`) before running, then
run the scripts in this order so that the workbook exists before the Figure 3 sheets are
appended:

```matlab
cd simulation
mkdir('../Science Robotics submission');   % once
sr_error_figs        % Figure 4, creates data_S1.xlsx
neighborhood_figs    % Figure 3, appends to data_S1.xlsx
```

The random seeds are fixed, so the EPS figures and every value in `data_S1.xlsx` are
reproduced exactly. The full run takes a few minutes; the index-preservation check in
`sr_error_figs.m` (20000 trajectories of 25 steps) dominates the runtime.
