# PINN Power Series ODE Solver

A Julia implementation of a Physics-Informed Neural Network (PINN) that approximates the solution of a given ODE via a truncated power series.

| Folder | What’s inside |
|--------|---------------|
| `src/` | (currently empty) – This folder is reserved for future development where core functions will be abstracted into a formal module. |
| `scripts/` | **`PINN_specific.jl`** – the main driver script. It defines the ODE and its boundary conditions, builds the neural network model, trains it to find the truncated power series, and generates all output plots and GIFs. |
| `data/` | output goes here (`.jld2`, `.png`, `.gif`) |

## Quick start
## Quick start (fresh machine)

```bash
# 1. grab the code
git clone https://github.com/cvictor2/polymathjr-PINNs.git
cd PINNs

# 2. download the dependencies once
julia --project=. -e 'using Pkg; Pkg.instantiate()'

# 3. run the demo simulation (saves data & figs to ./data)
julia --project=. scripts/PINN_specific.jl
