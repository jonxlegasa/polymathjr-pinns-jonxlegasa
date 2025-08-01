#=
This script solves a third-order ordinary differential equation (ODE) using a
Physics-Informed Neural Network (PINN) inspired method.
Created based on the example available here: https://docs.sciml.ai/NeuralPDE/stable/examples/3rd/

Instead of the neural network approximating the solution u(x) directly, it learns
the optimal coefficients of a truncated power series that solves the ODE.

The process involves:
1. Defining the ODE and its boundary conditions.
2. Setting up a neural network that outputs a vector of coefficients.
3. Creating a loss function that measures how poorly the power series (built from the NN's coefficients)
   satisfies the ODE and boundary conditions.
4. Using an optimization algorithm (Adam) to train the network's parameters to minimize this loss.
5. Plotting the results to see how well our solution approximates the true, analytic solution.
=#

# ---------------------------------------------------------------------------
# Step 1: Import necessary libraries
# ---------------------------------------------------------------------------
# You may need to install these packages first. In the Julia REPL, press `]` to enter Pkg mode, then run:
# add Lux, ModelingToolkit, NeuralPDE, Optimization, OptimizationOptimJL, OptimizationOptimisers, Zygote, ComponentArrays, Plots, ProgressMeter

module PINNSpecific

using Lux, ModelingToolkit
using Optimization, OptimizationOptimJL, OptimizationOptimisers
using Zygote
using ComponentArrays
import IntervalSets: Interval
using Plots, ProgressMeter
import Random
using TaylorSeries

# Ensure a "data" directory exists for saving plots.
isdir("data") || mkpath("data")

# Define the floating point type to use throughout the script (e.g., Float32).
# Using Float32 is standard for neural networks as it's computationally faster.
F = Float32
# F = Float64

# ---------------------------------------------------------------------------
# Step 2: Define the Mathematical Problem (The ODE)
# ---------------------------------------------------------------------------

# Using ModelingToolkit, we define the independent variable `x` and the dependent variable `u(x)`.
@parameters x
@variables u(..)

x_left = F(0.0)  # Left boundary of the domain
x_right = F(1.0) # Right boundary of the domain

# Define differential operators for convenience.
<<<<<<< HEAD
Dxxx = Differential(x)^3
Dx = Differential(x)

# Define the ordinary differential equation.
# Dxxx(u(x)) = cos(pi*x)
equation = Dxxx(u(x)) ~ cos(pi * x)

# Define the boundary conditions for the ODE.

# bcs is not used
#=
bcs = [u(0.0) ~ 0.0,        # u(x) at x=0 is 0
=======
# Dxxx = Differential(x)^3
Dx = Differential(x) # we are considering ay'+ ay = 0 with constant coefficients

# Define the ordinary differential equation.
# Dxxx(u(x)) = cos(pi*x)
# equation = Dxxx(u(x)) ~ cos(pi * x)

equation = Dx(u(x)) + u(x) ~ 0

# Define the boundary conditions for the ODE.
#= bcs = [u(0.0) ~ 0.0,        # u(x) at x=0 is 0
>>>>>>> 330a7c99eeeec47ec4f56d67678faaee2f93d485
  u(1.0) ~ cos(pi),   # u(x) at x=1 is -1
  Dx(u(1.0)) ~ 1.0]      # The first derivative u'(x) at x=1 is 1
=#

# Define the domain over which the ODE is valid.
domains = [x ∈ Interval(x_left, x_right)]

# For verification, we define the true, known analytic solution to the ODE.
# This will be used to calculate the error of our approximation.
analytic_sol_func(x) = (pi * x * (-x + (pi^2) * (2x - 3) + 1) - sin(pi * x)) / (pi^3)

# ---------------------------------------------------------------------------
# Step 3: Setup the Power Series and Neural Network
# ---------------------------------------------------------------------------

# We will approximate the solution u(x) with a truncated power series of degree N.
# u(x) ≈ a₀ + a₁x/1! + a₂x²/2! + ... + aₙxⁿ/N!
N = 10 # The degree of the highest power term in the series.

# Pre-calculate factorials (0!, 1!, ..., N!) for use in the series.
fact = factorial.(0:N)


num_supervised = 5 # The number of coefficients we will supervise during training.
supervised_weight = F(1.0)  # Weight for the supervised loss term in the total loss function.


#= This is where I have to replace the approximation to the ODE with the
coefficients generated from the plugboardmethod =#
# The true coefficients a_n are the n-th derivatives of the analytic solution at x=0.
# We approximate them using TaylorSeries.jl.
t = Taylor1(F, N)
taylor_expansion = analytic_sol_func(t)
a_true = taylor_expansion.coeffs .* fact
<<<<<<< HEAD

training_data = a_true[1:num_supervised]


=======
>>>>>>> 330a7c99eeeec47ec4f56d67678faaee2f93d485

training_data = a_true[1:num_supervised] # replace this with the plugboard coefficients

# Create a set of points inside the domain to enforce the ODE. These are called "collocation points".
num_points = 1000
xs = range(x_left, x_right, length=num_points)

# Define a weight for the boundary condition part of the loss. This is a hyperparameter
# that helps balance the importance of satisfying the ODE vs. the boundary conditions.
bc_weight = F(100.0)

# Define the neural network architecture using Lux.
# It takes one dummy input and outputs N+1 values, which will be our coefficients a₀ to aₙ.
# coeff_net = Lux.Chain(
#   Lux.Dense(1, 64, σ), # Hidden layer with 100 neurons and sigmoid activation.
#   Lux.Dense(64, N+1)             # Output layer with N+1 neurons (one for each coefficient).
# )
coeff_net = Lux.Chain(
  Lux.Dense(1, 64, σ), # Hidden layer with 64 neurons and tanh activation.
  # Lux.Dense(64, 64, σ), # Hidden layer with 64 neurons and tanh activation.
  # Lux.Dense(64, 64, σ), # Hidden layer with 64 neurons and tanh activation.
  Lux.Dense(64, N + 1)      # Output layer with N+1 neurons (one for each coefficient).
)

# Initialize the network's parameters (p) and state (st).
rng = Random.default_rng()
Random.seed!(rng, 1234) # Seeding for reproducibility.
p_init, st = Lux.setup(rng, coeff_net)

# Wrap the initial parameters in a `ComponentArray`. This makes the nested `NamedTuple` of
# parameters behave like a standard vector, making it compatible with the optimizer.
p_init_ca = ComponentArray(p_init)


# ---------------------------------------------------------------------------
# Step 4: Define the Loss Function
# ---------------------------------------------------------------------------

# The loss function measures how "wrong" our current approximation is. The optimizer's
# job is to find the network parameters `p_net` that minimize this function.
function loss_fn(p_net, _)
  # First, run the network to get the current vector of power series coefficients.
  # The network takes a dummy input [0.0] and outputs our N+1 coefficients.
  a_vec = first(coeff_net([x_left], p_net, st))[:, 1]

  # Define the approximate solution and its derivatives using the coefficients from the network.
  # This is the power series representation: u(x) = Σ aᵢ * x^(i-1) / (i-1)!
  u_approx(x) = sum(a_vec[i] * x^(i - 1) / fact[i] for i in 1:N+1)
  Du_approx(x) = sum(a_vec[i] * x^(i - 2) / fact[i-1] for i in 2:N+1) # First derivative
  D3u_approx(x) = sum(a_vec[i] * x^(i - 4) / fact[i-3] for i in 4:N+1) # Third derivative

  # Calculate the loss from the ODE itself (the PDE loss, though it's an ODE).
  # This is the mean squared error between the two sides of our equation over all collocation points.
  loss_pde = sum(abs2, D3u_approx(xi) - cos(F(pi) * xi) for xi in xs) / num_points

  # Calculate the loss from the boundary conditions.
  # This is the sum of squared errors for each boundary condition.
<<<<<<< HEAD
  loss_bc = abs2(u_approx(x_left) - F(0.0)) + abs2(u_approx(x_right) - F(0) + abs2(Du_approx(x_right) - F(1.0))
=======
  loss_bc = abs2(u_approx(x_left) - F(0.0)) +
            abs2(u_approx(x_right) - F(cos(pi))) +
            abs2(Du_approx(x_right) - F(1.0))
>>>>>>> 330a7c99eeeec47ec4f56d67678faaee2f93d485


  loss_supervised = sum(abs2, a_vec[1:num_supervised] - training_data) / num_supervised
  # loss_supervised = 0.0 # This is a placeholder for any supervised loss, if needed.
  # The total loss is a weighted sum of the two components.
  return loss_pde + bc_weight * loss_bc + supervised_weight * loss_supervised
end


# ---------------------------------------------------------------------------
# Step 5: Train the Neural Network
# ---------------------------------------------------------------------------

maxiters = 500 # The total number of training iterations.
p_bar = Progress(maxiters, desc="Training coefficient net...") # The progress bar.

# The callback function is called after each optimization step.
# We use it to update our progress bar.
iter_count = 0
callback = function (p, l)
  global iter_count += 1
  global last_shown_values

  # If the current iteration is a milestone (1, 100, 200, etc.),
  # update the values that we want to display.
  if iter_count % 100 == 0 || iter_count == 1
    last_shown_values = [(:iter, iter_count), (:loss, l)]
    # Also, ensure the very last iteration's values are shown.
  elseif iter_count == maxiters
    last_shown_values = [(:iter, iter_count), (:loss, l)]
  end

  # On every single step, advance the progress bar, but always display
  # the stored "last_shown_values". This makes the text static between milestones.
  ProgressMeter.next!(p_bar; showvalues=last_shown_values)

  return false # Return false to continue the optimization.
end

# Define the optimization problem. We specify the loss function, the initial parameters,
# and the automatic differentiation backend (Zygote), which calculates the gradients.
adtype = Optimization.AutoZygote()
optfun = OptimizationFunction(loss_fn, adtype)
prob = OptimizationProblem(optfun, p_init_ca)

# Now, we solve the problem using the Adam optimizer.
res = solve(prob,
  OptimizationOptimisers.Adam(F(1e-3)); # Adam optimizer with a learning rate of 0.001
  callback=callback,
  maxiters=maxiters)



# ---------------- stage 2 : LBFGS ----------------
maxiters_lbfgs = 100
p_bar2 = Progress(maxiters_lbfgs, desc="LBFGS fine-tune... ")
iter2 = 0
cb2 = function (_, l)
  global iter2 += 1
  ProgressMeter.next!(p_bar2; showvalues=[(:iter, iter2), (:loss, l)])
  return false
end

prob2 = remake(prob; u0=res.u)
res = solve(prob2,
  OptimizationOptimJL.LBFGS();
  callback=cb2,
  maxiters=maxiters_lbfgs)

# Extract the final, trained parameters from the result.
p_trained = res.u
# Run the network one last time with the trained parameters to get the final coefficients.
a_learned = first(coeff_net([F(0.0)], p_trained, st))[:, 1]


# ---------------------------------------------------------------------------
# Step 6: Analyze and Plot the Results
# ---------------------------------------------------------------------------

println("\nTraining complete.")
println("Learned coefficients:")
display(a_learned)

# --- Plot 1: Compare learned solution vs. analytic solution ---

# Create a function for our predicted solution using the learned coefficients.
u_predict_func(x) = sum(a_learned[i] * x^(i - 1) / fact[i] for i in 1:N+1)

# Generate a dense set of points for smooth plotting.
x_plot = x_left:F(0.01):x_right
u_real = analytic_sol_func.(x_plot)
u_predict = u_predict_func.(x_plot)

plot_compare = plot(x_plot, u_real, label="Analytic Solution", linestyle=:dash, linewidth=3)
plot!(plot_compare, x_plot, u_predict, label="PINN Power Series", linewidth=2)
title!(plot_compare, "ODE Solution Comparison")
xlabel!(plot_compare, "x")
ylabel!(plot_compare, "u(x)")
savefig(plot_compare, "data/solution_comparison.png")

# --- Plot 2: Plot the absolute error of the solution ---

error = max.(abs.(u_real .- u_predict), F(1e-20))
plot_error = plot(x_plot, error,
  title="Absolute Error of Power Series Solution",
  label="|Analytic - Predicted|",
  yscale=:log10, # Use a logarithmic scale for the y-axis
  xlabel="x",
  ylabel="Error",
  linewidth=2)
savefig(plot_error, "data/error.png")

# --- Plot 3: Plot the error of the learned coefficients ---



# Calculate the absolute error between the true and learned coefficients.
# Clamp the error to a minimum value to avoid log(0) issues.
coeff_error = max.(abs.(a_true - a_learned), F(1e-20))

plot_coeffs = plot(0:N, coeff_error,
  yscale=:log10, # Use a logarithmic scale for the y-axis
  title="Error in Learned Coefficients (log scale)",
  xlabel="Coefficient Index",
  ylabel="Absolute Error",
  label="Coefficient Error",
  legend=:topright)
savefig(plot_coeffs, "data/coefficient_error.png")


println("\nPlots saved to 'data' directory.")
println("- solution_comparison.png")
println("- error.png")
println("- coefficient_error.png")
<<<<<<< HEAD
=======


end
>>>>>>> 330a7c99eeeec47ec4f56d67678faaee2f93d485
