#=

This is the general script for the PINN.
This will be agnostic to architecture and size of the neural network (right now it is only for feedforeward)

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

module PINN

using Lux, ModelingToolkit
using Optimization, OptimizationOptimJL, OptimizationOptimisers
using Zygote
using ComponentArrays
import IntervalSets: Interval
using Plots, ProgressMeter
import Random
using TaylorSeries

include("../utils/ProgressBar.jl")
using .ProgressBar

# Ensure a "data" directory exists for saving plots.
isdir("data") || mkpath("data")

# Define the floating point type to use throughout the script (e.g., Float32).
# Using Float32 is standard for neural networks as it's computationally faster.
F = Float32

# ---------------------------------------------------------------------------
# Step 2: Define the PINN Settings Structure
# ---------------------------------------------------------------------------

struct PINNSettings
  neuron_num::Int
  seed::Int
  ode_matrix::Array
  maxiters_adam::Int
  maxiters_lbfgs::Int
end

# ---------------------------------------------------------------------------
# Step 3: Define the Mathematical Problem (The ODE)
# ---------------------------------------------------------------------------

# Using ModelingToolkit, we define the independent variable `x` and the dependent variable `u(x)`.
@parameters x
@variables u(..)

# Domain boundaries
x_left = F(0.0)  # Left boundary of the domain
x_right = F(1.0) # Right boundary of the domain

# Differential operator
Dx = Differential(x)

# Define the ordinary differential equation: u'(x) + u(x) = 0
equation = Dx(u(x)) + u(x) ~ 0

# Define the domain over which the ODE is valid.
domains = [x ∈ Interval(x_left, x_right)]

# ---------------------------------------------------------------------------
# Step 4: Setup the Power Series and Neural Network
# ---------------------------------------------------------------------------

# We will approximate the solution u(x) with a truncated power series of degree N.
N = 10 # The degree of the highest power term in the series.

# Pre-calculate factorials (0!, 1!, ..., N!) for use in the series.
fact = factorial.(0:N)

num_supervised = 5 # The number of coefficients we will supervise during training.
supervised_weight = F(1.0)  # Weight for the supervised loss term in the total loss function.

# Create a set of points inside the domain to enforce the ODE. These are called "collocation points".
num_points = 1000
xs = range(x_left, x_right, length=num_points)

# Define a weight for the boundary condition part of the loss.
bc_weight = F(100.0)

# ---------------------------------------------------------------------------
# Step 5: Initialize Neural Network with Settings
# ---------------------------------------------------------------------------

function initialize_network(settings::PINNSettings)
  # Find the maximum matrix dimensions for input layer size
  max_input_size = maximum(prod(size(matrix)) for matrix in settings.ode_matrix) # I think this might be used in src/main.jl

  # Define the neural network architecture using the settings
  coeff_net = Lux.Chain(
    Lux.Dense(max_input_size, settings.neuron_num, σ), # Hidden layer with configurable neurons
    Lux.Dense(settings.neuron_num, N + 1)              # Output layer with N+1 coefficients
  )

  # Initialize the network's parameters with the specified seed
  rng = Random.default_rng()
  Random.seed!(rng, settings.seed)
  p_init, st = Lux.setup(rng, coeff_net)

  # Wrap the initial parameters in a ComponentArray
  p_init_ca = ComponentArray(p_init)

  return coeff_net, p_init_ca, st
end

# ---------------------------------------------------------------------------
# Step 6: Define the Loss Function
# ---------------------------------------------------------------------------

function loss_fn(p_net, data, coeff_net, st, ode_matrix_flat)
  # Run the network to get the current vector of power series coefficients
  a_vec = first(coeff_net(ode_matrix_flat, p_net, st))[:, 1]

  # Define the approximate solution and its derivatives using the coefficients
  u_approx(x) = sum(a_vec[i] * x^(i - 1) / fact[i] for i in 1:N+1)
  Du_approx(x) = sum(a_vec[i] * x^(i - 2) / fact[i-1] for i in 2:N+1) # First derivative

  # Calculate the loss from the ODE itself
  loss_pde = sum(abs2, Du_approx(xi) + u_approx(xi) - 0 for xi in xs) / num_points

  # Calculate the loss from the boundary conditions
  loss_bc = abs2(u_approx(x_left) - F(0.0))

  # Calculate supervised loss using the plugboard coefficients
  loss_supervised = sum(abs2, a_vec[1:num_supervised] - data) / num_supervised

  # The total loss is a weighted sum of the components
  return loss_pde + bc_weight * loss_bc + supervised_weight * loss_supervised
end

# ---------------------------------------------------------------------------
# Step 7: Global Loss Function
# ---------------------------------------------------------------------------

function global_loss(p_net, settings::PINNSettings, data_dict, coeff_net, st)
  total_loss = F(0.0)

  for matrix in settings.ode_matrix
    # Process each matrix at its natural size
    matrix_flat = reshape(matrix, :, 1)  # Column vector for network
    local_loss = loss_fn(p_net, data_dict[matrix], coeff_net, st, matrix_flat)
    total_loss += local_loss
  end

  return total_loss
end

# ---------------------------------------------------------------------------
# Step 8: Training Function
# ---------------------------------------------------------------------------

function train_pinn(settings::PINNSettings, data_dict)
  # Initialize network
  coeff_net, p_init_ca, st = initialize_network(settings)

  # Create wrapper function for optimization
  function loss_wrapper(p_net, _)
    return global_loss(p_net, settings, data_dict, coeff_net, st)
  end

  # ---------------- Stage 1: ADAM ----------------
  println("Starting Adam training...")
  p_one = ProgressBar.ProgressBarSettings(settings.maxiters_adam, "Adam Training...")
  callback_one = ProgressBar.Bar(p_one)

  # Define the optimization problem
  adtype = Optimization.AutoZygote()
  optfun = OptimizationFunction(loss_wrapper, adtype)
  prob = OptimizationProblem(optfun, p_init_ca)

  # Solve with Adam optimizer
  res = solve(prob,
    OptimizationOptimisers.Adam(F(1e-3));
    callback=callback_one,
    maxiters=settings.maxiters_adam)

  # ---------------- Stage 2: LBFGS ----------------
  println("Starting LBFGS fine-tuning...")
  p_two = ProgressBar.ProgressBarSettings(settings.maxiters_lbfgs, "LBFGS fine-tune...")
  callback_two = ProgressBar.Bar(p_two)

  prob2 = remake(prob; u0=res.u)
  res = solve(prob2,
    OptimizationOptimJL.LBFGS();
    callback=callback_two,
    maxiters=settings.maxiters_lbfgs)

  # Extract final trained parameters
  p_trained = res.u

  println("\nTraining complete.")

  return p_trained, coeff_net, st
end

# ---------------------------------------------------------------------------
# Step 9: Evaluation and Analysis Functions
# ---------------------------------------------------------------------------

# This code is the true solution for the ODE
# analytic_sol_func(x) = (pi * x * (-x + (pi^2) * (2x - 3) + 1) - sin(pi * x)) / (pi^3) # We replace with our training examples
# This is then represented as a TaylorSeries 

function evaluate_solution(p_trained, coeff_net, st, ode_matrix_sample, analytic_sol_func=nothing)
  # Get learned coefficients for a sample matrix
  max_input_size = prod(size(ode_matrix_sample)) # remove?
  matrix_flat = reshape(ode_matrix_sample, :, 1)

  a_learned = first(coeff_net(matrix_flat, p_trained, st))[:, 1]

  println("Learned coefficients:")
  display(a_learned)

  # Create prediction function
  u_predict_func(x) = sum(a_learned[i] * x^(i - 1) / fact[i] for i in 1:N+1)

  if analytic_sol_func !== nothing
    # Generate plotting points
    x_plot = x_left:F(0.01):x_right
    # It makes sense that this has to be replaced because this is used for plotting the error as well
    u_real = analytic_sol_func.(x_plot) # instead we have to make this be plugboard coefficients k
    u_predict = u_predict_func.(x_plot)

    # Plot comparison
    plot_compare = plot(x_plot, u_real, label="Analytic Solution", linestyle=:dash, linewidth=3)
    plot!(plot_compare, x_plot, u_predict, label="PINN Power Series", linewidth=2)
    title!(plot_compare, "ODE Solution Comparison")
    xlabel!(plot_compare, "x")
    ylabel!(plot_compare, "u(x)")
    savefig(plot_compare, "data/solution_comparison.png")

    # Plot error
    error = max.(abs.(u_real .- u_predict), F(1e-20))
    plot_error = plot(x_plot, error,
      title="Absolute Error of Power Series Solution",
      label="|Analytic - Predicted|",
      yscale=:log10,
      xlabel="x",
      ylabel="Error",
      linewidth=2)
    savefig(plot_error, "data/error.png")

    println("\nPlots saved to 'data' directory.")
    println("- solution_comparison.png")
    println("- error.png")
  end

  return a_learned, u_predict_func
end

# ---------------------------------------------------------------------------
# Step 10: Export Functions
# ---------------------------------------------------------------------------

export PINNSettings, train_pinn, global_loss, evaluate_solution, initialize_network

end
