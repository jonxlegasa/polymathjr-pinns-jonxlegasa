using Dates

include("../utils/plugboard.jl")
using .Plugboard

s = Plugboard.Settings(1, 0, 1)

function setup_training_run(run_number::Int64, training_examples::Vector{Float64})
  """
  Creates a training run directory and output file with specified naming convention.
  Args:
      run_number: The training run number (will be zero-padded to 2 digits)
      training_examples: Array of natural numbers representing training examples for each model
  """
  # Create data directory if it doesn't exist
  data_dir = "data"
  if !isdir(data_dir)
    mkdir(data_dir)
    println("Created data directory: $data_dir")
  end
  # Format run number with zero padding (01, 02, 03, etc.)
  run_number_formatted = lpad(run_number, 2, '0')
  # Create training run directory
  training_run_dir = joinpath(data_dir, "training-run-$run_number_formatted")
  if !isdir(training_run_dir)
    mkdir(training_run_dir)
    println("Created training run directory: $training_run_dir")
  end

  # Generate output file with training run information
  output_file = joinpath(training_run_dir, "training_info.txt")

  # Get current date and time
  current_datetime = now()

  # Write training run information to file
  open(output_file, "w") do file
    println(file, "Training Run Information")
    println(file, "="^30)
    println(file, "Training Run Number: $run_number_formatted")
    println(file, "Training Examples per Model: $training_examples")
    println(file, "Training Run Commenced: $current_datetime")
    println(file, "="^30)
  end

  println("Training run $run_number_formatted setup complete!")
  println("Output file created: $output_file")

  return training_run_dir, output_file
end


#=
# This will generate a training set and a becnhmark dataset
#
=#
function generate_datasets()
end


function run_training_sequence(training_examples_array::Vector{Vector{Float64}})
  """
  Runs a sequence of training runs with different training example configurations.

  Args:
      training_examples_array: Array of arrays, where each inner array contains
                             the training examples for that particular training run
  """

  for (run_idx, training_examples) in enumerate(training_examples_array)
    println("\n" * "="^50)
    println("Starting Training Run $run_idx")
    println("="^50)

    # Setup the training run directory and files
    training_dir, info_file = setup_training_run(run_idx, training_examples)

    # Here you would add your training loop logic
    # For now, we'll just simulate with a comment
    println("Training examples for this run: $training_examples")
    println("Ready for training implementation...")

    # TODO: Add the training implementation for the PINN Here
    # for num_examples in training_examples
    # PINN training here
    # end
  end
end

# Example usage:
# Define training examples for multiple runs
example_training_runs = [
  [4.0, -5.0, 3.125, -0.6510416666666666, 0.03390842013888889, -0.0003532127097800926, 6.13216510034883e-7, -1.5208742808404835e-10, 4.715012031375507e-15, -1.624163646169363e-20], # training run #01
]

Plugboard.generate_random_ode_dataset(s)

# Uncomment to run the example
run_training_sequence(example_training_runs)
