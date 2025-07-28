using Dates

include("../utils/plugboard.jl")
using .Plugboard

function setup_training_run(run_number::Int, training_examples::Vector{Int})
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

function run_training_sequence(training_examples_array::Vector{Vector{Int}})
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

    # TODO: Add your training implementation here
    # for num_examples in training_examples
    #     # Your training logic for each model with num_examples
    # end
  end
end

# Example usage:
# Define training examples for multiple runs
example_training_runs = [
  [100, 200, 300],     # Training run 01: 3 models with 100, 200, 300 examples
  [150, 250, 350, 450], # Training run 02: 4 models with different example counts
  [500, 1000]          # Training run 03: 2 models with larger datasets
]


expPlugboard.generate_random_ode_dataset()

# Uncomment to run the example
run_training_sequence(example_training_runs)
