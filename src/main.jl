using Dates
using JSON

include("../utils/plugboard.jl")
using .Plugboard

s = Plugboard.Settings(1, 0, 10)

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

function run_training_sequence()
  """
  Runs a sequence of training runs with different training example configurations.

  Args:
      training_examples_array: Array of arrays, where each inner array contains
                             the training examples for that particular training run
  """
  # Load the JSON data
  dataset = JSON.parsefile("./data/dataset.json")

  # Loop through each entry in the JSON object
  for (run_idx, (alpha_matrix_key, series_coeffs)) in enumerate(dataset)
    println("\n" * "="^50)
    println("Starting Training Run $run_idx")
    println("="^50)
    println("Processing alpha matrix: $alpha_matrix_key")
    println("Series coefficients: $series_coeffs")
    # Convert string key back to matrix
    alpha_matrix = eval(Meta.parse(alpha_matrix_key))
    println("Converted alpha matrix: ", alpha_matrix)
    # TODO: Add the training implementation for the PINN Here
    # PINN training here using:
    # - alpha_matrix (converted from key)
    # - series_coeffs as target
    println("Ready for PINN training with this alpha matrix and series coefficients...")
  end
end

Plugboard.generate_random_ode_dataset(s)

# Uncomment to run the example
run_training_sequence()
