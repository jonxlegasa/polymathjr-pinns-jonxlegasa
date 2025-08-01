using Dates
using JSON

include("../utils/plugboard.jl")
using .Plugboard

include("../scripts/PINN.jl")
using .PINN

function setup_training_run(run_number::Int64, batch_size::Any)
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
    println(file, "Training Examples per Model: $batch_size")
    println(file, "Training Run Commenced: $current_datetime")
    println(file, "="^30)
  end

  println("Training run $run_number_formatted setup complete!")
  println("Output file created: $output_file")

  return training_run_dir, output_file
end

#=
# Notes: 
#  We want each training run to follow these batch sizes [1, 10, 50, 100]
#   lets make this an array babyyy
=#
function run_training_sequence(batch_sizes::Array{Int})
  """
  Runs a sequence of training runs with different training example configurations.

  Args:
      training_examples_array: Array of arrays, where each inner array contains
                             the training examples for that particular training run
  """
  for (batch_index, k) in enumerate(batch_sizes)
    s::Settings = Plugboard.Settings(1, 0, k)
    println("Batch size:", k)

    println("Batch_sizes length", length(batch_sizes))

    Plugboard.generate_random_ode_dataset(s, batch_index) # training data
    # Plugboard.generate_random_ode_dataset(s, batch_index) # create the validation file


    # TODO: have a variable that defines dataset.json and benchmark.json

    dataset = JSON.parsefile("./data/dataset.json")

    # Create the training dirs
    run_number_formatted = lpad(batch_index, 2, '0')
    println("Beginining training run for:", batch_index)

    println("\n" * "="^50)
    println("Starting Training Run $run_number_formatted")
    println("="^50)

    setup_training_run(batch_index, k)

    test = dataset[run_number_formatted]
    println("Processing alpha matrix: $test")

    # Loop through each entry in the JSON object
    for (run_idx, (alpha_matrix_key, series_coeffs)) in enumerate(dataset)

      # println("Series coefficients: $series_coeffs")

      # Convert string key back to matrix
      alpha_matrix = eval(Meta.parse(alpha_matrix_key))

      settings = PINNSettings(64, 1234, ode_matrices, 500, 100)

      # Train the network
      p_trained, coeff_net, st = train_pinn(settings, data_dict)

      # Evaluate results
      a_learned, u_func = evaluate_solution(p_trained, coeff_net, st, sample_matrix)

      # TODO: Add the training implementation for the PINN Here
      # PINN training here using:
      # - alpha_matrix (converted from key)
      # - series_coeffs as target
      # - ASK VICTOR HOW TO IMPLEMENT THE PINN WITH THIS
      println("Ready for PINN training with this alpha matrix and series coefficients...")
    end
  end
end

batch = [1, 10, 100]

# Uncomment to run the example
run_training_sequence(batch)
