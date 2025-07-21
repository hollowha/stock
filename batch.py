import json
import subprocess

# Define file paths and script name
config_path = "config.json"
seed_output_path = "seed_output.txt"
main_script = "main.py"

# Define testing parameters
testing_intervals = [1, 6, 3]  # Different testing intervals in months
# start_trading_weights = [0]  # Various starting trading weights
start_trading_weights = [1, 50, 0.5, 20, 0.1, 5, 0]  # Various starting trading weights

def get_valid_seeds(seed_output_path):
    valid_seeds = []
    with open(seed_output_path, "r") as f:
        lines = f.readlines()[1:]  # Skip the header line
        for line in lines[:5]:  # Take the first 10 lines
            seed, _ = line.strip().split(",")  # Extract the seed value, ignore the rest
            valid_seeds.append(int(seed))  # Append the seed as an integer to the list
    return valid_seeds


def batch_execute(config_path, main_script, valid_seeds, testing_intervals, start_trading_weights):
    """
    Execute the main script for all combinations of seeds, testing intervals, and start trading weights.
    :param config_path: Path to the configuration file.
    :param main_script: Name of the main script to execute.
    :param valid_seeds: List of valid seeds.
    :param testing_intervals: List of testing intervals.
    :param start_trading_weights: List of starting trading weights.
    """
    with open(config_path, "r") as f:
        config = json.load(f)

    for seed in valid_seeds:
        for testing_interval in testing_intervals:
            for weight in start_trading_weights:
                # Update configuration
                config["seed"] = seed
                config["testing_interval_months"] = testing_interval
                config["start_trading_weight"] = weight
                config["output"] = f"2025_{testing_interval}_{weight}_{seed}.png"

                # Write updated configuration back to the file
                with open(config_path, "w") as f:
                    json.dump(config, f, indent=4)

                # Execute the main script
                print(f"Executing {main_script} with seed={seed}, testing_interval={testing_interval}, start_trading_weight={weight}")
                subprocess.run(["python3.8", main_script], check=True)

if __name__ == "__main__":
    # Get valid seeds from the seed output file
    valid_seeds = get_valid_seeds(seed_output_path)
    if valid_seeds:
        # Run batch execution for all combinations
        batch_execute(config_path, main_script, valid_seeds, testing_intervals, start_trading_weights)
    else:
        print("No valid seeds found with Validation Loss < 1.")
