import json
import subprocess
import sys

# Check if the index parameter is provided
if len(sys.argv) < 2:
    print("Usage: python3 multiseeds_batch.py <index>")
    sys.exit(1)

# The index parameter from the command line
index_param = sys.argv[1]

# File paths and script names
config_path = "config.json"
seed_output_path = "seed_output.txt"
seeds_script = "seed.py"
# main_script = "main_seeds_diff_holding.py"
main_script = "main_seeds.py"

# Define parameter combinations:
# (a) Extra combination: testing_interval = 7, start_trading_weight = 0
# first_combination = [(8, 0)]
first_combination = []
# (b) Other combinations: testing_intervals in [1, 6, 3] and start_trading_weights in [1, 50, 0.5, 20, 0.1, 5, 0]
testing_intervals = [(1, 20, 20, 0)]
start_trading_weights = [0]
other_combinations = [(ti, stw) for ti in testing_intervals for stw in start_trading_weights]
# Total combinations: 1 + (3*7) = 22
all_combinations = first_combination + other_combinations

# Step 1: Update config.json BEFORE executing seeds.py
try:
    with open(config_path, "r") as f:
        config = json.load(f)
    # Update parameters for seeds.py generation
    config["delay_interval_months"] = 159
    config["testing_interval_months"] = 12
    config["seed"] = 30
    config["index"] = index_param  # Set index parameter from command-line
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)
    print("Updated config.json before seeds.py execution.")
except Exception as e:
    print(f"Error updating config.json before seeds.py: {e}")
    sys.exit(1)

# Step 2: Execute seeds.py to generate seed_output.txt
try:
    print("Executing seeds.py to generate seed_output.txt...")
    subprocess.run(["python3", seeds_script], check=True)
except Exception as e:
    print(f"Error executing {seeds_script}: {e}")
    sys.exit(1)

# Step 3: Update config.json AFTER executing seeds.py
try:
    with open(config_path, "r") as f:
        config = json.load(f)
    config["delay_interval_months"] = 174
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)
    print("Updated config.json after seeds.py execution.")
except Exception as e:
    print(f"Error updating config.json after seeds.py: {e}")
    sys.exit(1)

# Step 4: Loop over all parameter combinations and execute main_seeds.py
for testing_interval, start_trading_weight in all_combinations:
    # Update configuration for the current combination
    config["testing_interval_days"] = testing_interval[0]
    config["loss2_interval_days"] = testing_interval[1]
    config["validation_days"] = testing_interval[2]
    config["keep_previous_pf"] = testing_interval[3]
    config["start_trading_weight"] = start_trading_weight
    config["output"] = f"2025_{testing_interval[0]}_{testing_interval[1]}_{testing_interval[2]}_{testing_interval[3]}.png"
    
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)
    
    print(f"Executing {main_script} with testing_interval={testing_interval}, start_trading_weight={start_trading_weight}...")
    subprocess.run(["python3.12", main_script], check=True)

# Step 5: Archive generated files (all files starting with '2025_' and seed_output.txt)
tar_command = f"tar -cf results_20230704_37_avg_5_{index_param}.tar 2025_* seed_output.txt"
print(f"Archiving generated files with command: {tar_command}")
subprocess.run(tar_command, shell=True, check=True)

# Step 6: Delete the generated files (adjust command if you prefer to move instead)
delete_command = "rm 2025_*"
print(f"Deleting generated files with command: {delete_command}")
subprocess.run(delete_command, shell=True, check=True)
