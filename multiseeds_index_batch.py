import json
import subprocess
import sys

# List of indices to process
indices = ["D21"] 
weights = [(0.03, 0.3, 0.5, 0, 0)]
# weights = [(0.03, 0.3, 0.5, 0, 0)]

# Path to the config.json file
config_path = "config.json"

# Step 1: Update config.json with specified data and etf values
try:
    with open(config_path, "r") as f:
        config = json.load(f)
except Exception as e:
    print(f"Error reading {config_path}: {e}")
    sys.exit(1)

config["data"] = "all_41_v1.csv"
config["etf"] = "ETF_41_v1.csv"
config["market_time"] = "marketTime_41_v2.csv"
config["holding"] = "holding_20250627_w.csv"

# Write the updated config back to file
try:
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)
    print("Updated config.json with data and etf values.")
except Exception as e:
    print(f"Error writing {config_path}: {e}")
    sys.exit(1)

# Step 2: Iterate over each index and execute the main script
for weight in weights:
  reg_weight = weight[0]
  reg_ratio = weight[1]
  loss2_weight = weight[2]
  SR_weight = weight[3]
  diff_weight = weight[4]
  config["reg_weight"] = reg_weight
  config["reg_ratio"] = reg_ratio
  config["loss2_weight"] = loss2_weight
  config["SR_weight"] = SR_weight
  config["diff_weight"] = diff_weight
  for index_value in indices:
      # Update the index in the configuration
      config["index"] = index_value
      try:
          with open(config_path, "w") as f:
              json.dump(config, f, indent=4)
          print(f"Updated config.json with index: {index_value}")
      except Exception as e:
          print(f"Error writing config.json with index {index_value}: {e}")
          continue
  
      # Execute the main script with the current index as a command-line argument
      command = ["python3", "multiseeds_batch.py", index_value]
      print(f"Executing command: {' '.join(command)}")
      try:
          subprocess.run(command, check=True)
      except subprocess.CalledProcessError as e:
          print(f"Error executing command {command}: {e}")
  
  tar_command = f"tar -cf all_results_{reg_weight}_{reg_ratio}_{loss2_weight}_{SR_weight}_{diff_weight}.tar results_*_avg_5_*.tar"
  print(f"Archiving generated files with command: {tar_command}")
  subprocess.run(tar_command, shell=True, check=True)
  
  delete_command = "rm results_*_avg_5_*.tar"
  print(f"Deleting generated files with command: {delete_command}")
  subprocess.run(delete_command, shell=True, check=True)
