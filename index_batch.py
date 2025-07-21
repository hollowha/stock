import json
import subprocess

# Define file paths and scripts
config_path = "config.json"
seed_script = "seed.py"
seed_batch_script = "seed_batch.py"
batch_script = "batch.py"
tar_script = "tar.sh"

# Define index range (9999 to 9991)
# index_range = range(9999, 9990, -1)
index_range = ["0050", "130050"]

def update_config(index, delay_interval_months, testing_interval_months, seed):
    """
    Update the config.json file with given values for index, delay_interval_months, testing_interval_months, and seed.
    :param index: The current index to update.
    :param delay_interval_months: The delay interval in months.
    :param testing_interval_months: The testing interval in months.
    :param seed: The seed value.
    """
    with open(config_path, "r") as f:
        config = json.load(f)

    # Update the configuration
    config["index"] = str(index)
    config["delay_interval_months"] = delay_interval_months
    config["testing_interval_months"] = testing_interval_months
    config["seed"] = seed

    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)


def execute_script(script, *args):
    """
    Run a given script with optional arguments.
    :param script: The script to execute.
    :param args: The arguments for the script.
    """
    subprocess.run([script] + list(args), check=True)
    
def remove_directories_with_wildcard(pattern):
    """
    Remove directories matching a specific wildcard pattern using shell.
    :param pattern: The wildcard pattern to match directories.
    """
    command = f"rm -r {pattern}"
    print(f"Executing command: {command}")
    subprocess.run(command, shell=True, check=True)


def main():
    # Loop through the index range
    for index in index_range:
        print(f"Processing index: {index}")

        # Step 1: Update "index" in config.json
        update_config(index, 162, 12, 50)

        # Step 2: Execute seed.py
        print(f"Executing {seed_script}...")
        execute_script("python3", seed_script)

        # Step 3: Update delay_interval_months to 168 in config.json
        update_config(index, 174, 7, 50)

        # Step 4: Execute seed_batch.py
        print(f"Executing {seed_batch_script}...")
        execute_script("python3", seed_batch_script)

        # Step 5: Execute batch.py
        print(f"Executing {batch_script}...")
        execute_script("python3", batch_script)

        # Step 6: Execute bash tar.sh
        print(f"Executing {tar_script}...")
        execute_script("bash", tar_script, f"2025", f"results_2025_{index}")

        # Step 7: Remove 2024* directories
        print(f"Removing 2025* directories...")
        remove_directories_with_wildcard("2025*")

        print(f"Completed processing for index: {index}\n")


if __name__ == "__main__":
    main()
