import json
import subprocess

config_path = "config.json"
seed_output_path = "seed_output.txt"
main_script = "predict.py"


def get_valid_seeds(seed_output_path):
    valid_seeds = []
    with open(seed_output_path, "r") as f:
        lines = f.readlines()[1:]  # Skip the header line
        for line in lines[:10]:  # Take the first 10 lines
            seed, _ = line.strip().split(",")  # Extract the seed value, ignore the rest
            valid_seeds.append(int(seed))  # Append the seed as an integer to the list
    return valid_seeds


def batch_execute(config_path, main_script, valid_seeds):
    with open(config_path, "r") as f:
        config = json.load(f)

    for seed in valid_seeds:
        config["seed"] = seed
        config["output"] = f"2025_n_{seed}.png"

        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)

        print(f"Executing {main_script} with seed: {seed}")
        
        subprocess.run(["python3.8", main_script], check=True)


if __name__ == "__main__":
    valid_seeds = get_valid_seeds(seed_output_path)
    if valid_seeds:
        batch_execute(config_path, main_script, valid_seeds)
    else:
        print("No valid seeds found with Validation Loss < 1.")
