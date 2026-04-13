import os
import shutil
import subprocess
import yaml

CONFIG_PATH = "configs/default.yaml"
OUTPUTS_DIR = "outputs"
RESULTS_DIR = "results"

scenarios = [
    {
        "id": "scenario1",
        "target": "Dengue_fever_rates",
        "cases_col": "Dengue_fever_cases",
        "compute_rate_per100k": False,
        "include_other_diseases_as_features": False
    },
    {
        "id": "scenario2",
        "target": "Dengue_fever_rates",
        "cases_col": "Dengue_fever_cases",
        "compute_rate_per100k": False,
        "include_other_diseases_as_features": True
    },
    {
        "id": "scenario3",
        "target": "Dengue_fever_rates",
        "cases_col": "Dengue_fever_cases",
        "compute_rate_per100k": True,
        "include_other_diseases_as_features": False
    },    
    {
        "id": "scenario4",
        "target": "Influenza_rates",
        "cases_col": "Influenza_cases",
        "compute_rate_per100k": False,
        "include_other_diseases_as_features": False
    },
    {
        "id": "scenario5",
        "target": "Diarrhoea_rates",
        "cases_col": "Diarrhoea_cases",
        "compute_rate_per100k": False,
        "include_other_diseases_as_features": False
    }    
]

def update_config(scenario):
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
    
    config["experiment"]["target"] = scenario["target"]
    config["experiment"]["cases_col"] = scenario["cases_col"]
    config["experiment"]["compute_rate_per100k"] = scenario["compute_rate_per100k"]
    config["experiment"]["include_other_diseases_as_features"] = scenario["include_other_diseases_as_features"]
    
    with open(CONFIG_PATH, "w") as f:
        yaml.dump(config, f)

def run_pipeline():
    subprocess.run(["python", "run_all.py", "--config", CONFIG_PATH], check=True)

def copy_results(scenario_id):
    dest_dir = os.path.join(RESULTS_DIR, scenario_id, OUTPUTS_DIR)
    if os.path.exists(dest_dir):
        shutil.rmtree(dest_dir)
    os.makedirs(os.path.join(RESULTS_DIR, scenario_id), exist_ok=True)
    shutil.copytree(OUTPUTS_DIR, dest_dir)

if __name__ == "__main__":
    # # Clean up old scenario5 if it exists to avoid confusion
    # if os.path.exists(os.path.join(RESULTS_DIR, "scenario5")):
    #     try:
    #         shutil.rmtree(os.path.join(RESULTS_DIR, "scenario5"))
    #     except:
    #         pass

    for scenario in scenarios:
        print(f"--- Running {scenario['id']} ---")
        update_config(scenario)
        try:
            run_pipeline()
            copy_results(scenario["id"])
            print(f"--- Successfully completed {scenario['id']} ---")
        except Exception as e:
            print(f"--- Failed to run {scenario['id']}: {e} ---")
