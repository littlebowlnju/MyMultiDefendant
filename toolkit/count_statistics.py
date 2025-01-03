import os
import json
import jsonlines
from collections import defaultdict
from common import add_to_log

def count_defendant_num(dataset_file):
    """Count the total defendant num in the dataset file."""
    dataset_file_name = os.path.basename(dataset_file).removesuffix(".jsonl")
    log_file = f"/home/hwh/my_multi_defendant/logs/data_logs/{dataset_file_name}_statistics.log"
    total_defendant_num = 0
    with jsonlines.open(dataset_file, "r") as reader:
        for case in reader:
            total_defendant_num += len(case["defendants"])
    add_to_log(log_file, f"total defendant num: {total_defendant_num}")

def count_charge_distribution(dataset_file, dataset_type="pair"):
    """Calculate the number of occurrences of each charge in dataset
    Args:
        dataset_type: "pair" or "original"
    """
    charge_time = defaultdict(int)
    dataset_file_name = os.path.basename(dataset_file).removesuffix(".jsonl")
    log_file = f"/home/hwh/my_multi_defendant/logs/data_logs/{dataset_file_name}_charge_distribution.json"
    with jsonlines.open(dataset_file, "r") as reader:
        if dataset_type == "pair":
            judgment_keys = ["defendant1_outcome", "defendant2_outcome"]
            for obj in reader:
                for key in judgment_keys:
                    for c in obj[key]:
                        charge_time[c["standard_accusation"]] += 1
        else:
            for obj in reader:
                for outcome in obj["outcomes"]:
                    for c in outcome["judgment"]:
                        charge_time[c["standard_accusation"]] += 1
    charge_time = dict(sorted(charge_time.items(), key=lambda x: -x[1]))
    with open(log_file, "w") as f:
        json.dump(charge_time, f, ensure_ascii=False)

if __name__ == "__main__":
    # count_defendant_num("/home/hwh/my_multi_defendant/data/cmdl/train_smaller_with_id.jsonl")
    # count_charge_distribution("/home/hwh/my_multi_defendant/data/curated/train_smaller_defendant_pairs_all_full.jsonl", dataset_type="pair")
    count_charge_distribution("/home/hwh/my_multi_defendant/data/cmdl/train_smaller_with_id.jsonl", dataset_type="original")