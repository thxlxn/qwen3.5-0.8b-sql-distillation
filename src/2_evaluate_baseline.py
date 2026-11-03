import json
import requests
from utils import RESULTS_PATH, TEST_POOL_PATH, evaluate_model, print_eval_summary

# Test untrained student (0.8b) and teacher (9b)
MODELS_TO_TEST = ["qwen3.5:0.8b", "qwen3.5:9b"]


def main():
    with open(TEST_POOL_PATH, "r", encoding="utf-8") as f:
        test_pool = json.load(f)

    all_results = {}
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH, "r", encoding="utf-8") as f:
            all_results = json.load(f)

    session = requests.Session()
    for model_name in MODELS_TO_TEST:
        res = evaluate_model(model_name, test_pool, session)
        all_results[model_name] = res
        with open(RESULTS_PATH, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)

    print_eval_summary(all_results, title="Final Baseline Summary")


if __name__ == "__main__":
    main()