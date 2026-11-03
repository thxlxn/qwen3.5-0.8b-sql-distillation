import json
import requests
from utils import RESULTS_PATH, TEST_POOL_PATH, evaluate_model, print_eval_summary

DISTILLED_MODEL = "qwen3.5-sql:0.8b"


def main():
    with open(TEST_POOL_PATH, "r", encoding="utf-8") as f:
        test_pool = json.load(f)

    all_results = {}
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH, "r", encoding="utf-8") as f:
            all_results = json.load(f)

    session = requests.Session()
    res = evaluate_model(DISTILLED_MODEL, test_pool, session)
    all_results[DISTILLED_MODEL] = res

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print_eval_summary(all_results, title="Distillation Benchmark Comparison")


if __name__ == "__main__":
    main()