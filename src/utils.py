import re
import sqlite3
import time
from pathlib import Path
import requests

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
TEST_POOL_PATH = DATA_DIR / "test_eval_pool.json"
RESULTS_PATH = DATA_DIR / "eval_results.json"

OLLAMA_URL = "http://localhost:11434/api/generate"

SYSTEM_PROMPT = (
    "You are a precise SQLite query generator. Given a database schema and a user question, "
    "output ONLY a valid SQLite query that answers the question. Do not include explanations."
)


def extract_sql(raw_text: str) -> str:
    match = re.search(r"```(?:sql)?\s*(.*?)\s*```", raw_text, re.DOTALL | re.IGNORECASE)
    sql = match.group(1).strip() if match else raw_text.strip()
    return sql.split(";")[0].strip() + ";"


def check_sql_execution(schema_ddl: str, sql_query: str) -> tuple[bool, str]:
    """Returns (is_valid, error_message_if_any)."""
    conn = sqlite3.connect(":memory:")
    try:
        conn.executescript(schema_ddl)
        conn.execute(sql_query)
        return True, ""
    except Exception as e:
        return False, str(e)
    finally:
        conn.close()


def evaluate_model(model_name: str, test_pool: list, session: requests.Session) -> dict:
    print(f"\n==========================================")
    print(f" Evaluating Model: {model_name} ({len(test_pool)} unseen test queries)")
    print(f"==========================================")

    # WARM-UP REQUEST
    # Forces Ollama to load the model into VRAM before the timer starts
    try:
        session.post(
            OLLAMA_URL,
            json={"model": model_name, "prompt": "warmup", "stream": False},
            timeout=30
        )
    except Exception:
        pass

    valid_sql_count = 0
    errors_breakdown = {"syntax_error": 0, "no_such_column": 0, "no_such_table": 0, "other_error": 0}
    latencies = []
    avg_accs = []
    avg_lats = []

    for idx, row in enumerate(test_pool, start=1):
        schema = row["context"]
        question = row["question"]
        prompt = f"{SYSTEM_PROMPT}\n\nSchema:\n{schema}\n\nQuestion: {question}\n"

        t0 = time.perf_counter()
        resp = session.post(
            OLLAMA_URL,
            json={
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "think": False,
                "options": {"temperature": 0.0, "num_ctx": 2048},
            },
            timeout=60,
        ).json()
        latency_ms = (time.perf_counter() - t0) * 1000
        latencies.append(latency_ms)

        generated_sql = extract_sql(resp.get("response", ""))
        is_valid, err_msg = check_sql_execution(schema, generated_sql)

        if is_valid:
            valid_sql_count += 1
        else:
            err_lower = err_msg.lower()
            if "syntax error" in err_lower:
                errors_breakdown["syntax_error"] += 1
            elif "no such column" in err_lower:
                errors_breakdown["no_such_column"] += 1
            elif "no such table" in err_lower:
                errors_breakdown["no_such_table"] += 1
            else:
                errors_breakdown["other_error"] += 1

        if idx % 25 == 0 or idx == len(test_pool):
            acc = (valid_sql_count / idx) * 100
            avg_lat = sum(latencies) / len(latencies)
            avg_accs.append(acc)
            avg_lats.append(avg_lat)
            print(f"[{idx}/{len(test_pool)}] Valid Execution Rate: {acc:.1f}% | Avg Latency: {avg_lat:.0f} ms")

    final_acc = round((valid_sql_count / len(test_pool)) * 100, 2)
    avg_latency = round(sum(latencies) / len(latencies), 1)

    return {
        "model": model_name,
        "total_queries": len(test_pool),
        "valid_sql_count": valid_sql_count,
        "execution_valid_pct": final_acc,
        "avg_latency_ms": avg_latency,
        "errors_breakdown": errors_breakdown,
        "cont_accs": avg_accs,
        "cont_lats": avg_lats
    }


def print_eval_summary(all_results: dict, title: str = "Final Evaluation Summary") -> None:
    print(f"\n--- {title} ---")
    for m, stats in all_results.items():
        print(
            f"{m:18} | Valid SQL: {stats['execution_valid_pct']:6.2f}% "
            f"({stats['valid_sql_count']}/{stats['total_queries']}) | "
            f"Avg Latency: {stats['avg_latency_ms']:6.1f} ms | Errors: {stats['errors_breakdown']}"
        )