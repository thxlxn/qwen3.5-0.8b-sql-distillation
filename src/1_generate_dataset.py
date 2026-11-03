import json
import re
import sqlite3
import requests
from pathlib import Path

# --- CONSTANTS ---
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
INPUT_POOL_PATH = DATA_DIR / "train_seed_pool.json"
OUTPUT_JSONL_PATH =  DATA_DIR / "distilled_train.jsonl"

OLLAMA_URL = "http://localhost:11434/api/generate"
TEACHER_MODEL = "qwen3.5:9b"

SYSTEM_PROMPT = ("You are a precise SQLite query generator. Given a database schema and a user question, "
                "output ONLY a valid SQLite query that answers the question. Do not include explanations.")

def extract_sql(raw_text: str) -> str:
    """Extracts raw SQL from Markdown fences or plain text."""
    match = re.search(r"```(?:sql)?\s*(.*?)\s*```", raw_text, re.DOTALL | re.IGNORECASE)
    sql = match.group(1).strip() if match else raw_text.strip()
    # Keep only the first statement and ensure clean formatting
    return sql.split(";")[0].strip() + ";"


def verify_sql_in_sqlite(schema_ddl: str, sql_query: str) -> bool:
    """Compiles the schema and executes the query in an in-memory SQLite DB."""
    conn = sqlite3.connect(":memory:")
    try:
        conn.executescript(schema_ddl)
        conn.execute(sql_query)
        return True
    except Exception:
        return False
    finally:
        conn.close()


def query_teacher(schema_ddl: str, question: str) -> str:
    prompt = f"{SYSTEM_PROMPT}\n\nSchema:\n{schema_ddl}\n\nQuestion: {question}\n"
    resp = requests.post(
        OLLAMA_URL,
        json={
            "model": TEACHER_MODEL,
            "prompt": prompt,
            "stream": False,
            "think": False,
            "options": {"temperature": 0.0},
        },
        timeout=60,
    ).json()
    return extract_sql(resp.get("response", ""))


def main():
    with open(INPUT_POOL_PATH, "r", encoding="utf-8") as f:
        train_pool = json.load(f)

    # Check how many rows were already processed if resuming
    processed_count = 0
    if OUTPUT_JSONL_PATH.exists():
        with open(OUTPUT_JSONL_PATH, "r", encoding="utf-8") as f:
            processed_count = sum(1 for _ in f)
        print(f"Resuming from row {processed_count}/{len(train_pool)}...")

    valid_count = processed_count
    rejected_count = 0

    with open(OUTPUT_JSONL_PATH, "a", encoding="utf-8") as out_f:
        for idx, row in enumerate(train_pool[processed_count:], start=processed_count + 1):
            schema = row["context"]
            question = row["question"]

            try:
                teacher_sql = query_teacher(schema, question)
            except Exception as e:
                print(f"[{idx}/{len(train_pool)}] Ollama connection error: {e}")
                break

            if verify_sql_in_sqlite(schema, teacher_sql):
                training_example = {
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": f"Schema:\n{schema}\n\nQuestion: {question}"},
                        {"role": "assistant", "content": teacher_sql},
                        ]
                        }
                out_f.write(json.dumps(training_example) + "\n")
                out_f.flush()
                valid_count += 1
                print(f"[{idx}/{len(train_pool)}] Verified and Saved (Total valid: {valid_count})")
            else:
                rejected_count += 1
                print(f"[{idx}/{len(train_pool)}] Rejected invalid Teacher SQL (Total rejected: {rejected_count})")

    print(f"\nComplete. Saved {valid_count} verified examples to {OUTPUT_JSONL_PATH}")


if __name__ == "__main__":
    main()