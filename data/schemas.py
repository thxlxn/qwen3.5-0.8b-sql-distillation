import json
import random
from pathlib import Path
from datasets import load_dataset

DATA_DIR = Path(__file__).resolve().parent
ds = load_dataset("b-mc2/sql-create-context", split="train")

# Only extract rows with 2+ tables, so at least one JOIN is required across relational tables
multi_table_rows = [row for row in ds if row["context"].count("CREATE TABLE") >= 2]

# Extract unique schemas
unique_schemas = sorted(list({row["context"] for row in multi_table_rows}))

print(f"Dataset contains {len(unique_schemas)} unique 2+ table relational schemas")
print(f"Total relational (Schema + Question + Gold SQL) rows: {len(multi_table_rows)}")

# Perform a strict split (85 train / 15 test)
random.seed(42)
random.shuffle(unique_schemas)

split_idx = int(len(unique_schemas) * 0.85)
train_schemas_set = set(unique_schemas[:split_idx])
test_schemas_set = set(unique_schemas[split_idx:])

train_pool = [row for row in multi_table_rows if row["context"] in train_schemas_set]
test_pool = [row for row in multi_table_rows if row["context"] in test_schemas_set]

print(f"\nSplit Summary")
print(f"Train Pool: {len(train_pool)} rows across {len(train_schemas_set)} unique schemas")
print(f"Test Pool:  {len(test_pool)} rows across {len(test_schemas_set)} unique schemas")

# Save both pools to disk for the distillation and evaluation scripts
with open(DATA_DIR / "train_seed_pool.json", "w", encoding="utf-8") as f:
    json.dump(train_pool, f, indent=2)

with open(DATA_DIR / "test_eval_pool.json", "w", encoding="utf-8") as f:
    json.dump(test_pool, f, indent=2)

print("\nSchema extraction and test/train split complete")