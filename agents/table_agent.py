import os
import json
import re

TABLE_PATH = "rag/vector_store/tables.json"

def query_table(question: str):

    if not os.path.exists(TABLE_PATH):
        return None

    with open(TABLE_PATH) as f:
        tables = json.load(f)

    year_match = re.search(r"(20X\d\s*[-–]\s*X\d)", question, re.IGNORECASE)

    if not year_match:
        return None

    year = year_match.group(1)

    keywords = ["tax", "interest", "net sales"]
    target = next((k for k in keywords if k in question.lower()), None)

    if not target:
        return None

    for row in tables:
        for key, value in row.items():
            if key and year in key and row.get(target.capitalize()):
                return f"{target.capitalize()} for {year} is {row[target.capitalize()]}"

    return None
