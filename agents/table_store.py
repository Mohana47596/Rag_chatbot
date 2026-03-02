import pdfplumber
import json
import os

TABLE_PATH = "rag/vector_store/tables.json"

def extract_tables_from_pdf(pdf_path):
    tables = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_tables = page.extract_tables()
            for table in page_tables:
                if len(table) > 1:
                    headers = table[0]
                    for row in table[1:]:
                        row_data = dict(zip(headers, row))
                        tables.append(row_data)

    os.makedirs(os.path.dirname(TABLE_PATH), exist_ok=True)
    with open(TABLE_PATH, "w") as f:
        json.dump(tables, f, indent=2)
