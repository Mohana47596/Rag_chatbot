# backend/agents/table_extractor.py

def extract_tables(text: str):
    """
    Extract tables from a text chunk.
    Returns a list of tables. Each table is a list of rows (rows are lists of strings/numbers).
    """
    # This is a stub. Replace with actual table extraction logic from text.
    tables = []

    # Example: detect lines that look like tables (simplest heuristic)
    lines = text.split("\n")
    current_table = []
    for line in lines:
        if "|" in line or "\t" in line:  # pipe or tab-separated
            current_table.append([cell.strip() for cell in line.split("|")])
        else:
            if current_table:
                tables.append(current_table)
                current_table = []
    if current_table:
        tables.append(current_table)

    return tables
