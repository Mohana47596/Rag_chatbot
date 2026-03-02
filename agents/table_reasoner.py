# backend/agents/table_reasoner.py

from agents.table_number_extractor import extract_numbers_from_table

def analyze_table(table):
    """
    Basic table reasoning:
    - Finds max, min, sum
    - Detects trends (increase/decrease)
    """
    numbers = extract_numbers_from_table(table)
    if not numbers:
        return "No numeric data found in table."

    summary = {
        "max": max(numbers),
        "min": min(numbers),
        "sum": sum(numbers),
        "average": sum(numbers)/len(numbers)
    }

    # Trend detection (simple example)
    trend = ""
    if numbers == sorted(numbers):
        trend = "Numbers are increasing."
    elif numbers == sorted(numbers, reverse=True):
        trend = "Numbers are decreasing."
    else:
        trend = "No clear trend."

    summary_text = (
        f"Table summary: max={summary['max']}, min={summary['min']}, "
        f"sum={summary['sum']}, avg={summary['average']}. {trend}"
    )
    return summary_text
