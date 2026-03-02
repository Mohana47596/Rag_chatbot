import re

def finance_math_agent(context: str, question: str):
    """
    Extracts correct numeric value from table based on year & metric.
    """

    q = question.lower()

    # Extract year like 20X6-X7
    year_match = re.search(r"(20x\d\s*-\s*x\d)", q)
    year = year_match.group(1).replace(" ", "") if year_match else None

    # Metric
    metric = None
    if "interest" in q:
        metric = "interest"
    elif "net sales" in q or "sales" in q:
        metric = "sales"

    if not year or not metric:
        return ""

    for line in context.split("\n"):
        line_lower = line.lower()

        if metric in line_lower and year.lower() in line_lower:
            numbers = re.findall(r"\d+", line)
            if numbers:
                return f"{metric.title()} for {year} is {numbers[-1]}"

    return ""
