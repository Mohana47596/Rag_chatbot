import re

def extract_numbers_from_table(table):
    numbers = []
    for row in table:
        for cell in row:
            clean_cell = re.sub(r"[^\d\.]", "", cell)
            if clean_cell:
                try:
                    numbers.append(float(clean_cell))
                except:
                    pass
    return numbers
