import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "rag"))

from pipeline import rag_answer

# List of questions to test your finance chatbot
questions = [
    "What are the main financial statements?",
    "Explain working capital management.",
    "What is the time value of money?",
    "How does diversification reduce risk?",
    "What is the difference between assets and liabilities?"
]

for q in questions:
    result = rag_answer(q)
    print("\nQUESTION:", result["question"])
    print("ANSWER:\n", result["answer"])
    print("=================================================")
