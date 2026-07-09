"""Time _classify_question across representative queries for each category."""

import time
from query import _classify_question, get_rag

TEST_CASES = [
    # (expected_category, question)
    ("application", "Can I deduct my home office if I use it 60% for business?"),
    ("application", "What is the maximum 401k contribution limit for someone over 50?"),

    ("survey",      "What tax credits can I claim as a student?"),
    ("survey",      "What deductions are available for small business owners?"),

    ("exception",   "What businesses qualify for the QBI deduction and what limits apply?"),
    ("exception",   "What are the income phase-out ranges for the Roth IRA contribution?"),

    ("definitional","What counts as qualified business income under the IRC?"),
    ("definitional","What is the tax definition of a dependent?"),

    ("procedural",  "How do I elect S-corp status and what is the deadline?"),
    ("procedural",  "What form do I use to report foreign bank accounts?"),

    ("comparison",  "What is the tax difference between an LLC and an S-corp?"),
    ("comparison",  "Traditional IRA vs Roth IRA — how do they compare for someone in a high tax bracket?"),
]

def main():
    rag_system = get_rag()

    print(f"{'#':<3} {'Expected':<14} {'Got':<14} {'Match':<6} {'Time (s)':<10} Question")
    print("-" * 110)

    total_time = 0.0
    correct = 0

    for i, (expected, question) in enumerate(TEST_CASES, 1):
        t0 = time.time()
        got = _classify_question(rag_system, question)
        elapsed = time.time() - t0
        total_time += elapsed
        match = got == expected
        if match:
            correct += 1
        mark = "✓" if match else "✗"
        print(f"{i:<3} {expected:<14} {got:<14} {mark:<6} {elapsed:<10.2f} {question}")

    print("-" * 110)
    print(f"Accuracy: {correct}/{len(TEST_CASES)}   Total: {total_time:.2f}s   Avg: {total_time/len(TEST_CASES):.2f}s/query")

if __name__ == "__main__":
    main()