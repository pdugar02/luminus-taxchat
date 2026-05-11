# Evaluate RAG Answer and Suggest Prompt Improvements

You are helping evaluate and improve the luminus-taxchat RAG system.

## Steps

1. Run the evaluator script with the user's question:
   ```
   python eval_improve.py $ARGUMENTS
   ```
   Make sure to run from the project root. If ANTHROPIC_API_KEY is not set, tell the user to export it first.

2. Read the output carefully. It contains:
   - Scores (1–5) on citation accuracy, completeness, step-by-step clarity, legal precision, and appropriate hedging
   - A 2–3 sentence critique
   - Specific suggested edits for EXPANSION_PROMPT, RERANK_PROMPT, and/or ANSWER_PROMPT in query.py

3. Present the results to the user in a clear, readable way. Highlight any scores below 4 with ⚠.

4. For each prompt that has a suggestion, show the **current** prompt text from query.py alongside the **suggested change**, so the user can compare them directly.

5. Ask the user which suggestions (if any) they want to apply. Do not edit query.py until the user explicitly approves a specific change.

6. If the user approves a change, apply it to the relevant prompt in query.py, then confirm what was changed.
