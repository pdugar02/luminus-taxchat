"""
Query expansion prompts for each of the six question types.
Each prompt instructs the LLM to generate N targeted IRC search queries.
Imported by query.py — do not import query.py from here.
"""

EXPANSION_PROMPT_APPLICATION = """You are an expert in US tax law (Title 26, Internal Revenue Code). Break down this tax question into 3 targeted search queries that together will surface the specific provisions needed to answer it precisely.

Question: {question}

Think through:
- What is the core tax issue? (deductibility, income recognition, credit eligibility, rate computation, contribution limit, etc.)
- Which IRC section directly governs this? Reference ranges: §§1-59 (rates/brackets), §§61-140 (gross income/exclusions), §§21-45 (credits), §§161-291 (deductions), §§401-436 (retirement), §§1001-1298 (gains/losses).
- Has this area been amended by TCJA 2017, CARES 2020, SECURE 2019/2022, or IRA 2022? If so, phrase query 2 as words that actually appear in the amended text.
- What specific thresholds, phase-outs, or eligibility tests does the answer turn on?

Query 1: The primary statutory rule or rate table that directly governs this issue
Query 2: The current (post-amendment) version of that provision — phrased as content that would appear in that specific amended section, not as a general search for amendments
Query 3: Key dollar thresholds, percentage limits, phase-outs, definitions, or eligibility tests the answer depends on

Return ONLY a valid JSON array of exactly 3 plain strings. No markdown, no code fences, no explanation:
["query 1", "query 2", "query 3"]"""


EXPANSION_PROMPT_SURVEY = """You are an expert in US tax law (Title 26, Internal Revenue Code). This is an open-ended survey question — the goal is to find ALL relevant IRC sections across the code, not just the most obvious one.

Question: {question}

Generate 5 targeted search queries that together cast a wide net. Think about:
- What are the different categories or types of the subject that exist in the tax code?
- Which IRC ranges cover each category? Reference: §§21-30 (dependent/childcare credits), §§25A-25D (education/energy credits), §§32 (EITC), §§35-45 (miscellaneous credits), §§61-140 (income exclusions), §§161-221 (deductions for individuals), §§401-530 (retirement/exempt orgs), §§1001-1400 (capital gains/losses/special rules).
- What eligibility requirements, income limits, or phase-outs govern each type?
- Are there lesser-known or specialized provisions that might also apply?

Each query MUST target a DIFFERENT part of the IRC or a different category — do not write 5 variations of the same query.

Return ONLY a valid JSON array of exactly 5 plain strings. No markdown, no code fences, no explanation:
["query 1", "query 2", "query 3", "query 4", "query 5"]"""


EXPANSION_PROMPT_EXCEPTION = """You are an expert in US tax law (Title 26, Internal Revenue Code). This question asks about conditions, limits, exceptions, or cross-references that constrain a primary tax rule.

Question: {question}

Generate 4 targeted search queries:
Query 1: The primary rule itself — what the general provision allows or requires
Query 2: Dollar caps, percentage limits, income thresholds, or phase-outs that reduce or eliminate the benefit
Query 3: Outright exceptions, exclusions, or disqualifying conditions — which entities, activities, or transactions are carved out or ineligible
Query 4: Cross-referenced IRC sections cited within the primary section that impose additional restrictions (e.g., passive activity rules §469, at-risk rules §465, specified service trades, W-2 wage limits)

Return ONLY a valid JSON array of exactly 4 plain strings. No markdown, no code fences, no explanation:
["query 1", "query 2", "query 3", "query 4"]"""


EXPANSION_PROMPT_DEFINITIONAL = """You are an expert in US tax law (Title 26, Internal Revenue Code). This question asks for the meaning of a legal term or concept under the IRC.

Question: {question}

Generate 2 targeted search queries:
Query 1: The definition itself — search for the section that defines this term (common definition sections: §7701 general definitions, §152 dependents, §162 trade or business, §469 passive activity; or the definition embedded within the relevant substantive section)
Query 2: Edge cases, inclusions, and exclusions — how the definition is applied in practice, what specifically qualifies or doesn't qualify, and any special rules or safe harbors that clarify the boundary of the term

Return ONLY a valid JSON array of exactly 2 plain strings. No markdown, no code fences, no explanation:
["query 1", "query 2"]"""


EXPANSION_PROMPT_PROCEDURAL = """You are an expert in US tax law (Title 26, Internal Revenue Code). This question asks how to comply with a tax requirement — filing, elections, deadlines, or penalties.

Question: {question}

Generate 3 targeted search queries:
Query 1: The procedural requirement or compliance obligation itself (what must be done and by whom)
Query 2: Timing rules — deadlines, due dates, election windows, extension rules, or contribution cutoff dates
Query 3: Consequences of non-compliance — penalties, interest, disqualification, or late-filing remedies (reference §§6651-6724 for penalties, §6501 for statute of limitations)

Return ONLY a valid JSON array of exactly 3 plain strings. No markdown, no code fences, no explanation:
["query 1", "query 2", "query 3"]"""


EXPANSION_PROMPT_COMPARISON = """You are an expert in US tax law (Title 26, Internal Revenue Code). This question asks for a side-by-side comparison of two provisions, entity types, account types, or strategies.

Question: {question}

Generate 4 targeted search queries:
Query 1: The IRC provisions governing the FIRST item being compared (its rules, rates, limits, or treatment)
Query 2: The IRC provisions governing the SECOND item being compared (its rules, rates, limits, or treatment)
Query 3: The key points of difference — how income, contributions, deductions, distributions, or tax treatment differ between the two
Query 4: Eligibility limits, income phase-outs, or conversion/election rules that affect the choice between the two

Return ONLY a valid JSON array of exactly 4 plain strings. No markdown, no code fences, no explanation:
["query 1", "query 2", "query 3", "query 4"]"""
