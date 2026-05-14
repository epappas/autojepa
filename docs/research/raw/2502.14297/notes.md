# Sakana evaluation — extraction notes

## The 42% number (load-bearing for AutoJEPA)
- Of all experiments AI Scientist v1 attempted to run, 42% failed due to *coding errors* (Python exceptions, broken imports, type errors).
- Many remaining experiments produced flawed or misleading numerical output.
- This is the empirical floor for "what happens if you let an LLM mutate ML code without a static validator."

## Other relevant findings
- Novelty assessment is unreliable: established concepts (micro-batching for SGD cited as one example) classified as novel.
- Code modifications averaged only +8% characters/iteration — proposer is timid.
- Manuscripts: median 5 citations, 5/34 from 2020+; placeholder text like "Conclusions Here" leaks through; some hallucinated numbers.

## Why this is the falsification reference for AutoJEPA
- AutoJEPA's AST-diff validator + dry-run gate exists *because* the unguarded version of this loop fails 42% of the time in published evaluation.
- Every defense of "validator before GPU" should cite this paper.
- Note: the 42% figure is from AI Scientist v1 (arxiv 2408.06292); v2 (2504.08066) claims improved tree-search but no independent v2 replication has yet refuted the floor.
