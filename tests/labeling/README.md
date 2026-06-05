# Labeling Tests

Tests for preparing and running LLM-labeling inputs.

- `test_candidates.py` validates sentence-level labeling candidate construction.
- `test_build_labeling_candidates_script.py` covers the script wrapper for candidate export.
- `test_prompt_scaffold.py` checks binary prompt request formatting.
- `test_runner.py` checks labeling runner behavior.
- `test_llama_cpp.py` checks the llama-cpp provider boundary without requiring a real model run.

