# Labeling

Utilities for preparing sentence-level LLM labeling inputs.

This package does not define a concrete LLM provider. It defines the binary
labeling prompt, parses future model responses into `relevant` or `irrelevant`,
builds reviewable prompt rows from sentence candidates, and applies any
caller-provided labeling function to a prompt dataframe.

`llama_cpp.py` is the first concrete provider adapter. It lazy-imports
`llama_cpp` so normal tests and local development do not require the remote
GPU labeling dependency or GGUF weights.
