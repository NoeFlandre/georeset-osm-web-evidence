# Labeling

Utilities for preparing sentence-level LLM labeling inputs.

This package does not define a concrete LLM provider. It defines the binary
labeling prompt, parses future model responses into `relevant` or `irrelevant`,
builds reviewable prompt rows from sentence candidates, and applies any
caller-provided labeling function to a prompt dataframe.
