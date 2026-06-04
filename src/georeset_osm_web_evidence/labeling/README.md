# Labeling

Utilities for preparing sentence-level LLM labeling inputs.

This package does not call an LLM. It defines the binary labeling prompt,
parses future model responses into `relevant` or `irrelevant`, and builds
reviewable prompt rows from sentence candidates.
