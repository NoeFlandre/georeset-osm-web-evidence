# Mutation Testing

Mutation testing answers a different question than unit testing:

- unit tests ask whether the current code returns expected values;
- mutation tests ask whether the tests fail when production logic is slightly
  broken.

This repository uses `mutmut` through the dev dependency group:

```bash
uv run --group dev mutmut run
uv run --group dev mutmut results
```

The configuration lives in `pyproject.toml`. Mutations target only
`src/georeset_osm_web_evidence`, not scripts or generated data artifacts.

Typical workflow:

1. Run the normal unit tests first.
2. Run `mutmut run` when the normal suite is green.
3. Inspect surviving mutants with `mutmut results` and `mutmut show <mutant>`.
4. Add or strengthen a focused unit test if a surviving mutant represents real
   behavior we care about.

Do not add tests only to kill meaningless mutants. The goal is stronger
behavioral guarantees, not a perfect score.
