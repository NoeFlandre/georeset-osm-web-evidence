# Tests

The test suite is organized by project domain. Run the full suite from the
repository root with:

```bash
uv run python -m unittest discover -s tests -p 'test_*.py'
```

The same tests can also be collected through `pytest`, which is useful for
coverage and mutation testing:

```bash
uv run --group dev pytest
```

Run coverage when you want to see which production lines are still weakly
exercised:

```bash
uv run --group dev coverage run -m unittest discover -s tests -p 'test_*.py'
uv run --group dev coverage report
```

Run mutation tests when you want to check whether the tests detect small logic
changes in production code:

```bash
uv run --group dev mutmut run
uv run --group dev mutmut results
```

Mutation testing is slower than normal unit tests. Use it before larger
refactors or before trusting a module whose behavior is mostly pure logic.

Each subfolder keeps tests close to the module or pipeline stage it validates.
