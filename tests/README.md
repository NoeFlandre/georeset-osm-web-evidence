# Tests

The test suite is organized by project domain. Run the full suite from the
repository root with:

```bash
uv run python -m unittest discover -s tests -p 'test_*.py'
```

Each subfolder keeps tests close to the module or pipeline stage it validates.

