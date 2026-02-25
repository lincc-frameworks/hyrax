# Hyrax — GitHub Copilot Instructions

> **`HYRAX_GUIDE.md` in the repo root is the canonical reference** for architecture,
> config system, registries, and conventions. This file contains only Copilot-specific
> overrides (primarily timeout handling for async execution). When in doubt, follow
> `HYRAX_GUIDE.md`.

## Critical: Long-Running Commands

**NEVER CANCEL** these commands — they are expected to take minutes, not seconds:

| Command | Typical duration |
|---------|-----------------|
| `echo 'y' \| bash .setup_dev.sh` | 5–15 min |
| `python -m pytest -m "not slow"` | 2–5 min |
| `python -m pytest` | 15–25 min |
| `pre-commit run --all-files` | 3–8 min |

Set timeouts generously (at least 2× the typical duration). If a command appears to
hang, it is almost certainly still working.

**Network Issues:** Installation may fail with ReadTimeoutError due to PyPI connectivity. Retry installation 
multiple times if needed.

## Validation Workflow

After every change, run these three steps in order:

```bash
ruff check src/ tests/ && ruff format src/ tests/
python -m pytest -m "not slow"
pre-commit run --all-files
```

Let the linter fix style issues — do not hand-tune formatting.

## Quick Reference

| Item | Value |
|------|-------|
| **Python version** | ≥ 3.11 |
| **Primary interface** | Jupyter notebooks |
| **Secondary interface** | `hyrax` CLI (for HPC / Slurm) |
| **CLI entry point** | `hyrax = "hyrax_cli.main:main"` |
| **Line length** | 110 (ruff-enforced) |
| **Config format** | TOML (`hyrax_default_config.toml`) |
| **Config override** | `--runtime-config path/to/config.toml` or `-c` |
| **Test markers** | `slow` (integration), unmarked (fast) |

Common verbs: `train`, `infer`, `download`, `prepare`, `umap`, `visualize`, `lookup`,
`save_to_database`, `rebuild_manifest`, `to_onnx`, `test`, `search`, `engine`,
`database_connection`, `model`.

## Key Pitfalls

See `HYRAX_GUIDE.md` for the full list. Key points:

- **`key = false` means `None`** — TOML has no null. Hyrax uses `false` as a sentinel
  for "not set." Code must treat `False` as `None` for these keys.
- **`ConfigDict` is Pydantic's** — the runtime config is an ordinary mutable `dict`,
  not a custom immutable wrapper.
- **Verbs are internal only** — external plugins register models and datasets, not verbs.
- **Manifest files** — ask the user before extending this pattern.
- **Pydantic validation** — do not add to new config sections.

## Repository Layout

```
src/hyrax/              Main package (models, data_sets, verbs, config_schemas, vector_dbs)
src/hyrax_cli/          CLI entry point
tests/hyrax/            Test suite
docs/                   Sphinx documentation
example_notebooks/      Jupyter examples
benchmarks/             ASV performance benchmarks
```

See `HYRAX_GUIDE.md` for detailed structure and architecture.

### Adding New Features
Only skip these if specifically requested by the user, otherwise:

1. **ALWAYS** run full validation first: `python -m pytest -m "not slow"`
2. Make changes in appropriate `src/hyrax/` subdirectory
3. Add tests in `tests/hyrax/test_<name>.py` following existing patterns
4. **ALWAYS** run: `ruff format src/ tests/ && ruff check src/ tests/`
5. **ALWAYS** run: `python -m pytest -m "not slow"` (timeout: 10+ minutes)
6. **ALWAYS** run: `pre-commit run --all-files` (timeout: 15+ minutes)
