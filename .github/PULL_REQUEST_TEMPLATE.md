## Summary

What changes, and why. Link the issue.

## Checklist

- [ ] Tests first: a test fails before the change and passes after.
- [ ] `cargo test --workspace -j 4 -- --test-threads=2` passes.
- [ ] `python3 tests/test_scorers.py` passes (and the adapter tests, if you touched `python/adapters/`).
- [ ] No new path for agents to gradients, optimisers, fitting or search.
- [ ] Paths from `/work` go through `paths::file_under` or `paths::dir_under`.
- [ ] Docs updated (`README.md`, `docs/`, `CHANGELOG.md`) when behaviour changes.
- [ ] Writing style: short active sentences, no em dashes, no semicolons, no banned words (see [CONTRIBUTING.md](https://github.com/Alexander-Aghili/OpenWiSH/blob/main/CONTRIBUTING.md#writing-style)).

## How you tested

Commands you ran, and the run folders or scores you checked.
