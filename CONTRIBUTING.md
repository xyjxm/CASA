# Contributing to CASA

CASA is developed with a small-team workflow:

- `main` is the stable public branch.
- `dev` is the integration branch.
- New work should use `feature/<short-name>` branches from `dev`.
- `main` is protected for non-admin collaborators: use pull requests and at
  least one approving review before merge.

## Reporting Issues

- Search existing issues first: <https://github.com/xyjxm/CASA/issues>
- Include the CASA phase, script name, command line, expected behavior, and
  observed behavior.
- Do not paste secrets, access tokens, private paths, or large generated logs.

## Pull Requests

1. Sync `dev`.
2. Create a branch: `git checkout -b feature/<short-name> dev`.
3. Keep generated files under ignored `outputs/` paths.
4. Update documentation when behavior, scripts, or phase outputs change.
5. Run the relevant small check before opening the PR.
6. Open a pull request for review.

Recommended checks:

```bash
python check_environment.py
git status --short
git ls-files | rg '(^|/)(outputs|models|__pycache__|node_modules|venv|\\.venv|\\.venv_sim|build|dist|log|logs|target)(/|$)|(^|/)\\.env(\\.|$)|\\.pem$|\\.key$|(^|/)id_rsa(\\.|$)|\\.(ckpt|safetensors|onnx|trt|engine|pt|pth|deb|so|a|o)$'
```

The last command should print nothing for a normal public-source PR.

## Documentation Rules

- CASA overview and research story: `docs/CASA_OVERVIEW.md`.
- Phase workflow and reproducibility: `docs/CASA_PIPELINE.md`.
- CASA package structure: `gear_sonic/casa/README.md`.
- CASA script index: `gear_sonic/scripts/README.md`.

## Naming

Use **CASA** consistently for this project. Use **GEAR-SONIC**, **GR00T**, and
**MotionBricks** only when referring to upstream components or dependencies.

## License

By contributing, you agree that your contributions are licensed under the
repository license in `LICENSE`. Preserve upstream attribution and third-party
notices when editing inherited GR00T / GEAR-SONIC code.
