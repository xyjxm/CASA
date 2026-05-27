# Public Repository Security Scan

Latest scan date: 2026-05-27

This scan is intended for public repository hygiene. It checks the current Git
snapshot, not private experiment archives outside the repository.

## Commands

```bash
git status --short
git ls-files | rg '(^|/)(outputs|models|__pycache__|node_modules|venv|\\.venv|\\.venv_sim|build|dist|log|logs|target)(/|$)|(^|/)\\.env(\\.|$)|\\.pem$|\\.key$|(^|/)id_rsa(\\.|$)|\\.(ckpt|safetensors|onnx|trt|engine|pt|pth|deb|so|a|o)$'
git ls-files -z | xargs -0 -r du -b | awk '$1>50000000 {print}'
git grep -I -n -E 'BEGIN (RSA|DSA|EC|OPENSSH|PGP) PRIVATE KEY|github_pat|ghp_|gho_|ghu_|ghs_|glpat-|hf_[A-Za-z0-9_-]{20,}|sk-[A-Za-z0-9]{20,}' -- . ':(exclude)**/thirdparty/**'
git grep -I -n -E '(api[_-]?key|access[_-]?token|secret[_-]?key|client[_-]?secret|password|passwd)[[:space:]]*[:=][[:space:]]*[A-Za-z0-9_./+@-]{8,}' -- . ':(exclude)**/thirdparty/**'
git lfs ls-files
```

## Results

| Check | Result |
|---|---|
| `.env`, private keys, PEM/key files | No tracked matches |
| Common token patterns | No tracked matches outside third-party references |
| Hardcoded assignment-style password/secret patterns | No tracked matches outside third-party references |
| `outputs/`, local logs, caches, virtualenvs | No tracked matches |
| Checkpoints and model-weight extensions | No tracked matches |
| Files larger than 50 MB in Git | No tracked matches |
| LFS assets | Meshes, images, GIFs, small replay assets, and docs media are tracked via Git LFS |

## Known Notes

- The repository contains upstream third-party source trees. Some third-party
  headers and docs mention words such as token, password, or private key as API
  names or documentation text; those are excluded from the hardcoded-secret scan.
- The public repository should not contain generated Phase 2-5 datasets. Keep
  those under ignored `outputs/` paths or external artifact storage.
- GitHub secret scanning and branch protection should be enabled in repository
  settings when available.
