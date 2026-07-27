# Tensor essential source and paper backup

Created 2026-07-27 from a read-only copy of the remote Tensor Research working tree. The original repository was not modified.

Archive: `tensor-essential-20260727.tar.gz.b64`

Decoded archive SHA-256:

`d29b8b5076fb1d7049048bf5e2c6d2b7f47e39007cf2fd342771adc0beb3f939`

Contents: 23 selected source, paper, audit, handoff, research-log, run-script, summary, and PDF artifacts that were modified or untracked relative to parent repository commit `0e1e5b4e52d18cd8638a9c445edb7ba67f3f3db5`. Credentials, licensed datasets, caches, journals, and bulk runtime results are excluded.

Restore on macOS:

```bash
base64 -D -i tensor-essential-20260727.tar.gz.b64 -o tensor-essential-20260727.tar.gz
shasum -a 256 tensor-essential-20260727.tar.gz
tar -xzf tensor-essential-20260727.tar.gz
```

Restore on Linux:

```bash
base64 -d tensor-essential-20260727.tar.gz.b64 > tensor-essential-20260727.tar.gz
sha256sum tensor-essential-20260727.tar.gz
tar -xzf tensor-essential-20260727.tar.gz
```
