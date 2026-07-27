# Microstructure Veer source backup

Created 2026-07-27 from the clean standalone repository at commit `f348607ef1af614cbd1dff32a0665da668654b2d`.

Archive: `microstructure-veers-source-f348607.tar.gz.b64`

Decoded archive SHA-256:

`89a22294916d1658f163d8040af08b35bff5ac0920f5c7350ef52a5c1fce3000`

Contents: 191 tracked source/configuration/documentation/test files, represented by 210 tar members under `microstructure-veers/`. Raw WRDS data, credentials, caches, runtime outputs, and untracked files are excluded.

## Restart-safety update

Commit `900e1e047705fe75dca9fe02a68e6b48f3b06b71` is preserved in two forms:

- `microstructure-veers-source-900e1e0.patch`
- `microstructure-veers-900e1e0.bundle.b64`

Patch SHA-256:

`7f03b2de69b1af669ca0e8a61af6f9a168a33ecdb103376e350f492d397a6b6e`

Decoded bundle SHA-256:

`972e422cd3ba9bb6a6e1f806faa0c4be61d1f3d83bdd41d6afa12ceea571e8c0`

The patch is an exact `git format-patch` against parent commit `f348607ef1af614cbd1dff32a0665da668654b2d`. The bundle preserves the exact Git commit object and is the preferred update method for an existing checkout at that parent. Both contain only shareable source, configuration, tests, and the research log.

Apply the exact bundle on macOS:

```bash
base64 -D -i microstructure-veers-900e1e0.bundle.b64 -o microstructure-veers-900e1e0.bundle
git fetch microstructure-veers-900e1e0.bundle HEAD
git merge --ff-only FETCH_HEAD
```

## Tensor-pilot remediation update

Commit `e3183f4e600e2c923f5591b0d8b019269b976b17` is preserved as an incremental update whose prerequisite is commit `900e1e047705fe75dca9fe02a68e6b48f3b06b71`:

- `microstructure-veers-e3183f4.patch`
- `microstructure-veers-e3183f4.bundle.b64`

Patch SHA-256:

`5ae53a23d85126d6c6a80c1a2d7f9d0d01b87ae90f8ee5b9eaf0e18019afd592`

Decoded bundle SHA-256:

`942c38966f18bc382e6bac9d482a4269a60a5999453234329982cdee66383fa8`

Base64 bundle SHA-256:

`e2fdbbb6d48a94aa539b42409f3fe238b975d024b6eb383209eb9bc7dcb1cc5c`

Apply after restoring commit `900e1e0`:

```bash
base64 -D -i microstructure-veers-e3183f4.bundle.b64 -o microstructure-veers-e3183f4.bundle
git fetch microstructure-veers-e3183f4.bundle HEAD
git merge --ff-only FETCH_HEAD
```

The update contains only the Tucker missing-target correction, CP convergence-search expansion, tests, compute accounting, and research log. It contains no licensed data, credentials, caches, or runtime outputs.

## Rank-identification and search-boundary update

Commit `2469f2f3e2a942efaa43c810265d8f9f2847dd3b` is preserved as both an exact one-commit patch and a complete-history Git bundle:

- `microstructure-veers-2469f2f.patch`
- `microstructure-veers-2469f2f.bundle.b64`

Patch SHA-256:

`1cb336edc84771a98d40a9c3a9fd7c635e86f92f91b5677b8ed1187a221824fa`

Decoded bundle SHA-256:

`922c6104b8d07f3e739235492a633422bf2ca18ff6a5c0d3e88f2f38b22986ae`

Base64 bundle SHA-256:

`71cefd764d40f45f2e93978ee92285994847bb9b2f99145b328b82c39d69ffd0`

Restore the complete repository directly on macOS:

```bash
base64 -D -i microstructure-veers-2469f2f.bundle.b64 -o microstructure-veers-2469f2f.bundle
git clone -b codex/intraday-liquidity-veer microstructure-veers-2469f2f.bundle microstructure-veers-2469f2f
git -C microstructure-veers-2469f2f rev-parse HEAD
```

The final command must print `2469f2f3e2a942efaa43c810265d8f9f2847dd3b`. The patch applies to parent commit `e3183f4e600e2c923f5591b0d8b019269b976b17`. This update contains only the rank-aware Tucker output-score correction, preserved failure diagnostics, evidence-based CP/Tucker search expansion, tests, documentation, and research log. It contains no licensed data, credentials, caches, runtime outputs, predictions, or model results.

Apply the source patch after extracting the base archive:

```bash
cd microstructure-veers
git init
git apply ../microstructure-veers-source-900e1e0.patch
git apply ../microstructure-veers-e3183f4.patch
```

## Restore base archive on macOS

```bash
base64 -D -i microstructure-veers-source-f348607.tar.gz.b64 -o microstructure-veers-source-f348607.tar.gz
shasum -a 256 microstructure-veers-source-f348607.tar.gz
tar -xzf microstructure-veers-source-f348607.tar.gz
```

## Restore base archive on Linux

```bash
base64 -d microstructure-veers-source-f348607.tar.gz.b64 > microstructure-veers-source-f348607.tar.gz
sha256sum microstructure-veers-source-f348607.tar.gz
tar -xzf microstructure-veers-source-f348607.tar.gz
```
