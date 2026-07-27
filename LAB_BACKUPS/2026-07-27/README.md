# Microstructure Veer source backup

Created 2026-07-27 from the clean standalone repository at commit `f348607ef1af614cbd1dff32a0665da668654b2d`.

Archive: `microstructure-veers-source-f348607.tar.gz.b64`

Decoded archive SHA-256:

`89a22294916d1658f163d8040af08b35bff5ac0920f5c7350ef52a5c1fce3000`

Contents: 191 tracked source/configuration/documentation/test files, represented by 210 tar members under `microstructure-veers/`. Raw WRDS data, credentials, caches, runtime outputs, and untracked files are excluded.

## Incremental source update

Commit `900e1e047705fe75dca9fe02a68e6b48f3b06b71` is preserved as:

- `microstructure-veers-source-900e1e0.patch`
- `microstructure-veers-source-900e1e0.patch.sha256`

Patch SHA-256:

`7f03b2de69b1af669ca0e8a61af6f9a168a33ecdb103376e350f492d397a6b6e`

The patch is an exact `git format-patch` against parent commit `f348607ef1af614cbd1dff32a0665da668654b2d`. It contains only shareable source, configuration, tests, and the research log. Apply it after extracting the base archive:

```bash
cd microstructure-veers
git init
git apply ../microstructure-veers-source-900e1e0.patch
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
