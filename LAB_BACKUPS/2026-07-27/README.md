# Microstructure Veer source backup

Created 2026-07-27 from the clean standalone repository at commit `f348607ef1af614cbd1dff32a0665da668654b2d`.

Archive: `microstructure-veers-source-f348607.tar.gz.b64`

Decoded archive SHA-256:

`89a22294916d1658f163d8040af08b35bff5ac0920f5c7350ef52a5c1fce3000`

Contents: 191 tracked source/configuration/documentation/test files, represented by 210 tar members under `microstructure-veers/`. Raw WRDS data, credentials, caches, runtime outputs, and untracked files are excluded.

Restore on macOS:

```bash
base64 -D -i microstructure-veers-source-f348607.tar.gz.b64 -o microstructure-veers-source-f348607.tar.gz
shasum -a 256 microstructure-veers-source-f348607.tar.gz
tar -xzf microstructure-veers-source-f348607.tar.gz
```

Restore on Linux:

```bash
base64 -d microstructure-veers-source-f348607.tar.gz.b64 > microstructure-veers-source-f348607.tar.gz
sha256sum microstructure-veers-source-f348607.tar.gz
tar -xzf microstructure-veers-source-f348607.tar.gz
```
