# Example export: opt-125m, RTN 4-bit, groups of 128

Produced on the CPU with:

```bash
uv run ptq export-mlir --model opt-125m --bits 4 --group-size 128 --score-windows 2 --score-seqlen 512 --out exports/opt125m-rtn4-g128
```

`export.json` and `layers.json` are committed. `model.mlir` (480 MB) is larger than
GitHub's 100 MB file limit, so it is hosted separately:

- **Download `model.mlir`:** https://www.dropbox.com/scl/fi/v7wqm41ji1sc0nae1cgo0/model.mlir?rlkey=9ew03k14f260jdxxb774rd28x&st=zp8j09no&dl=0

From a terminal, straight into this folder (`dl=1` makes Dropbox send the file itself):

```bash
curl -L -o exports/opt125m-rtn4-g128/model.mlir \
  "https://www.dropbox.com/scl/fi/v7wqm41ji1sc0nae1cgo0/model.mlir?rlkey=9ew03k14f260jdxxb774rd28x&st=zp8j09no&dl=1"
```

Put the downloaded file next to this README. Or regenerate it with the command above
(about 15 seconds once the model is downloaded). See `docs/EXPORT-MLIR.md` for the
step-by-step guide.
