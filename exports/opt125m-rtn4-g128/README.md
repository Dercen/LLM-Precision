# Example export: opt-125m, RTN 4-bit, groups of 128

Produced on the CPU with:

```bash
uv run ptq export-mlir --model opt-125m --bits 4 --group-size 128 --score-windows 2 --score-seqlen 512 --out exports/opt125m-rtn4-g128
```

`export.json`, `layers.json` and `model.mlir` are committed. 
