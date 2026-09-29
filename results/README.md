# Results — start here

Everything this project has measured lives in this folder. If you only read one thing,
open **[by-model/](by-model/README.md)**.

```
results/
├── README.md        ← this guide
├── by-model/        ← START HERE: one plain-language page per model, with charts
├── plots/           ← the charts (PNG), two per model
├── tables/          ← everything in two big tables, for spreadsheets and deep dives
│   ├── summary.md       every number, grouped by model and dataset
│   └── results.csv      one line per run, every column (open in Excel / LibreOffice / pandas)
└── raw/             ← what the program wrote while running; you rarely need to open it
    ├── runs/            one JSON file per run (the source of truth for everything above)
    ├── legacy-runs/     early runs from before the id scheme settled; kept, never counted
    └── timing.json      measured speeds, used by `ptq run --dry-run` to estimate run time
```

Only `raw/` is written during a run. Everything else is **rebuilt from it**:

```bash
uv run ptq aggregate   # raw/runs/ -> tables/ and by-model/
uv run ptq plot        # tables/results.csv -> plots/  (and refreshes by-model/)
```

So never edit `by-model/`, `tables/` or `plots/` by hand — the next rebuild overwrites them.

## The words you will see

| term | meaning |
|---|---|
| **perplexity** | How surprised the model is by real text it has never seen. Lower is better. A perplexity of 10 means the model is, on average, as unsure as if it were picking between 10 equally likely next words. |
| **full precision / fp16** | The original model, every weight stored as a 16-bit number. This is the baseline every other number is compared to. |
| **bits (8, 4, 3, 2)** | How many bits each weight is squeezed into. 4-bit is a quarter the size of fp16. Fewer bits = smaller model, but more damage. |
| **method / algo** | How the squeezing is done. See below. |
| **grouping** | How many weights share one scale factor. *Per row*: a whole row of the weight matrix (thousands of weights) shares one. *Groups of 128* or *64*: every 128 or 64 weights get their own, which costs a little space but fits the numbers much better. In the raw data this is `group_size` (`-1` = per row). |
| **dataset** | The text perplexity is measured on. `wikitext2` = Wikipedia articles, `c4_new` = web pages, `ptb_new` = 1980s Wall Street Journal sentences. These three are what published papers use, so our numbers can be checked against theirs. |
| **published / paper** | The number a research paper reports for the same model, method and setting. Matching it to within ~1% shows the pipeline is correct. |

The four methods, simplest first:

- **RTN (round-to-nearest)** — round every weight to the nearest allowed value. No data needed. The baseline that everything else tries to beat.
- **HQQ** — also data-free, but searches for a better rounding grid than plain min/max.
- **GPTQ** — runs 128 samples of real text through the model, and as it rounds each weight it nudges the not-yet-rounded ones to cancel out the error.
- **AWQ-lite** — runs real text through the model, finds which input channels matter most, and scales the weights so those channels lose less precision when rounded.

## How to read a result

From `by-model/opt-125m.md`, WikiText-2:

| method | grouping | 4-bit |
|---|---|---|
| RTN (plain rounding) | per row | 37.2831 (+34.8%) |
| GPTQ | groups of 128 | 29.4503 (+6.5%) |

The unmodified model scores 27.66. Squeezed to 4 bits with plain rounding it gets 34.8%
worse; with GPTQ and groups of 128, only 6.5% worse, at nearly the same size. That gap is
what this project measures, for every model, method, bit width and dataset.

Rough guide: under 5% worse is hard to notice in use, 5–20% is noticeable, and anything
shown as "N× worse" means the model is effectively broken at that setting.

## What is in a raw run file

Each `raw/runs/<run_id>.json` is one measurement: one model, one method, one bit width, one
dataset. The file name is a short fingerprint of the settings, so running the same settings
again overwrites the same file instead of duplicating it. The fields that matter most:

| field | meaning |
|---|---|
| `model`, `dataset`, `algo`, `bits`, `group_size` | what was measured |
| `ppl` | the result: perplexity |
| `status`, `reason` | `ok`, or `skipped` / `failed` with why (e.g. a group size the model's shape doesn't divide into) |
| `quant_seconds`, `eval_seconds` | how long quantizing and measuring took |
| `peak_vram_gb` | the most GPU memory used |
| `eval_mode` | `resident` (whole model on the GPU) or `streamed` (one layer at a time, for models too big for the GPU) |
| `git_sha`, `versions`, `device_name`, ... | exactly which code, libraries and hardware produced it, so any number can be traced |

For how the numbers are produced end to end, see [HOW-IT-WORKS.md](../HOW-IT-WORKS.md).
For the findings written up, see [docs/RESULTS.md](../docs/RESULTS.md).
