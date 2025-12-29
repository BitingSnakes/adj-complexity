# Adjusted Complexity Analyzer

This application applies the Adjusted Complexity Ratio `R(x)` from
`adjusted_complexity.py` to sparse network data. It normalizes for padding
by dividing an effective combinatorial complexity estimate by the empirical
entropy of zeros/ones, helping highlight payloads hidden in low-entropy traffic.

## Quick start

Analyze hex-encoded packets (one per line):

```bash
python main.py analyze --format hex --input packets.txt
```

Analyze a raw binary capture and split into fixed-size packets:

```bash
python main.py analyze --format raw --input capture.bin --chunk-size 128 --skip-bytes 42
```

Run a synthetic demo with padding-only vs padding+payload:

```bash
python main.py demo
```

## Input formats

- `hex`: hex string per line (whitespace and `0x` prefixes allowed)
- `binary`: bits per line (`0`/`1`, separators ignored)
- `text`: UTF-8 text per line (encoded to bytes)
- `raw`: read the full file as bytes (optional `--chunk-size`)

## Flagging

By default, the analyzer flags packets that meet both conditions:

- `ratio >= 1.4`
- `entropy <= 0.35`

Adjust these thresholds with `--ratio-min` and `--entropy-max`, or disable
flagging with `--no-flags`.
