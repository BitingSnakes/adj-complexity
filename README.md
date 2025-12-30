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

## Commands

- `analyze`: compute adjusted complexity metrics for packets
- `inspect`: show details for a flagged packet (or a specific index)
- `demo`: generate synthetic sparse packets with and without payloads

Common options:

- `-i/--input`: input file path or `-` for stdin
- `-f/--format`: `hex`, `binary`, `text`, or `raw`
- `--encoding`: text encoding when `--format=text`
- `--skip-bytes`: skip bytes before analysis
- `--chunk-size`: chunk size for `--format=raw`

Inspect-specific options:

- `--packet-index`: 1-based index to inspect (defaults to first flagged)
- `--pcap`: optional PCAP file to decode the selected packet
- `--pcap-verbose`: verbose `tshark` decode output

Demo-specific options:

- `--packet-bytes`: total bytes per synthetic packet
- `--payload-bytes`: bytes of random payload hidden in the padding
- `--seed`: random seed for reproducible payloads

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
