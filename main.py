import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np

from adjusted_complexity import AdjustedComplexityCalculator


@dataclass(frozen=True)
class PacketMetrics:
    index: int
    bit_length: int
    ones: int
    density: float
    entropy: float
    ratio: float
    deficiency: float
    flagged: bool


def binary_entropy(p: float) -> float:
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -1.0 * (p * np.log2(p) + (1 - p) * np.log2(1 - p))


def bytes_to_bits(data: bytes) -> np.ndarray:
    if not data:
        return np.array([], dtype=np.uint8)
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8))


def _skip_bytes(data: bytes, skip_bytes: int) -> bytes:
    if skip_bytes <= 0:
        return data
    if skip_bytes >= len(data):
        return b""
    return data[skip_bytes:]


def _skip_bits(bits: np.ndarray, skip_bytes: int) -> np.ndarray:
    skip_bits = skip_bytes * 8
    if skip_bits <= 0:
        return bits
    if skip_bits >= bits.size:
        return np.array([], dtype=np.uint8)
    return bits[skip_bits:]


def parse_binary_line(line: str, skip_bytes: int) -> np.ndarray:
    bits: List[int] = []
    for ch in line.strip():
        if ch in ("0", "1"):
            bits.append(1 if ch == "1" else 0)
        elif ch in (" ", "\t", "_", "-", ":", "|"):
            continue
        else:
            raise ValueError(f"Invalid binary character: {ch!r}")
    return _skip_bits(np.array(bits, dtype=np.uint8), skip_bytes)


def parse_hex_line(line: str, skip_bytes: int) -> np.ndarray:
    cleaned = (
        line.strip()
        .lower()
        .replace("0x", "")
        .replace(" ", "")
        .replace("_", "")
        .replace("-", "")
        .replace(":", "")
    )
    if not cleaned:
        return np.array([], dtype=np.uint8)
    if len(cleaned) % 2 != 0:
        raise ValueError("Hex input must have an even number of characters.")
    data = bytes.fromhex(cleaned)
    data = _skip_bytes(data, skip_bytes)
    return bytes_to_bits(data)


def parse_text_line(line: str, encoding: str, skip_bytes: int) -> np.ndarray:
    data = line.rstrip("\n").encode(encoding, errors="replace")
    data = _skip_bytes(data, skip_bytes)
    return bytes_to_bits(data)


def read_text_lines(path: str, encoding: str) -> List[str]:
    if path == "-":
        return sys.stdin.read().splitlines()
    return Path(path).read_text(encoding=encoding, errors="replace").splitlines()


def read_bytes(path: str) -> bytes:
    if path == "-":
        return sys.stdin.buffer.read()
    return Path(path).read_bytes()


def chunk_bytes(data: bytes, chunk_size: int) -> Iterable[bytes]:
    if chunk_size <= 0:
        yield data
        return
    for i in range(0, len(data), chunk_size):
        chunk = data[i : i + chunk_size]
        if chunk:
            yield chunk


def analyze_packets(
    packets: Sequence[np.ndarray],
    ratio_min: float,
    entropy_max: float,
    apply_flags: bool,
) -> List[PacketMetrics]:
    calc = AdjustedComplexityCalculator()
    results: List[PacketMetrics] = []
    for idx, bits in enumerate(packets, start=1):
        if bits.size == 0:
            continue
        ones = int(bits.sum())
        length = int(bits.size)
        density = ones / length
        entropy = binary_entropy(density)
        r_eff, d_eff = calc.compute_unconditional_batch(bits.reshape(1, -1))
        ratio = float(r_eff[0])
        deficiency = float(d_eff[0])
        flagged = apply_flags and ratio >= ratio_min and entropy <= entropy_max
        results.append(
            PacketMetrics(
                index=idx,
                bit_length=length,
                ones=ones,
                density=density,
                entropy=entropy,
                ratio=ratio,
                deficiency=deficiency,
                flagged=flagged,
            )
        )
    return results


def format_metrics(metrics: Sequence[PacketMetrics]) -> str:
    header = "idx  bits  ones  density  entropy  ratio    deficiency  flag"
    lines = [header, "-" * len(header)]
    for m in metrics:
        ratio = "inf" if np.isinf(m.ratio) else f"{m.ratio:.3f}"
        deficiency = "inf" if np.isinf(m.deficiency) else f"{m.deficiency:.3f}"
        flag = "!" if m.flagged else ""
        lines.append(
            f"{m.index:>3}  {m.bit_length:>4}  {m.ones:>4}  "
            f"{m.density:>7.3f}  {m.entropy:>7.3f}  {ratio:>6}  "
            f"{deficiency:>10}  {flag:>4}"
        )
    return "\n".join(lines)


def summarize(metrics: Sequence[PacketMetrics]) -> str:
    if not metrics:
        return "No packets analyzed."
    flagged = sum(1 for m in metrics if m.flagged)
    ratios = [m.ratio for m in metrics if not np.isinf(m.ratio)]
    avg_ratio = sum(ratios) / len(ratios) if ratios else float("inf")
    entropies = [m.entropy for m in metrics]
    avg_entropy = sum(entropies) / len(entropies)
    return (
        f"Packets: {len(metrics)}, "
        f"Flagged: {flagged}, "
        f"Avg Ratio: {avg_ratio:.3f}, "
        f"Avg Entropy: {avg_entropy:.3f}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=("Analyze sparse network data with the Adjusted Complexity Ratio.")
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Compute adjusted complexity metrics for packets.",
    )
    analyze_parser.add_argument(
        "-i",
        "--input",
        default="-",
        help="Input file path or '-' for stdin.",
    )
    analyze_parser.add_argument(
        "-f",
        "--format",
        choices=("hex", "binary", "text", "raw"),
        default="hex",
        help="Input format for each packet.",
    )
    analyze_parser.add_argument(
        "--encoding",
        default="utf-8",
        help="Text encoding when --format=text.",
    )
    analyze_parser.add_argument(
        "--skip-bytes",
        type=int,
        default=0,
        help="Skip this many bytes before analysis.",
    )
    analyze_parser.add_argument(
        "--chunk-size",
        type=int,
        default=0,
        help="Split raw input into chunks of this many bytes.",
    )
    analyze_parser.add_argument(
        "--ratio-min",
        type=float,
        default=1.4,
        help="Minimum adjusted ratio required to flag a packet.",
    )
    analyze_parser.add_argument(
        "--entropy-max",
        type=float,
        default=0.35,
        help="Maximum entropy required to flag a packet.",
    )
    analyze_parser.add_argument(
        "--no-flags",
        action="store_true",
        help="Disable flagging logic.",
    )
    analyze_parser.add_argument(
        "--max-packets",
        type=int,
        default=0,
        help="Stop after analyzing this many packets.",
    )

    demo_parser = subparsers.add_parser(
        "demo",
        help="Generate synthetic sparse packets with and without payloads.",
    )
    demo_parser.add_argument(
        "--packet-bytes",
        type=int,
        default=64,
        help="Total bytes per synthetic packet.",
    )
    demo_parser.add_argument(
        "--payload-bytes",
        type=int,
        default=8,
        help="Bytes of random payload hidden in the padding.",
    )
    demo_parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed for reproducible payloads.",
    )
    return parser


def run_analyze(args: argparse.Namespace) -> int:
    packets: List[np.ndarray] = []
    if args.format == "raw":
        data = read_bytes(args.input)
        data = _skip_bytes(data, args.skip_bytes)
        for chunk in chunk_bytes(data, args.chunk_size):
            bits = bytes_to_bits(chunk)
            if bits.size:
                packets.append(bits)
    else:
        lines = read_text_lines(args.input, args.encoding)
        for line in lines:
            if not line.strip():
                continue
            if args.format == "binary":
                bits = parse_binary_line(line, args.skip_bytes)
            elif args.format == "hex":
                bits = parse_hex_line(line, args.skip_bytes)
            else:
                bits = parse_text_line(line, args.encoding, args.skip_bytes)
            if bits.size:
                packets.append(bits)

    if args.max_packets > 0:
        packets = packets[: args.max_packets]

    metrics = analyze_packets(
        packets,
        ratio_min=args.ratio_min,
        entropy_max=args.entropy_max,
        apply_flags=not args.no_flags,
    )
    print(format_metrics(metrics))
    print(summarize(metrics))
    return 0


def run_demo(args: argparse.Namespace) -> int:
    if args.payload_bytes > args.packet_bytes:
        raise ValueError("payload-bytes must be <= packet-bytes")

    rng = np.random.default_rng(args.seed)
    padding_len = args.packet_bytes - args.payload_bytes
    padding = bytes([0] * padding_len)

    payload = rng.integers(
        0,
        256,
        size=args.payload_bytes,
        dtype=np.uint8,
    ).tobytes()

    packet_padding_only = padding + bytes([0] * args.payload_bytes)
    packet_with_payload = padding + payload
    packet_random = rng.integers(
        0,
        256,
        size=args.packet_bytes,
        dtype=np.uint8,
    ).tobytes()

    packets = [
        bytes_to_bits(packet_padding_only),
        bytes_to_bits(packet_with_payload),
        bytes_to_bits(packet_random),
    ]

    metrics = analyze_packets(
        packets,
        ratio_min=1.4,
        entropy_max=0.35,
        apply_flags=True,
    )
    print(format_metrics(metrics))
    print(
        "Legend: packet 1 = padding only, packet 2 = padding + payload, packet 3 = random"
    )
    print(summarize(metrics))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "analyze":
        return run_analyze(args)
    if args.command == "demo":
        return run_demo(args)
    parser.error("Unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
