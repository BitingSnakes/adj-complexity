import shutil
import subprocess
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Annotated, Iterable, List, Sequence

import numpy as np
import typer

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


def bits_to_bytes(bits: np.ndarray) -> tuple[bytes, int]:
    if bits.size == 0:
        return b"", 0
    pad_bits = (-bits.size) % 8
    if pad_bits:
        bits = np.pad(bits, (0, pad_bits), constant_values=0)
    return np.packbits(bits).tobytes(), pad_bits


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


def load_packets(
    input_path: str,
    format_name: str,
    encoding: str,
    skip_bytes: int,
    chunk_size: int,
    max_packets: int,
) -> List[np.ndarray]:
    packets: List[np.ndarray] = []
    if format_name == "raw":
        data = read_bytes(input_path)
        data = _skip_bytes(data, skip_bytes)
        for chunk in chunk_bytes(data, chunk_size):
            bits = bytes_to_bits(chunk)
            if bits.size:
                packets.append(bits)
    else:
        lines = read_text_lines(input_path, encoding)
        for line in lines:
            if not line.strip():
                continue
            if format_name == "binary":
                bits = parse_binary_line(line, skip_bytes)
            elif format_name == "hex":
                bits = parse_hex_line(line, skip_bytes)
            else:
                bits = parse_text_line(line, encoding, skip_bytes)
            if bits.size:
                packets.append(bits)

    if max_packets > 0:
        return packets[:max_packets]
    return packets


def format_packet_details(bits: np.ndarray, metrics: PacketMetrics) -> str:
    data, pad_bits = bits_to_bytes(bits)
    lines = [
        f"Packet {metrics.index}",
        f"Flagged: {metrics.flagged}",
        f"Bits: {metrics.bit_length} (padded {pad_bits} bits for byte view)",
        f"Ones: {metrics.ones}",
        f"Density: {metrics.density:.6f}",
        f"Entropy: {metrics.entropy:.6f}",
    ]
    ratio = "inf" if np.isinf(metrics.ratio) else f"{metrics.ratio:.6f}"
    deficiency = "inf" if np.isinf(metrics.deficiency) else f"{metrics.deficiency:.6f}"
    lines.append(f"Ratio: {ratio}")
    lines.append(f"Deficiency: {deficiency}")
    lines.append(f"Hex: {data.hex()}")
    return "\n".join(lines)


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


class InputFormat(str, Enum):
    hex = "hex"
    binary = "binary"
    text = "text"
    raw = "raw"


app = typer.Typer(
    help="Analyze sparse network data with the Adjusted Complexity Ratio."
)


def run_analyze(
    input_path: str,
    format_name: InputFormat,
    encoding: str,
    skip_bytes: int,
    chunk_size: int,
    ratio_min: float,
    entropy_max: float,
    no_flags: bool,
    max_packets: int,
) -> int:
    packets = load_packets(
        input_path=input_path,
        format_name=format_name.value,
        encoding=encoding,
        skip_bytes=skip_bytes,
        chunk_size=chunk_size,
        max_packets=max_packets,
    )

    metrics = analyze_packets(
        packets,
        ratio_min=ratio_min,
        entropy_max=entropy_max,
        apply_flags=not no_flags,
    )
    print(format_metrics(metrics))
    print(summarize(metrics))
    return 0


def run_inspect(
    input_path: str,
    format_name: InputFormat,
    encoding: str,
    skip_bytes: int,
    chunk_size: int,
    ratio_min: float,
    entropy_max: float,
    packet_index: int,
    pcap: str,
    pcap_verbose: bool,
    max_packets: int,
) -> int:
    packets = load_packets(
        input_path=input_path,
        format_name=format_name.value,
        encoding=encoding,
        skip_bytes=skip_bytes,
        chunk_size=chunk_size,
        max_packets=max_packets,
    )

    metrics = analyze_packets(
        packets,
        ratio_min=ratio_min,
        entropy_max=entropy_max,
        apply_flags=True,
    )
    if not metrics:
        print("No packets analyzed.")
        return 0

    if packet_index > 0:
        selected = next(
            (m for m in metrics if m.index == packet_index),
            None,
        )
        if selected is None:
            print(f"Packet index {packet_index} not found.")
            return 1
    else:
        selected = next((m for m in metrics if m.flagged), None)
        if selected is None:
            print("No packets flagged with the current thresholds.")
            return 0

    bits = packets[selected.index - 1]
    print(format_packet_details(bits, selected))
    if pcap:
        tshark = shutil.which("tshark")
        if not tshark:
            print("tshark not found; install Wireshark to decode PCAP packets.")
            return 0
        frame_number = selected.index
        if pcap_verbose:
            cmd = [
                tshark,
                "-r",
                pcap,
                "-Y",
                f"frame.number=={frame_number}",
                "-V",
            ]
        else:
            cmd = [
                tshark,
                "-r",
                pcap,
                "-Y",
                f"frame.number=={frame_number}",
            ]
        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            message = exc.stderr.strip() or exc.stdout.strip()
            print(f"tshark failed: {message or 'unknown error'}")
            return 1
        output = result.stdout.strip()
        if not output:
            print("tshark returned no output for that packet.")
        else:
            print("\nPCAP decode:")
            print(output)
    return 0


def run_demo(
    packet_bytes: int,
    payload_bytes: int,
    seed: int,
) -> int:
    if payload_bytes > packet_bytes:
        raise typer.BadParameter("payload-bytes must be <= packet-bytes")

    rng = np.random.default_rng(seed)
    padding_len = packet_bytes - payload_bytes
    padding = bytes([0] * padding_len)

    payload = rng.integers(
        0,
        256,
        size=payload_bytes,
        dtype=np.uint8,
    ).tobytes()

    packet_padding_only = padding + bytes([0] * payload_bytes)
    packet_with_payload = padding + payload
    packet_random = rng.integers(
        0,
        256,
        size=packet_bytes,
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


@app.command()
def analyze(
    input_path: Annotated[
        str, typer.Option("-i", "--input", help="Input file path or '-' for stdin.")
    ] = "-",
    format_name: Annotated[
        InputFormat,
        typer.Option("-f", "--format", help="Input format for each packet."),
    ] = InputFormat.hex,
    encoding: Annotated[
        str, typer.Option(help="Text encoding when --format=text.")
    ] = "utf-8",
    skip_bytes: Annotated[
        int, typer.Option(help="Skip this many bytes before analysis.")
    ] = 0,
    chunk_size: Annotated[
        int, typer.Option(help="Split raw input into chunks of this many bytes.")
    ] = 0,
    ratio_min: Annotated[
        float, typer.Option(help="Minimum adjusted ratio required to flag a packet.")
    ] = 1.4,
    entropy_max: Annotated[
        float, typer.Option(help="Maximum entropy required to flag a packet.")
    ] = 0.35,
    no_flags: Annotated[
        bool,
        typer.Option("--no-flags", help="Disable flagging logic.", is_flag=True),
    ] = False,
    max_packets: Annotated[
        int, typer.Option(help="Stop after analyzing this many packets.")
    ] = 0,
) -> None:
    code = run_analyze(
        input_path=input_path,
        format_name=format_name,
        encoding=encoding,
        skip_bytes=skip_bytes,
        chunk_size=chunk_size,
        ratio_min=ratio_min,
        entropy_max=entropy_max,
        no_flags=no_flags,
        max_packets=max_packets,
    )
    raise typer.Exit(code=code)


@app.command()
def inspect(
    input_path: Annotated[
        str, typer.Option("-i", "--input", help="Input file path or '-' for stdin.")
    ] = "-",
    format_name: Annotated[
        InputFormat,
        typer.Option("-f", "--format", help="Input format for each packet."),
    ] = InputFormat.hex,
    encoding: Annotated[
        str, typer.Option(help="Text encoding when --format=text.")
    ] = "utf-8",
    skip_bytes: Annotated[
        int, typer.Option(help="Skip this many bytes before analysis.")
    ] = 0,
    chunk_size: Annotated[
        int, typer.Option(help="Split raw input into chunks of this many bytes.")
    ] = 0,
    ratio_min: Annotated[
        float, typer.Option(help="Minimum adjusted ratio required to flag a packet.")
    ] = 1.4,
    entropy_max: Annotated[
        float, typer.Option(help="Maximum entropy required to flag a packet.")
    ] = 0.35,
    packet_index: Annotated[
        int,
        typer.Option(
            help="1-based packet index to inspect (defaults to first flagged)."
        ),
    ] = 0,
    pcap: Annotated[
        str, typer.Option(help="Optional PCAP file to decode the selected packet.")
    ] = "",
    pcap_verbose: Annotated[
        bool,
        typer.Option(
            "--pcap-verbose",
            help="Show verbose tshark decode when --pcap is provided.",
            is_flag=True,
        ),
    ] = False,
    max_packets: Annotated[
        int, typer.Option(help="Stop after analyzing this many packets.")
    ] = 0,
) -> None:
    code = run_inspect(
        input_path=input_path,
        format_name=format_name,
        encoding=encoding,
        skip_bytes=skip_bytes,
        chunk_size=chunk_size,
        ratio_min=ratio_min,
        entropy_max=entropy_max,
        packet_index=packet_index,
        pcap=pcap,
        pcap_verbose=pcap_verbose,
        max_packets=max_packets,
    )
    raise typer.Exit(code=code)


@app.command()
def demo(
    packet_bytes: Annotated[
        int, typer.Option(help="Total bytes per synthetic packet.")
    ] = 64,
    payload_bytes: Annotated[
        int, typer.Option(help="Bytes of random payload hidden in the padding.")
    ] = 8,
    seed: Annotated[
        int, typer.Option(help="Random seed for reproducible payloads.")
    ] = 7,
) -> None:
    code = run_demo(
        packet_bytes=packet_bytes,
        payload_bytes=payload_bytes,
        seed=seed,
    )
    raise typer.Exit(code=code)


if __name__ == "__main__":
    app()
