"""Terminal interface for PrivRAG-Guard.

Usage:
    python -m privrag attack "patient diabetes medication 12345"
    python -m privrag benchmark --samples 200 --epsilon 1.0
    python -m privrag protect --input raw.npy --output safe.npy --passphrase secret
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np


def _ansi(code: int, text: str) -> str:
    return f"\033[{code}m{text}\033[0m"


def _bold(text: str) -> str:
    return _ansi(1, text)


def _green(text: str) -> str:
    return _ansi(32, text)


def _red(text: str) -> str:
    return _ansi(31, text)


def _dim(text: str) -> str:
    return _ansi(90, text)


def _header(title: str) -> None:
    print(f"\n  {_bold(_green('*'))} {_bold(title)}")
    print(f"  {'-' * 52}")


def cmd_attack(args: argparse.Namespace) -> None:
    from privrag import PrivRAGGuard
    from privrag.attack_simulator import EmbeddingInversionSimulator

    vocabulary = [
        "patient", "diabetes", "medication", "password", "credit",
        "12345", "ssn", "clinic", "report", "confidential",
        "salary", "account", "diagnosis", "treatment", "address",
    ]
    simulator = EmbeddingInversionSimulator(
        dimension=args.dimension, random_state=7,
    ).fit(vocabulary)

    try:
        raw = simulator.encode(args.text)
    except ValueError as exc:
        print(f"  {_red('Error:')} {exc}", file=sys.stderr)
        sys.exit(1)

    guard = PrivRAGGuard(
        epsilon=args.epsilon, noise_fraction=0.10, random_state=42,
    )
    protected = guard.protect_vector(raw)
    result = simulator.compare(raw, protected, top_k=args.top_k)

    _header("Embedding Inversion Attack Simulation")
    print(f"  Input: {_dim(repr(args.text))}\n")

    for label, data, color in [
        ("RAW (unprotected)", result["raw"], _red),
        ("PROTECTED (guarded)", result["protected"], _green),
    ]:
        print(f"  {_bold(label)}  mean confidence: {color(f'{data["mean_confidence"]:.0%}')}")
        for rank, (token, conf) in enumerate(
            zip(data["tokens"], data["confidences"]), 1,
        ):
            bar = "#" * int(conf * 20)
            print(f"    {_dim(f'{rank:02}')}  {token:<16} {bar:<20} {conf:.0%}")
        print()

    drop = result["raw"]["mean_confidence"] - result["protected"]["mean_confidence"]
    print(f"  Confidence reduction: {_green(f'-{drop:.0%}')}\n")


def cmd_benchmark(args: argparse.Namespace) -> None:
    from privrag import PrivRAGGuard

    rng = np.random.default_rng(42)
    topics = rng.normal(size=(6, 128))
    topics /= np.linalg.norm(topics, axis=1, keepdims=True)
    labels = rng.integers(0, 6, size=args.samples)
    corpus = topics[labels] + rng.normal(scale=0.18, size=(args.samples, 128))
    corpus /= np.linalg.norm(corpus, axis=1, keepdims=True)

    guard = PrivRAGGuard(
        epsilon=args.epsilon, noise_fraction=args.noise, random_state=2026,
    )

    t0 = time.perf_counter()
    protected = guard.protect_batch(corpus)
    elapsed = time.perf_counter() - t0

    metrics = guard.benchmark_utility(corpus, protected, k=args.k)

    _header("PrivRAG-Guard Benchmark")
    print(f"  Corpus:    {args.samples} vectors x 128d")
    print(f"  Policy:    e={args.epsilon}  noise={args.noise:.0%}")
    print(f"  Elapsed:   {elapsed*1000:.1f} ms")
    print()
    print(f"  {'Metric':<28} {'Value':>10}")
    print(f"  {'-'*28} {'-'*10}")
    for key, val in metrics.items():
        name = key.replace("_", " ").title()
        print(f"  {name:<28} {_green(f'{val:.4f}'):>10}")
    print()

    ok = metrics["mean_cosine_similarity"] > 0.98
    status = _green("PASS") if ok else _red("FAIL")
    print(f"  Cosine retention >98%: {status}\n")


def cmd_protect(args: argparse.Namespace) -> None:
    from privrag import PrivRAGGuard

    source = Path(args.input)
    if not source.exists():
        print(f"  {_red('Error:')} file not found: {source}", file=sys.stderr)
        sys.exit(1)

    raw = np.load(str(source))
    if raw.ndim != 2:
        print(f"  {_red('Error:')} expected 2D array, got {raw.ndim}D", file=sys.stderr)
        sys.exit(1)

    guard = PrivRAGGuard(
        epsilon=args.epsilon,
        noise_fraction=args.noise,
        passphrase=args.passphrase,
        random_state=42 if args.passphrase is None else None,
    )

    t0 = time.perf_counter()
    protected = guard.protect_batch(raw)
    elapsed = time.perf_counter() - t0

    dest = Path(args.output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(dest), protected)

    _header("Batch Protection Complete")
    print(f"  Input:    {source}  ({raw.shape[0]} x {raw.shape[1]})")
    print(f"  Output:   {dest}")
    print(f"  Elapsed:  {elapsed*1000:.1f} ms")
    print(f"  Key:      {'passphrase-derived' if args.passphrase else 'random seed'}")
    print()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="privrag",
        description="PrivRAG-Guard - embedding privacy middleware CLI",
    )
    sub = parser.add_subparsers(dest="command")

    # attack
    p_attack = sub.add_parser("attack", help="simulate dictionary inversion attack")
    p_attack.add_argument("text", help="sensitive text to embed and attack")
    p_attack.add_argument("--epsilon", type=float, default=1.0)
    p_attack.add_argument("--dimension", type=int, default=128)
    p_attack.add_argument("--top-k", type=int, default=5)

    # benchmark
    p_bench = sub.add_parser("benchmark", help="benchmark retrieval retention")
    p_bench.add_argument("--samples", type=int, default=200)
    p_bench.add_argument("--epsilon", type=float, default=1.0)
    p_bench.add_argument("--noise", type=float, default=0.10)
    p_bench.add_argument("--k", type=int, default=5)

    # protect
    p_prot = sub.add_parser("protect", help="sanitize a .npy embedding batch")
    p_prot.add_argument("--input", required=True, help="input .npy file path")
    p_prot.add_argument("--output", required=True, help="output .npy file path")
    p_prot.add_argument("--passphrase", default=None, help="rotation key passphrase")
    p_prot.add_argument("--epsilon", type=float, default=1.0)
    p_prot.add_argument("--noise", type=float, default=0.10)

    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return

    {"attack": cmd_attack, "benchmark": cmd_benchmark, "protect": cmd_protect}[
        args.command
    ](args)


if __name__ == "__main__":
    main()
