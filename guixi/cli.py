"""
Command-line interface for GuiXi.

Usage:
    guixi                         # Launch GUI (default)
    guixi gui                     # Launch GUI explicitly
    guixi web --port 8080         # Start web server
    guixi cli --help              # Show CLI help
    guixi server --host 0.0.0.0   # Start inference server
"""

import argparse
import asyncio
import json
import logging
import sys
import time

from . import __version__
from .api import api_cache_stats, api_clear_cache, api_compress, api_infer
from .compress import CompressionMode
from .core import GuiXiServer


def setup_logging(verbose: bool, quiet: bool) -> None:
    """Setup logging based on CLI flags."""
    if quiet:
        level = logging.WARNING
    elif verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO

    logging.basicConfig(
        level=level,
        format="%(levelname)s: %(message)s",
    )


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        prog="guixi",
        description="GuiXi (龟息) - Bandwidth-efficient LLM inference framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  guixi gui                      # Launch GUI (default)
  guixi web --port 8080          # Start web server on port 8080
  guixi infer "What is AI?"       # Run inference from CLI
  guixi bench --prompts data/prompts.txt  # Benchmark bandwidth
  guixi cache --stats            # Show cache statistics
  guixi compress [1,2,3] --mode lz4  # Compress token list

Environment:
  GUIXI_SERVER   Default server URL (default: ws://localhost:8080)
  GUIXI_COMPRESS Default compression mode (default: lz4)
        """,
    )

    parser.add_argument(
        "-V",
        "--version",
        action="version",
        version=f"guixi {__version__}",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose output",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output file for results",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Output results as JSON",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress non-essential output",
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    gui_parser = subparsers.add_parser("gui", help="Launch GUI")
    gui_parser.add_argument("--no-sandbox", action="store_true", help="Disable sandbox")

    web_parser = subparsers.add_parser("web", help="Start web server")
    web_parser.add_argument("--host", default="127.0.0.1", help="Host to bind")
    web_parser.add_argument("--port", type=int, default=5000, help="Port to bind")
    web_parser.add_argument("--debug", action="store_true", help="Debug mode")

    server_parser = subparsers.add_parser("server", help="Start inference server")
    server_parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    server_parser.add_argument("--port", type=int, default=8080, help="Port to bind")
    server_parser.add_argument(
        "--compression", default="lz4", choices=["none", "lz4", "zstd"], help="Compression mode"
    )

    cli_parser = subparsers.add_parser("cli", help="CLI mode")
    cli_parser.add_argument("prompt", help="Input prompt")
    cli_parser.add_argument("--max-tokens", type=int, default=100, help="Max tokens")
    cli_parser.add_argument("--temperature", type=float, default=0.7, help="Temperature")
    cli_parser.add_argument(
        "--compression", default="lz4", choices=["none", "lz4", "zstd"], help="Compression mode"
    )
    cli_parser.add_argument(
        "--cache",
        default="read",
        choices=["none", "read", "write", "read_write"],
        help="Cache policy",
    )
    cli_parser.add_argument("--stats", action="store_true", help="Show bandwidth stats")

    bench_parser = subparsers.add_parser("bench", help="Benchmark bandwidth")
    bench_parser.add_argument("--prompts", required=True, help="Prompts file")
    bench_parser.add_argument("--iterations", type=int, default=100, help="Iterations")
    bench_parser.add_argument(
        "--compression", default="lz4", choices=["none", "lz4", "zstd"], help="Compression mode"
    )
    bench_parser.add_argument(
        "--mode", default="infer", choices=["infer", "compress", "cache"], help="Benchmark mode"
    )

    cache_parser = subparsers.add_parser("cache", help="Cache management")
    cache_parser.add_argument("--stats", action="store_true", help="Show cache stats")
    cache_parser.add_argument("--clear", action="store_true", help="Clear cache")
    cache_parser.add_argument("--size", type=int, help="Set max cache size")

    compress_parser = subparsers.add_parser("compress", help="Compress token data")
    compress_parser.add_argument("tokens", help="Token list as JSON")
    compress_parser.add_argument(
        "--mode", default="lz4", choices=["none", "lz4", "zstd"], help="Compression mode"
    )
    compress_parser.add_argument("--decompress", action="store_true", help="Decompress instead")

    infer_parser = subparsers.add_parser("infer", help="Run inference")
    infer_parser.add_argument("prompt", help="Input prompt")
    infer_parser.add_argument("--max-tokens", type=int, default=100)
    infer_parser.add_argument("--stats", action="store_true", help="Show stats")

    args = parser.parse_args()

    setup_logging(args.verbose, args.quiet)

    if args.command is None:
        run_gui()
        return

    if args.command == "gui":
        run_gui()
    elif args.command == "web":
        run_web(host=args.host, port=args.port, debug=args.debug)
    elif args.command == "server":
        run_server(host=args.host, port=args.port, compression=args.compression)
    elif args.command == "cli":
        run_cli(args)
    elif args.command == "bench":
        run_bench(args)
    elif args.command == "cache":
        run_cache(args)
    elif args.command == "compress":
        run_compress(args)
    elif args.command == "infer":
        run_infer(args)


def run_gui():
    """Launch GUI."""
    try:
        from .gui import main as gui_main

        gui_main()
    except ImportError as e:
        print("Error: PySide6 not installed. Install with: pip install guixi[gui]")
        print(f"Details: {e}")
        sys.exit(1)


def run_web(host: str = "127.0.0.1", port: int = 5000, debug: bool = False):
    """Start web server."""
    from .app import run_server

    print(f"Starting web server at http://{host}:{port}")
    run_server(host=host, port=port, debug=debug)


def run_server(host: str = "0.0.0.0", port: int = 8080, compression: str = "lz4"):
    """Start inference server."""
    comp_mode = CompressionMode(compression)
    server = GuiXiServer(compression=comp_mode)

    async def main():
        print(f"Starting GuiXi server at {host}:{port}")
        await server.start(host=host, port=port)
        print("Server running. Press Ctrl+C to stop.")
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            await server.stop()
            print("Server stopped.")

    asyncio.run(main())


def run_cli(args):
    """Run CLI inference."""

    async def main():
        result = await api_infer(
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            compression=args.compression,
            cache_policy=args.cache,
        )

        if args.json_output:
            print(json.dumps(result.to_dict(), indent=2))
        elif result.success:
            print(result.data.get("text", ""))
            if args.stats:
                stats = result.data.get("stats", {})
                print("\nBandwidth Stats:")
                print(f"  Compression ratio: {stats.get('compression_ratio', 0):.2f}x")
                print(f"  Cache hit rate: {stats.get('cache_hit_rate', 0):.2%}")
                print(f"  Bandwidth savings: {stats.get('bandwidth_savings', 0):.2%}")
        else:
            print(f"Error: {result.error}", file=sys.stderr)
            sys.exit(1)

    asyncio.run(main())


def run_infer(args):
    """Run inference."""

    async def main():
        result = await api_infer(
            prompt=args.prompt,
            max_tokens=args.max_tokens,
        )

        if args.json_output or args.stats:
            print(json.dumps(result.to_dict(), indent=2))
        else:
            print(result.data.get("text", ""))

    asyncio.run(main())


def run_bench(args):
    """Run bandwidth benchmark."""
    try:
        with open(args.prompts, "r") as f:
            prompts = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"Error: Prompts file not found: {args.prompts}")
        sys.exit(1)

    if args.verbose:
        print(f"Benchmarking {len(prompts)} prompts, {args.iterations} iterations each")

    async def main():
        results = []

        for i, prompt in enumerate(prompts[: args.iterations]):
            start = time.time()

            if args.mode == "infer":
                await api_infer(prompt, compression=args.compression)
            elif args.mode == "compress":
                tokens = list(range(100))
                await api_compress(tokens, mode=args.compression)
            else:
                await api_cache_stats()

            elapsed = time.time() - start
            results.append(elapsed)

            if args.verbose and (i + 1) % 10 == 0:
                avg = sum(results) / len(results)
                print(f"  Progress: {i + 1}/{args.iterations}, avg: {avg * 1000:.2f}ms")

        total = sum(results)
        avg = total / len(results)
        print(f"\nBenchmark Results ({args.mode}):")
        print(f"  Total time: {total:.2f}s")
        print(f"  Average: {avg * 1000:.2f}ms")
        print(f"  Throughput: {len(results) / total:.2f} ops/s")

    asyncio.run(main())


def run_cache(args):
    """Cache management."""

    async def main():
        if args.clear:
            result = await api_clear_cache()
            if result.success:
                print("Cache cleared successfully")
            else:
                print(f"Error: {result.error}")
            return

        if args.stats:
            result = await api_cache_stats()
            if args.json_output:
                print(json.dumps(result.data, indent=2))
            else:
                stats = result.data
                print("Cache Statistics:")
                print(f"  Entries: {stats['size']}/{stats['max_size']}")
                print(f"  Hit rate: {stats['hit_rate']:.2%}")
                print(f"  Hits: {stats['hits']}")
                print(f"  Misses: {stats['misses']}")
                print(f"  Total bytes: {stats['total_bytes']}")

    asyncio.run(main())


def run_compress(args):
    """Compress token data."""
    try:
        tokens = json.loads(args.tokens)
        if not isinstance(tokens, list):
            raise ValueError("Tokens must be a list")
    except json.JSONDecodeError:
        print("Error: Invalid JSON for tokens")
        sys.exit(1)

    if args.decompress:
        print("Decompression not yet implemented via CLI")
        return

    async def main():
        result = await api_compress(tokens, mode=args.mode)
        if result.success:
            print(f"Original: {result.data['original_size']} bytes")
            print(f"Compressed: {result.data['compressed_size']} bytes")
            print(f"Ratio: {result.data['ratio']:.2f}x")
            print(f"Savings: {result.data['savings']:.1f}%")
        else:
            print(f"Error: {result.error}")

    asyncio.run(main())


if __name__ == "__main__":
    main()
