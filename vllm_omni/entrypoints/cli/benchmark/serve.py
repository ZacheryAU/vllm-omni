import argparse

from vllm.benchmarks.serve import add_cli_args

from vllm_omni.benchmarks.serve import main
from vllm_omni.entrypoints.cli.benchmark.base import OmniBenchmarkSubcommandBase
from vllm_omni.entrypoints.cli.benchmark.cli_args import (
    add_serve_args,
    preprocess_serve_args,
)


class OmniBenchmarkServingSubcommand(OmniBenchmarkSubcommandBase):
    """The `serve` subcommand for vllm bench."""

    name = "serve"
    help = "Benchmark the online serving throughput. Supports Daily-Omni and Seed-TTS datasets."

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> None:
        add_cli_args(parser)
        add_serve_args(parser)

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        preprocess_serve_args(args)
        main(args)
