"""Benchmark system runners."""

from .base import BenchmarkRunner, PreflightResult
from .diffusers import DiffusersRunner
from .edge_dit import EdgeDitRunner
from .stable_diffusion_cpp import StableDiffusionCppRunner

RUNNERS = {
    "edge_dit": EdgeDitRunner,
    "diffusers": DiffusersRunner,
    "stable_diffusion_cpp": StableDiffusionCppRunner,
}

__all__ = [
    "BenchmarkRunner",
    "PreflightResult",
    "RUNNERS",
]
