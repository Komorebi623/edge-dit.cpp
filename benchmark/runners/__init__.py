"""Benchmark system runners."""

from .base import BenchmarkRunner, PreflightResult
from .diffusers import DiffusersRunner
from .edge_dit import EdgeDitRunner
from .stable_diffusion_cpp import StableDiffusionCppRunner
from .xdit import XditRunner

RUNNERS = {
    "edge_dit": EdgeDitRunner,
    "diffusers": DiffusersRunner,
    "stable_diffusion_cpp": StableDiffusionCppRunner,
    "xdit": XditRunner,
}

__all__ = [
    "BenchmarkRunner",
    "PreflightResult",
    "RUNNERS",
]
