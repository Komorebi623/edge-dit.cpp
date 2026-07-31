from __future__ import annotations

import argparse
from pathlib import Path

from edge_dit import Engine, EngineConfig, VideoRequest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Minimal edge-dit text-to-video example")
    parser.add_argument("--model", required=True, help="Path to a model or diffusers directory")
    parser.add_argument("--prompt", required=True, help="Prompt text")
    parser.add_argument("--negative-prompt", default="", help="Negative prompt text")
    parser.add_argument("--output", default="output.gif", help="Output GIF path")
    parser.add_argument("--backend", default=None, help="Backend name, for example cuda or cpu")
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--frames", type=int, default=17)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--guidance", type=float, default=None, help="Distilled guidance value")
    parser.add_argument("--cfg-scale", type=float, default=None, help="CFG scale")
    parser.add_argument("--eta", type=float, default=None, help="Sampler eta")
    parser.add_argument("--flow-shift", type=float, default=None, help="Flow scheduler shift")
    parser.add_argument("--sampler", default=None, help="Sampler name, for example auto or euler")
    parser.add_argument(
        "--scheduler",
        default=None,
        help="Scheduler name, for example auto or karras",
    )
    parser.add_argument("--threads", type=int, default=None, help="CPU thread count")
    parser.add_argument("--weight-type", default=None, help="Weight type, for example auto or q4_k")
    parser.add_argument("--tensor-type-rules", default=None, help="Tensor type rules string")
    parser.add_argument("--max-vram", type=float, default=None, help="Maximum VRAM budget in GiB")
    parser.add_argument(
        "--offload-to-cpu",
        action="store_true",
        help="Keep model weights on CPU and copy them to GPU as needed",
    )
    parser.add_argument(
        "--keep-text-encoder-on-cpu",
        action="store_true",
        help="Keep text encoders on CPU",
    )
    parser.add_argument(
        "--keep-vae-on-cpu",
        action="store_true",
        help="Keep VAE on CPU",
    )
    parser.add_argument(
        "--flash-attention",
        dest="flash_attention",
        action="store_true",
        default=None,
        help="Enable flash attention",
    )
    parser.add_argument(
        "--no-flash-attention",
        dest="flash_attention",
        action="store_false",
        help="Disable flash attention",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def build_engine_config(args: argparse.Namespace) -> EngineConfig:
    return EngineConfig(
        model_path=args.model,
        backend=args.backend,
        n_threads=args.threads,
        weight_type=args.weight_type,
        tensor_type_rules=args.tensor_type_rules,
        offload_params_to_cpu=args.offload_to_cpu or None,
        text_encoder_offload=args.keep_text_encoder_on_cpu or None,
        vae_offload=args.keep_vae_on_cpu or None,
        flash_attention=args.flash_attention,
        max_vram_gb=args.max_vram,
    )


def build_video_request(args: argparse.Namespace) -> VideoRequest:
    return VideoRequest(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        width=args.width,
        height=args.height,
        frames=args.frames,
        steps=args.steps,
        seed=args.seed,
        guidance=args.guidance,
        cfg_scale=args.cfg_scale,
        eta=args.eta,
        flow_shift=args.flow_shift,
        sampler=args.sampler,
        scheduler=args.scheduler,
    )


def save_frames_as_gif(frames: list[object], output: str) -> None:
    if not frames:
        raise RuntimeError("no frames were generated")

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    first = frames[0]
    rest = list(frames[1:])
    first.save(output_path, save_all=True, append_images=rest, duration=100, loop=0)


def main() -> int:
    args = parse_args()

    config = build_engine_config(args)
    request = build_video_request(args)

    with Engine(config) as engine:
        frames = engine.generate_video(request)
        save_frames_as_gif(frames, args.output)

    print(f"saved video gif to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
