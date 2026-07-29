#!/usr/bin/env python3
"""One-shot downloader for the quality-evaluation model weights.

The benchmark scores image/video quality with three model families whose
weights are NOT committed to git:

  1. LAION aesthetic-v2 MLP head  (sac+logos+ava1-l14-linearMSE.pth)
  2. ImageReward-v1.0             (ImageReward.pt + med_config.json)
  3. CLIP ViT-B/32 and ViT-L/14   (auto-cached by HuggingFace on first use)

Run this ONCE after cloning so `scripts/eval_all.py` can compute aesthetic /
ImageReward / CLIP scores:

    python benchmark/scripts/setup_assets.py

The download targets are the exact paths the evaluation code reads:

  * aesthetic -> benchmark/cache/aesthetic/sac+logos+ava1-l14-linearMSE.pth
        (eval_all.py --aesthetic-weights default; cal_aesthetic.py --weights)
  * ImageReward -> ~/.cache/ImageReward/{ImageReward.pt,med_config.json}
        (ImageReward.load("ImageReward-v1.0") default download_root)

Behaviour:
  * idempotent  -- an asset already on disk is skipped (use --force to redownload)
  * mirror-aware -- honours the HF_ENDPOINT env var for the HuggingFace assets
                    (e.g. export HF_ENDPOINT=https://hf-mirror.com)
  * every asset prints its source URL and a license reminder
  * a failed download raises a clear, actionable error

Only the Python standard library + `requests` + `huggingface_hub` are used
(both already required by requirements/core.txt); no heavy deps are imported.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths -- MUST match what the evaluation code loads.
#   REPO = repository root (this file is <repo>/benchmark/scripts/setup_assets.py)
# --------------------------------------------------------------------------- #
REPO = Path(__file__).resolve().parents[2]

AESTHETIC_DIR = REPO / "benchmark" / "cache" / "aesthetic"
AESTHETIC_FILENAME = "sac+logos+ava1-l14-linearMSE.pth"
AESTHETIC_PATH = AESTHETIC_DIR / AESTHETIC_FILENAME
# improved-aesthetic-predictor stores the head with a URL-encoded '+' ("%2B").
AESTHETIC_URL = (
    "https://github.com/christophschuhmann/improved-aesthetic-predictor/"
    "raw/main/sac%2Blogos%2Bava1-l14-linearMSE.pth"
)
AESTHETIC_LICENSE = "Apache-2.0 (christophschuhmann/improved-aesthetic-predictor)"
AESTHETIC_EXPECTED_BYTES = 3714759  # released head is 3.71 MB; guards truncated pulls.

# ImageReward.load("ImageReward-v1.0") with no download_root uses this dir.
IMAGEREWARD_DIR = Path(os.path.expanduser("~/.cache/ImageReward"))
IMAGEREWARD_REPO_ID = "THUDM/ImageReward"
IMAGEREWARD_FILES = ("ImageReward.pt", "med_config.json")
IMAGEREWARD_LICENSE = "Apache-2.0 (THUDM/ImageReward)"


class AssetError(RuntimeError):
    """Raised when an asset cannot be fetched; message is user-facing."""


def _fmt_mb(num_bytes: int) -> str:
    return f"{num_bytes / (1024 * 1024):.1f} MB"


def hf_endpoint() -> str:
    """Effective HuggingFace endpoint (honours the HF_ENDPOINT mirror env var)."""
    return os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")


# --------------------------------------------------------------------------- #
# 1. LAION aesthetic-v2 MLP head
# --------------------------------------------------------------------------- #
def setup_aesthetic(force: bool = False) -> bool:
    """Fetch the aesthetic MLP head. Returns True if a download happened."""
    print("\n[1/3] LAION aesthetic-v2 predictor head")
    print(f"      source : {AESTHETIC_URL}")
    print(f"      license: {AESTHETIC_LICENSE}")
    print(f"      target : {AESTHETIC_PATH}")

    if AESTHETIC_PATH.is_file() and not force:
        size = AESTHETIC_PATH.stat().st_size
        print(f"      -> already present ({_fmt_mb(size)}), skipping. Use --force to redownload.")
        return False

    try:
        import requests
    except ImportError as exc:  # pragma: no cover - requests is in core.txt
        raise AssetError(
            "The 'requests' package is required to download the aesthetic head.\n"
            "  Install it with: pip install -r benchmark/requirements/core.txt"
        ) from exc

    AESTHETIC_DIR.mkdir(parents=True, exist_ok=True)
    tmp_path = AESTHETIC_PATH.with_suffix(AESTHETIC_PATH.suffix + ".part")
    print("      downloading ...")
    try:
        with requests.get(AESTHETIC_URL, stream=True, timeout=60, allow_redirects=True) as resp:
            resp.raise_for_status()
            written = 0
            with tmp_path.open("wb") as fh:
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    if chunk:
                        fh.write(chunk)
                        written += len(chunk)
    except requests.RequestException as exc:
        _cleanup(tmp_path)
        raise AssetError(
            f"Failed to download the aesthetic head from:\n  {AESTHETIC_URL}\n"
            f"Reason: {exc}\n"
            "If GitHub is unreachable, download the file manually and place it at:\n"
            f"  {AESTHETIC_PATH}"
        ) from exc

    if written < AESTHETIC_EXPECTED_BYTES // 2:
        _cleanup(tmp_path)
        raise AssetError(
            f"Downloaded aesthetic head looks truncated ({written} bytes, "
            f"expected ~{AESTHETIC_EXPECTED_BYTES}). Re-run to retry."
        )

    tmp_path.replace(AESTHETIC_PATH)
    print(f"      -> saved ({_fmt_mb(written)}).")
    return True


# --------------------------------------------------------------------------- #
# 2. ImageReward-v1.0
# --------------------------------------------------------------------------- #
def setup_imagereward(force: bool = False) -> bool:
    """Fetch ImageReward weights into the dir ImageReward.load() reads by default."""
    print("\n[2/3] ImageReward-v1.0")
    endpoint = hf_endpoint()
    print(f"      source : {endpoint}/{IMAGEREWARD_REPO_ID} (files: {', '.join(IMAGEREWARD_FILES)})")
    print(f"      license: {IMAGEREWARD_LICENSE}")
    print(f"      target : {IMAGEREWARD_DIR}")
    if endpoint != "https://huggingface.co":
        print(f"      mirror : using HF_ENDPOINT={endpoint}")

    have_all = all((IMAGEREWARD_DIR / name).is_file() for name in IMAGEREWARD_FILES)
    if have_all and not force:
        print("      -> already present, skipping. Use --force to redownload.")
        return False

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise AssetError(
            "The 'huggingface_hub' package is required to download ImageReward.\n"
            "  It ships with image-reward; install: pip install -r benchmark/requirements/core.txt"
        ) from exc

    IMAGEREWARD_DIR.mkdir(parents=True, exist_ok=True)
    print("      downloading (ImageReward.pt is ~1.7 GB) ...")
    for name in IMAGEREWARD_FILES:
        dest = IMAGEREWARD_DIR / name
        if dest.is_file() and not force:
            print(f"      - {name}: present, skipping.")
            continue
        try:
            # local_dir places the file exactly where ImageReward.load() expects it,
            # matching image-reward's own ImageReward_download(hf_hub_download(...)).
            hf_hub_download(
                repo_id=IMAGEREWARD_REPO_ID,
                filename=name,
                local_dir=str(IMAGEREWARD_DIR),
            )
            print(f"      - {name}: done.")
        except Exception as exc:  # huggingface_hub raises many subclasses
            raise AssetError(
                f"Failed to download '{name}' from {endpoint}/{IMAGEREWARD_REPO_ID}\n"
                f"Reason: {exc}\n"
                "If HuggingFace is unreachable from this host, set a mirror and retry:\n"
                "  export HF_ENDPOINT=https://hf-mirror.com\n"
                "  python benchmark/scripts/setup_assets.py"
            ) from exc
    print(f"      -> saved to {IMAGEREWARD_DIR}.")
    return True


# --------------------------------------------------------------------------- #
# 3. CLIP backbones (informational -- HF auto-caches on first eval)
# --------------------------------------------------------------------------- #
def setup_clip(prefetch: bool = False) -> bool:
    """CLIP weights are auto-cached by transformers on first eval. Optionally prefetch."""
    print("\n[3/3] CLIP backbones (openai/clip-vit-base-patch32, openai/clip-vit-large-patch14)")
    endpoint = hf_endpoint()
    print(f"      source : {endpoint}/openai/clip-vit-base-patch32 and .../clip-vit-large-patch14")
    print("      license: CLIP weights released by OpenAI (see model cards on HuggingFace)")
    print("      target : HuggingFace cache (HF_HOME / ~/.cache/huggingface)")
    if endpoint != "https://huggingface.co":
        print(f"      mirror : using HF_ENDPOINT={endpoint}")

    if not prefetch:
        print("      -> no action needed: transformers downloads these automatically on the")
        print("         first `eval_all.py` run. Pass --prefetch-clip to fetch them now.")
        return False

    print("      prefetching CLIP weights ...")
    clip_names = ("openai/clip-vit-base-patch32", "openai/clip-vit-large-patch14")
    try:
        from transformers import CLIPModel, CLIPProcessor

        for name in clip_names:
            CLIPProcessor.from_pretrained(name)
            CLIPModel.from_pretrained(name)
            print(f"      - {name}: cached.")
    except Exception as exc:
        raise AssetError(
            f"Failed to prefetch CLIP weights.\nReason: {exc}\n"
            "If HuggingFace is unreachable, set a mirror and retry:\n"
            "  export HF_ENDPOINT=https://hf-mirror.com"
        ) from exc
    print("      -> CLIP backbones cached.")
    return True


def _cleanup(path: Path) -> None:
    try:
        if path.exists():
            path.unlink()
    except OSError:
        pass


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download the quality-evaluation model weights (aesthetic, ImageReward, CLIP).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Redownload assets even if they already exist on disk.",
    )
    p.add_argument(
        "--prefetch-clip",
        action="store_true",
        help="Also download the CLIP backbones now (otherwise transformers fetches them lazily).",
    )
    p.add_argument(
        "--only",
        choices=["aesthetic", "imagereward", "clip"],
        default=None,
        help="Fetch only one asset instead of all.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    print("=" * 74)
    print("edge-dit.cpp benchmark -- quality-evaluation asset setup")
    print(f"repo root: {REPO}")
    print(f"HF endpoint: {hf_endpoint()}  (override with HF_ENDPOINT)")
    print("=" * 74)

    steps = {
        "aesthetic": lambda: setup_aesthetic(force=args.force),
        "imagereward": lambda: setup_imagereward(force=args.force),
        "clip": lambda: setup_clip(prefetch=args.prefetch_clip),
    }
    selected = [args.only] if args.only else list(steps)

    downloaded = 0
    try:
        for key in selected:
            if steps[key]():
                downloaded += 1
    except AssetError as exc:
        print("\n[ERROR] " + str(exc), file=sys.stderr)
        return 1

    print("\n" + "=" * 74)
    if downloaded:
        print(f"Done. {downloaded} asset(s) downloaded.")
    else:
        print("Done. All requested assets already present (nothing to download).")
    print("Next: python benchmark/scripts/eval_all.py --results-root <...>")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
