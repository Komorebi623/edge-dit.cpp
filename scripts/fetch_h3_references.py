#!/usr/bin/env python3
"""Fetch immutable, selected public GitHub source snapshots; dry-run by default.

Standard library only. Never clones, runs reference code, installs dependencies,
fetches submodules, follows arbitrary URLs, or downloads model weights.
Use --self-test for offline unit tests; --fetch is the explicit network opt-in.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import sys
import tempfile
import time
import unittest
from unittest import mock
import urllib.error
import urllib.parse
import urllib.request
import uuid

WORKSPACE = Path(__file__).resolve().parents[1]
DEFAULT_LOCK = WORKSPACE / "benchmark/h3_cuda/reference-lock.json"
DEFAULT_DEST = WORKSPACE / "_scratch/h3-reference-src"
MIB = 1024 * 1024
OFFICIAL_REPOS = {
    "sglang": "sgl-project/sglang",
    "cache-dit": "vipshop/cache-dit",
    "fastvideo": "hao-ai-lab/FastVideo",
    "runninghub": "RH-RunningHub/MiniMax-H3-MultiGPU-Lightning",
    "vdn": "OpenVDN/vdn-minimax-h3",
    "sageattention": "thu-ml/SageAttention",
    "vllm-omni": "vllm-project/vllm-omni",
    "lightx2v": "ModelTC/LightX2V",
    "lightx2v-turbo": "ModelTC/Minimax-H3-Turbo",
}
ALLOWED_SUFFIXES = {".py", ".cu", ".cuh", ".h", ".hpp", ".cpp", ".c", ".md", ".mdx", ".txt", ".json", ".yaml", ".yml", ".toml", ".in"}
HEX40 = re.compile(r"[0-9a-f]{40}\Z")


class SafetyError(ValueError):
    """Unsafe source, destination, content, or disk budget."""


def checked_source_path(value: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise SafetyError("invalid source path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(p in ("", ".", "..") for p in value.split("/")):
        raise SafetyError(f"non-relative or traversal path: {value}")
    if any(p.startswith(".") for p in path.parts):
        raise SafetyError(f"hidden/git path forbidden: {value}")
    if path.suffix not in ALLOWED_SUFFIXES and path.name not in ("LICENSE", "NOTICE", "COPYING"):
        raise SafetyError(f"non-source/weight/binary path forbidden: {value}")
    return path


def validate_lock(lock: dict) -> None:
    if not isinstance(lock, dict) or lock.get("schema_version") != 1:
        raise SafetyError("unsupported reference-lock schema")
    if not isinstance(lock.get("sources"), list) or not lock["sources"]:
        raise SafetyError("empty source list")
    seen = set()
    for source in lock["sources"]:
        name = source.get("name")
        if name in seen or name not in OFFICIAL_REPOS:
            raise SafetyError(f"duplicate/unknown source: {name}")
        seen.add(name)
        if source.get("repository") != OFFICIAL_REPOS[name]:
            raise SafetyError(f"not an allowlisted official public repository: {name}")
        if not isinstance(source.get("commit"), str) or not HEX40.fullmatch(source["commit"]):
            raise SafetyError(f"immutable full commit required: {name}")
        if source.get("selection") != "selected-files":
            raise SafetyError("only selected snapshots are supported")
        if not isinstance(source.get("files"), list) or not source["files"]:
            raise SafetyError(f"empty file selection: {name}")
        paths = set()
        for item in source["files"]:
            path = str(checked_source_path(item.get("path")))
            if path in paths:
                raise SafetyError(f"duplicate path: {name}/{path}")
            paths.add(path)
            if type(item.get("size")) is not int or item["size"] < 0:
                raise SafetyError("nonnegative Git tree size required")
            blob = item.get("git_blob_sha1")
            if not isinstance(blob, str) or not HEX40.fullmatch(blob):
                raise SafetyError("Git blob identity required")
        license_path = source.get("license", {}).get("path")
        if license_path not in paths:
            raise SafetyError(f"snapshot must include license: {name}")


def checked_destination(value: Path, allowed_root: Path = DEFAULT_DEST) -> Path:
    """No writes outside the designated scratch tree, including via symlinks."""
    path = Path(os.path.abspath(value))
    root = Path(os.path.abspath(allowed_root))
    if not path.is_relative_to(root):
        raise SafetyError(f"destination must be within {root}")
    assert_no_symlink(path)
    return path


def assert_no_symlink(path: Path) -> None:
    for part in (path, *path.parents):
        try:
            mode = part.lstat().st_mode
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(mode):
            raise SafetyError(f"symlink destination refused: {part}")
        if part != path and not stat.S_ISDIR(mode):
            raise SafetyError(f"non-directory parent refused: {part}")


def disk_usage(root: Path) -> int:
    assert_no_symlink(root)
    if not root.exists():
        return 0
    if not root.is_dir():
        raise SafetyError(f"destination is not a directory: {root}")
    total = 0
    for parent, dirs, files in os.walk(root, followlinks=False):
        for name in dirs + files:
            path = Path(parent) / name
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode):
                raise SafetyError(f"symlink inside destination refused: {path}")
            if stat.S_ISREG(info.st_mode):
                total += max(info.st_size, getattr(info, "st_blocks", 0) * 512)
            elif not stat.S_ISDIR(info.st_mode):
                raise SafetyError(f"special file refused: {path}")
    return total


def blob_sha1(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def validate_content(data: bytes, item: dict) -> None:
    if len(data) != item["size"] or blob_sha1(data) != item["git_blob_sha1"]:
        raise SafetyError(f"size/Git blob mismatch: {item['path']}")
    if data.startswith(b"version https://git-lfs.github.com/spec/v1"):
        raise SafetyError(f"Git LFS pointer refused: {item['path']}")
    if b"\0" in data:
        raise SafetyError(f"binary content refused: {item['path']}")


def existing_matches(path: Path, item: dict) -> bool:
    assert_no_symlink(path)
    if not path.exists():
        return False
    if not path.is_file() or path.stat().st_size != item["size"]:
        raise SafetyError(f"existing user file will not be overwritten: {path}")
    validate_content(path.read_bytes(), item)
    return True


def source_url(source: dict, item: dict) -> str:
    # Ignore arbitrary URLs in metadata: construct only the pinned official URL.
    relative = urllib.parse.quote(str(checked_source_path(item["path"])), safe="/")
    return f"https://raw.githubusercontent.com/{source['repository']}/{source['commit']}/{relative}"


class OfficialRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        parsed = urllib.parse.urlsplit(newurl)
        if parsed.scheme != "https" or parsed.hostname != "raw.githubusercontent.com" or parsed.port not in (None, 443):
            raise SafetyError("redirect outside official raw GitHub refused")
        if parsed.username or parsed.password:
            raise SafetyError("credentialed redirect refused")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def download(source: dict, item: dict, *, max_file_bytes: int, timeout: float, retries: int) -> bytes:
    if item["size"] > max_file_bytes:
        raise SafetyError(f"file budget exceeded: {item['path']}")
    opener = urllib.request.build_opener(OfficialRedirects())
    request = urllib.request.Request(source_url(source, item), headers={
        "User-Agent": "edge-dit-h3-reference-fetch/1", "Accept-Encoding": "identity",
    })
    for attempt in range(retries + 1):
        try:
            with opener.open(request, timeout=timeout) as response:
                length = response.headers.get("Content-Length")
                if length is not None and int(length) > min(max_file_bytes, item["size"]):
                    raise SafetyError(f"oversized HTTP response: {item['path']}")
                data = response.read(min(max_file_bytes, item["size"]) + 1)
            validate_content(data, item)
            return data
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            if isinstance(exc, urllib.error.HTTPError) and exc.code not in (408, 429, 500, 502, 503, 504):
                raise
            if attempt == retries:
                raise
            time.sleep(min(2 ** attempt, 4))
    raise AssertionError("unreachable")


def create_once(path: Path, data: bytes) -> None:
    """Atomically publish a complete regular file without replacing anything."""
    assert_no_symlink(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    assert_no_symlink(path.parent)
    fd, temporary_name = tempfile.mkstemp(prefix=".h3-fetch-", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o644)  # Reference scripts are data, not executable.
        os.link(temporary, path)  # Fails if a user file appears during the fetch.
    finally:
        temporary.unlink(missing_ok=True)  # Only our own temporary file.


def make_plan(lock: dict, names: list[str] | None, destination: Path, max_bytes: int, max_file_bytes: int) -> dict:
    validate_lock(lock)
    sources = lock["sources"]
    if names:
        missing = set(names) - {s["name"] for s in sources}
        if missing:
            raise SafetyError(f"unknown source names: {sorted(missing)}")
        sources = [s for s in sources if s["name"] in names]
    used = disk_usage(destination)
    tasks, matched = [], []
    for source in sources:
        for item in source["files"]:
            if item["size"] > max_file_bytes:
                raise SafetyError(f"file budget exceeded: {source['name']}/{item['path']}")
            path = destination / source["name"] / source["commit"] / item["path"]
            if existing_matches(path, item):
                matched.append((source, item, path))
            else:
                tasks.append((source, item, path))
    # Conservative allowance for blocks, directories, receipts, and one staging
    # copy; all existing files in this destination count toward the cap.
    allowance = 64 * 1024 + sum(((i["size"] + 4095) // 4096 + 2) * 4096 for _, i, _ in tasks)
    peak_staging = max((i["size"] for _, i, _ in tasks), default=0)
    projected = used + allowance + peak_staging
    if projected > max_bytes:
        raise SafetyError(f"disk cap exceeded: projected {projected} > {max_bytes} bytes")
    return {"sources": sources, "tasks": tasks, "matched": matched, "used_bytes": used,
            "download_bytes": sum(i["size"] for _, i, _ in tasks), "projected_bytes": projected}


def fetch_plan(plan: dict, destination: Path, lock_digest: str, *, max_bytes: int, max_file_bytes: int,
               timeout: float, retries: int, jobs: int) -> tuple[dict, Path]:
    destination.mkdir(parents=True, exist_ok=True)
    assert_no_symlink(destination)
    guard = destination / ".h3-fetch-active.lock"
    descriptor = os.open(guard, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.close(descriptor)
    receipt = {"schema_version": 1, "lock_sha256": lock_digest, "status": "complete",
               "files": [], "errors": [], "source_execution": False, "weights_downloaded": False}
    try:
        if shutil.disk_usage(destination).free < max(1024 * 1024, plan["projected_bytes"] - plan["used_bytes"]):
            raise SafetyError("insufficient free disk space")
        # Recheck after acquiring our lock; the plan must not silently change.
        for source, item, path in plan["matched"]:
            if not existing_matches(path, item):
                raise SafetyError(f"verified file disappeared before fetch: {path}")
            receipt["files"].append({"source": source["name"], "commit": source["commit"],
                                     **item, "action": "verified-existing"})
        with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
            pending = {pool.submit(download, source, item, max_file_bytes=max_file_bytes,
                                   timeout=timeout, retries=retries): (source, item, path)
                       for source, item, path in plan["tasks"]}
            for future in concurrent.futures.as_completed(pending):
                source, item, path = pending[future]
                try:
                    data = future.result()
                    validate_content(data, item)
                    if disk_usage(destination) + 2 * len(data) + 65536 > max_bytes:
                        raise SafetyError("runtime disk cap exceeded")
                    create_once(path, data)
                    receipt["files"].append({"source": source["name"], "commit": source["commit"],
                                             **item, "sha256": hashlib.sha256(data).hexdigest(), "action": "downloaded"})
                    print(f"OK {source['name']}/{item['path']}", flush=True)
                except Exception as exc:
                    receipt["status"] = "partial"
                    receipt["errors"].append({"source": source["name"], "path": item["path"],
                                              "error": f"{type(exc).__name__}: {exc}"})
                    print(f"FAIL {source['name']}/{item['path']}: {exc}", file=sys.stderr, flush=True)
        failed_names = {entry["source"] for entry in receipt["errors"]}
        for source in plan["sources"]:
            if source["name"] in failed_names:
                continue
            metadata = json.dumps({"lock_sha256": lock_digest, "repository": source["repository"],
                                   "commit": source["commit"], "license": source["license"],
                                   "selection": "selected-files", "complete_repository": False,
                                   "files": source["files"]}, sort_keys=True, indent=2).encode() + b"\n"
            path = destination / source["name"] / source["commit"] / f"SNAPSHOT-{lock_digest[:16]}.json"
            if path.exists():
                assert_no_symlink(path)
                if path.read_bytes() != metadata:
                    raise SafetyError(f"snapshot manifest conflict: {path}")
            else:
                if disk_usage(destination) + 2 * len(metadata) + 65536 > max_bytes:
                    raise SafetyError("manifest disk cap exceeded")
                create_once(path, metadata)
        receipt["files"].sort(key=lambda f: (f["source"], f["path"]))
        encoded = json.dumps(receipt, sort_keys=True, indent=2).encode() + b"\n"
        if disk_usage(destination) + 2 * len(encoded) > max_bytes:
            raise SafetyError("receipt disk cap exceeded")
        log_path = destination / "fetch-receipts" / f"{time.strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex}.json"
        create_once(log_path, encoded)
        return receipt, log_path
    finally:
        guard.unlink()  # Only the lock successfully created by this run.


def positive_bytes(value: str) -> int:
    try:
        number = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected a positive MiB number") from exc
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("expected a finite positive MiB number")
    return max(1, int(number * MIB))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST, help="must remain within the designated scratch root")
    parser.add_argument("--source", action="append", help="repeat to select sources; default all")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--fetch", action="store_true", help="explicitly download public reference sources")
    mode.add_argument("--dry-run", action="store_true", help="default: no network or filesystem writes")
    parser.add_argument("--max-mib", type=positive_bytes, default=64 * MIB)
    parser.add_argument("--max-file-mib", type=positive_bytes, default=2 * MIB)
    parser.add_argument("--jobs", type=int, choices=range(1, 9), default=4)
    parser.add_argument("--timeout", type=float, default=25)
    parser.add_argument("--retries", type=int, choices=range(4), default=1)
    parser.add_argument("--self-test", action="store_true", help="offline stdlib unit tests; no reference code is run")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.self_test:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(FetchTests)
        return 0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        raise SafetyError("timeout must be finite and positive")
    destination = checked_destination(args.dest)
    lock_bytes = args.lock.read_bytes()
    lock = json.loads(lock_bytes)
    plan = make_plan(lock, args.source, destination, args.max_mib, args.max_file_mib)
    print(json.dumps({"mode": "fetch" if args.fetch else "dry-run", "destination": str(destination),
                      "sources": [{"name": s["name"], "commit": s["commit"], "files": len(s["files"])} for s in plan["sources"]],
                      "download_files": len(plan["tasks"]), "verified_existing_files": len(plan["matched"]),
                      "download_bytes": plan["download_bytes"], "projected_disk_bytes": plan["projected_bytes"],
                      "disk_cap_bytes": args.max_mib, "runs_source": False, "downloads_weights": False}, indent=2))
    if not args.fetch:
        return 0
    result, log_path = fetch_plan(plan, destination, hashlib.sha256(lock_bytes).hexdigest(),
                                 max_bytes=args.max_mib, max_file_bytes=args.max_file_mib,
                                 timeout=args.timeout, retries=args.retries, jobs=args.jobs)
    print(f"Receipt: {log_path}\nStatus: {result['status']}; errors: {len(result['errors'])}")
    return 0 if result["status"] == "complete" else 1


class FetchTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="h3-fetch-test-")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name) / "sources"
        self.data = b"# example source\n"
        self.item = {"path": "README.md", "size": len(self.data), "git_blob_sha1": blob_sha1(self.data)}
        self.source = {"name": "sglang", "repository": OFFICIAL_REPOS["sglang"], "commit": "a" * 40,
                       "selection": "selected-files", "license": {"path": "README.md"}, "files": [self.item]}
        self.lock = {"schema_version": 1, "sources": [self.source]}

    def test_default_dryrun_has_no_network_or_writes(self):
        lock = Path(self.directory.name) / "lock.json"
        lock.write_text(json.dumps(self.lock))
        with mock.patch(__name__ + ".checked_destination", return_value=self.root), mock.patch(__name__ + ".download", side_effect=AssertionError("network")), mock.patch("builtins.print"):
            self.assertEqual(main(["--lock", str(lock)]), 0)
        self.assertFalse(self.root.exists())
        self.assertFalse(parse_args([]).fetch)

    def test_valid_real_lock(self):
        validate_lock(json.loads(DEFAULT_LOCK.read_text()))

    def test_reject_unpinned_or_private_repo(self):
        for field, value in (("commit", "main"), ("repository", "someone/private")):
            with self.subTest(field=field):
                old = self.source[field]
                self.source[field] = value
                with self.assertRaises(SafetyError):
                    validate_lock(self.lock)
                self.source[field] = old

    def test_reject_traversal_hidden_and_weights(self):
        for path in ("../escape.py", "/tmp/source.py", "a/../b.py", "a//b.py", ".git/config", "model.safetensors", "foo.gguf", "foo.bin", "a\\b.py"):
            with self.subTest(path=path), self.assertRaises(SafetyError):
                checked_source_path(path)

    def test_blob_hash(self):
        self.assertEqual(blob_sha1(b""), "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391")
        with self.assertRaises(SafetyError):
            validate_content(b"wrong", self.item)

    def test_reject_lfs_pointer_and_binary(self):
        for data in (b"version https://git-lfs.github.com/spec/v1\noid sha256:foo\n", b"x\0y"):
            item = {**self.item, "size": len(data), "git_blob_sha1": blob_sha1(data)}
            with self.assertRaises(SafetyError):
                validate_content(data, item)

    def test_budget_accounts_existing_data(self):
        self.root.mkdir()
        (self.root / "user.txt").write_bytes(b"x" * MIB)
        with self.assertRaises(SafetyError):
            make_plan(self.lock, None, self.root, MIB, MIB)
        with self.assertRaises(SafetyError):
            make_plan(self.lock, None, self.root, 10 * MIB, 1)

    def test_refuse_destination_outside_root_or_symlink(self):
        with self.assertRaises(SafetyError):
            checked_destination(Path(self.directory.name), self.root)
        self.root.symlink_to(Path(self.directory.name), target_is_directory=True)
        with self.assertRaises(SafetyError):
            checked_destination(self.root, self.root)

    def test_atomic_no_overwrite(self):
        path = self.root / "README.md"
        create_once(path, self.data)
        with self.assertRaises(FileExistsError):
            create_once(path, b"replace")
        self.assertEqual(path.read_bytes(), self.data)
        self.assertFalse(path.stat().st_mode & 0o111)
        self.assertFalse(list(self.root.glob(".h3-fetch-*")))

    def test_resume_and_reject_local_edits(self):
        plan = make_plan(self.lock, None, self.root, 4 * MIB, MIB)
        self.assertEqual(len(plan["tasks"]), 1)
        path = plan["tasks"][0][2]
        create_once(path, self.data)
        self.assertEqual(len(make_plan(self.lock, None, self.root, 4 * MIB, MIB)["matched"]), 1)
        path.write_bytes(b"local edit")
        with self.assertRaises(SafetyError):
            make_plan(self.lock, None, self.root, 4 * MIB, MIB)

    def test_source_selection_and_url(self):
        with self.assertRaises(SafetyError):
            make_plan(self.lock, ["unknown"], self.root, 4 * MIB, MIB)
        self.assertEqual(source_url(self.source, self.item), f"https://raw.githubusercontent.com/sgl-project/sglang/{'a' * 40}/README.md")

    def test_fetch_mocked_download_and_receipt(self):
        plan = make_plan(self.lock, None, self.root, 4 * MIB, MIB)
        with mock.patch(__name__ + ".download", return_value=self.data), mock.patch("builtins.print"):
            receipt, log = fetch_plan(plan, self.root, "b" * 64, max_bytes=4 * MIB,
                                      max_file_bytes=MIB, timeout=1, retries=0, jobs=1)
        self.assertEqual(receipt["status"], "complete")
        self.assertEqual(receipt["files"][0]["sha256"], hashlib.sha256(self.data).hexdigest())
        self.assertTrue(log.is_file())
        self.assertFalse((self.root / ".h3-fetch-active.lock").exists())

    def test_failed_fetch_is_partial_and_resumable(self):
        plan = make_plan(self.lock, None, self.root, 4 * MIB, MIB)
        with mock.patch(__name__ + ".download", side_effect=urllib.error.URLError("offline")), mock.patch("builtins.print"):
            receipt, _ = fetch_plan(plan, self.root, "b" * 64, max_bytes=4 * MIB,
                                    max_file_bytes=MIB, timeout=1, retries=0, jobs=1)
        self.assertEqual(receipt["status"], "partial")
        self.assertEqual(len(receipt["errors"]), 1)
        self.assertFalse(plan["tasks"][0][2].exists())

    def test_lock_conflict(self):
        self.root.mkdir()
        guard = self.root / ".h3-fetch-active.lock"
        guard.write_text("user-owned")
        plan = make_plan(self.lock, None, self.root, 4 * MIB, MIB)
        with self.assertRaises(FileExistsError):
            fetch_plan(plan, self.root, "b" * 64, max_bytes=4 * MIB,
                       max_file_bytes=MIB, timeout=1, retries=0, jobs=1)
        self.assertEqual(guard.read_text(), "user-owned")

    def test_http_response_over_budget_rejected(self):
        response = mock.MagicMock()
        response.__enter__.return_value = response
        response.headers = {"Content-Length": str(MIB)}
        opener = mock.MagicMock()
        opener.open.return_value = response
        with mock.patch("urllib.request.build_opener", return_value=opener), self.assertRaises(SafetyError):
            download(self.source, self.item, max_file_bytes=MIB, timeout=1, retries=0)
        response.read.assert_not_called()

    def test_tampered_download_not_published(self):
        plan = make_plan(self.lock, None, self.root, 4 * MIB, MIB)
        with mock.patch(__name__ + ".download", return_value=b"tampered"), mock.patch("builtins.print"):
            receipt, _ = fetch_plan(plan, self.root, "b" * 64, max_bytes=4 * MIB,
                                    max_file_bytes=MIB, timeout=1, retries=0, jobs=1)
        self.assertEqual(receipt["status"], "partial")
        self.assertFalse(plan["tasks"][0][2].exists())

    def test_nonfinite_disk_limits_rejected(self):
        for value in ("nan", "inf", "0", "-1"):
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                positive_bytes(value)

    def test_redirect_restriction(self):
        handler = OfficialRedirects()
        for url in ("https://example.com/source", "http://raw.githubusercontent.com/file", "https://raw.githubusercontent.com:8443/file"):
            with self.assertRaises(SafetyError):
                handler.redirect_request(None, None, 302, "", {}, url)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (SafetyError, OSError, json.JSONDecodeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(2)
