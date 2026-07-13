"""Host process memory helpers."""

from __future__ import annotations

from pathlib import Path


def rss_mib(pid: int) -> float | None:
    status = Path(f"/proc/{pid}/status")
    if not status.exists():
        return None
    for line in status.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("VmRSS:"):
            parts = line.split()
            if len(parts) >= 2:
                return int(parts[1]) / 1024.0
    return None


def process_tree_rss_mib(pid: int) -> float | None:
    pids = [pid, *child_pids(pid)]
    total = 0.0
    seen = False
    for item in pids:
        rss = rss_mib(item)
        if rss is None:
            continue
        total += rss
        seen = True
    return total if seen else None


def child_pids(pid: int) -> list[int]:
    children: list[int] = []
    for status in Path("/proc").glob("[0-9]*/status"):
        try:
            current = int(status.parent.name)
            ppid = read_ppid(status)
        except (OSError, ValueError):
            continue
        if ppid == pid:
            children.append(current)
            children.extend(child_pids(current))
    return children


def read_ppid(status: Path) -> int | None:
    for line in status.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("PPid:"):
            parts = line.split()
            if len(parts) >= 2:
                return int(parts[1])
    return None
