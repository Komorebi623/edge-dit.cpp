"""External GPU memory monitor based on nvidia-smi."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import shutil
import subprocess
import threading
import time


@dataclass
class GpuSample:
    timestamp_s: float
    index: int
    memory_used_mib: int


class NvidiaSmiMonitor:
    def __init__(
        self,
        output_csv: Path,
        interval_s: float = 0.2,
        visible_devices: list[int] | None = None,
    ) -> None:
        self.output_csv = output_csv
        self.interval_s = interval_s
        self.visible_devices = visible_devices
        self.samples: list[GpuSample] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def available(self) -> bool:
        return shutil.which("nvidia-smi") is not None

    def start(self) -> None:
        if not self.available():
            raise RuntimeError("nvidia-smi is not available")
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self.write_csv()

    def peak_mib(self) -> int | None:
        if not self.samples:
            return None
        return max(sample.memory_used_mib for sample in self.samples)

    def _run(self) -> None:
        while not self._stop.is_set():
            self._collect_once()
            time.sleep(self.interval_s)

    def _collect_once(self) -> None:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        now = time.time()
        for line in output.splitlines():
            if not line.strip():
                continue
            index_s, used_s = [part.strip() for part in line.split(",", 1)]
            index = int(index_s)
            if self.visible_devices is not None and index not in self.visible_devices:
                continue
            self.samples.append(
                GpuSample(
                    timestamp_s=now,
                    index=index,
                    memory_used_mib=int(used_s),
                )
            )

    def write_csv(self) -> None:
        self.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with self.output_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp_s", "gpu_index", "memory_used_mib"])
            for sample in self.samples:
                writer.writerow([sample.timestamp_s, sample.index, sample.memory_used_mib])


def parse_visible_devices(value: str | None) -> list[int] | None:
    if not value:
        return None
    devices: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if item.isdigit():
            devices.append(int(item))
    return devices or None
