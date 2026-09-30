"""On-demand AFDB pack reader: pack_index.sqlite → extract triple into a local cache.

Keeps the 6.46M corpus packed on HDD (inode quota) while TrainingDataset loads
flat ``records/`` / ``structures/`` / ``msa_npz/`` paths from a node-local cache.
"""

from __future__ import annotations

import fcntl
import os
import sqlite3
import tarfile
import threading
import time
from pathlib import Path
from typing import Optional


def afdb_member_names(accession: str) -> dict[str, str]:
    """Expected tar member paths for one AFDB accession."""
    return {
        "record": f"records/{accession}.json",
        "structure": f"structures/{accession}.npz",
        "msa": f"msa_npz/{accession}.npz",
    }


def _extract_timeout_s() -> float:
    # Bound cold extraction so a slow shard cannot stall distributed training.
    return float(os.environ.get("BOLTZ_AFDB_EXTRACT_TIMEOUT_S", "45"))


class AfdbPackStore:
    """Resolve accession → tar and ensure the training triple is cached locally."""

    def __init__(
        self,
        pack_index: str | Path,
        cache_dir: str | Path,
        *,
        max_cached: int = 100_000,
    ) -> None:
        self.pack_index = Path(pack_index)
        self.cache_dir = Path(cache_dir)
        self.max_cached = int(max_cached)
        self.records_dir = self.cache_dir / "records"
        self.structures_dir = self.cache_dir / "structures"
        self.msa_dir = self.cache_dir / "msa_npz"
        for d in (self.records_dir, self.structures_dir, self.msa_dir):
            d.mkdir(parents=True, exist_ok=True)

        self._lock = threading.Lock()
        self._tar_locks: dict[str, threading.Lock] = {}
        # Read-only index connection (shared across workers in-process).
        self._con = sqlite3.connect(
            f"file:{self.pack_index}?mode=ro",
            uri=True,
            timeout=60,
            check_same_thread=False,
        )
        self._con.execute("PRAGMA query_only=ON")

    def close(self) -> None:
        try:
            self._con.close()
        except Exception:  # noqa: BLE001
            pass

    def resolve_tar(self, accession: str) -> Path:
        row = self._con.execute(
            "SELECT tar_path FROM pack_index WHERE accession=?",
            (accession,),
        ).fetchone()
        if row is None:
            msg = f"AFDB accession {accession!r} not found in {self.pack_index}"
            raise KeyError(msg)
        return Path(row[0])

    def is_cached(self, accession: str) -> bool:
        return (
            (self.records_dir / f"{accession}.json").exists()
            and (self.structures_dir / f"{accession}.npz").exists()
            and (self.msa_dir / f"{accession}.npz").exists()
        )

    def ensure_cached(
        self,
        accession: str,
        *,
        timeout_s: float | None = None,
    ) -> dict[str, Path]:
        """Extract record/structure/msa for ``accession`` into the cache if needed.

        Raises
        ------
        TimeoutError
            If the extract lock is busy or extract exceeds ``timeout_s``. Callers
            (TrainingDataset) treat this as a skippable sample under DDP.
        """
        paths = {
            "record": self.records_dir / f"{accession}.json",
            "structure": self.structures_dir / f"{accession}.npz",
            "msa": self.msa_dir / f"{accession}.npz",
        }
        if all(p.exists() for p in paths.values()):
            return paths

        budget = _extract_timeout_s() if timeout_s is None else float(timeout_s)
        deadline = time.monotonic() + max(budget, 1.0)
        tar_path = self.resolve_tar(accession)
        members = afdb_member_names(accession)
        rank = os.environ.get("RANK", os.environ.get("LOCAL_RANK", "?"))

        with self._lock:
            if tar_path.as_posix() not in self._tar_locks:
                self._tar_locks[tar_path.as_posix()] = threading.Lock()
            tar_lock = self._tar_locks[tar_path.as_posix()]

        # Don't block forever on another thread in this process extracting the
        # same tar; skip and let TrainingDataset resample.
        if not tar_lock.acquire(blocking=False):
            raise TimeoutError(
                f"AFDB in-process tar lock busy for {accession} ({tar_path.name}); "
                f"rank={rank} skipping"
            )
        try:
            if all(p.exists() for p in paths.values()):
                return paths

            lock_path = self.cache_dir / f".extract_{accession}.lock"
            with lock_path.open("w") as lf:
                # Non-blocking flock: if another rank is extracting this
                # accession, skip instead of waiting through NCCL timeout.
                while True:
                    try:
                        fcntl.flock(lf.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                        break
                    except BlockingIOError as exc:
                        if all(p.exists() for p in paths.values()):
                            return paths
                        if time.monotonic() > deadline:
                            raise TimeoutError(
                                f"AFDB extract lock busy for {accession} "
                                f"({tar_path.name}); rank={rank} after {budget:.0f}s"
                            ) from exc
                        time.sleep(0.05)
                try:
                    if all(p.exists() for p in paths.values()):
                        return paths
                    t0 = time.monotonic()
                    print(
                        f"[afdb-pack] rank={rank} extracting {accession} "
                        f"from {tar_path.name} (timeout={budget:.0f}s)",
                        flush=True,
                    )
                    with tarfile.open(tar_path, "r:gz") as tar:
                        for kind, member in members.items():
                            if time.monotonic() > deadline:
                                raise TimeoutError(
                                    f"AFDB extract timed out for {accession} "
                                    f"after {budget:.0f}s (member={member})"
                                )
                            src = tar.extractfile(member)
                            if src is None:
                                msg = (
                                    f"Missing {member} for {accession} in {tar_path}"
                                )
                                raise FileNotFoundError(msg)
                            dest = paths[kind]
                            dest.parent.mkdir(parents=True, exist_ok=True)
                            tmp = dest.with_suffix(dest.suffix + ".partial")
                            with tmp.open("wb") as out:
                                while True:
                                    if time.monotonic() > deadline:
                                        try:
                                            tmp.unlink(missing_ok=True)
                                        except OSError:
                                            pass
                                        raise TimeoutError(
                                            f"AFDB extract timed out for {accession} "
                                            f"after {budget:.0f}s (writing {kind})"
                                        )
                                    chunk = src.read(8 * 1024 * 1024)
                                    if not chunk:
                                        break
                                    out.write(chunk)
                            os.replace(tmp, dest)
                    print(
                        f"[afdb-pack] rank={rank} extracted {accession} "
                        f"in {time.monotonic() - t0:.1f}s",
                        flush=True,
                    )
                finally:
                    fcntl.flock(lf.fileno(), fcntl.LOCK_UN)
            try:
                lock_path.unlink(missing_ok=True)
            except OSError:
                pass
        finally:
            tar_lock.release()

        return paths


def default_cache_dir(explicit: Optional[str | Path] = None) -> Path:
    """Prefer SLURM/node temp for the extract cache."""
    if explicit is not None:
        return Path(explicit)
    base = os.environ.get("SLURM_TMPDIR") or os.environ.get("TMPDIR") or "/tmp"
    return Path(base) / "afdb_pack_cache"
