from __future__ import annotations

import hashlib
import os
from pathlib import Path
import tempfile

from .errors import ControlledRecoveryError

TEMP_SUFFIX = ".tmp"


def _real_path(value: str | Path) -> Path:
    return Path(os.path.realpath(str(Path(value).expanduser())))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


class CheckpointManager:
    def __init__(self, root: str | Path):
        if not root:
            raise ValueError("checkpoint root is required")
        self.root = _real_path(root)

    def resolve_owned(self, candidate: str | Path) -> Path:
        candidate_path = Path(candidate).expanduser()
        raw = candidate_path if candidate_path.is_absolute() else self.root / candidate_path
        resolved = _real_path(raw)
        if resolved == self.root:
            raise ControlledRecoveryError(f"checkpoint path must name a file, not the root: {candidate}")
        try:
            resolved.relative_to(self.root)
        except ValueError as exc:
            raise ControlledRecoveryError(f"checkpoint path escapes the checkpoint root: {candidate}") from exc
        return resolved

    def to_relative(self, resolved: Path) -> str:
        return resolved.relative_to(self.root).as_posix()

    def write_bytes(self, relative_name: str, payload: bytes) -> tuple[str, str]:
        target = self.resolve_owned(relative_name)
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=target.name + ".",
            suffix=TEMP_SUFFIX,
            dir=str(target.parent),
        )
        digest = hashlib.sha256()
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                digest.update(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, str(target))
        except BaseException:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
            raise
        return self.to_relative(target), digest.hexdigest()

    def read_verified(self, relative_name: str, expected_sha256: str) -> tuple[Path, str]:
        resolved = self.resolve_owned(relative_name)
        if not resolved.is_file():
            raise ControlledRecoveryError(f"checkpoint file missing: {relative_name}")
        digest = sha256_file(resolved)
        if digest != expected_sha256:
            raise ControlledRecoveryError(
                f"checkpoint hash mismatch for {relative_name}: expected {expected_sha256}, found {digest}"
            )
        return resolved, digest

    def verify(self, relative_name: str, expected_sha256: str) -> bool:
        try:
            self.read_verified(relative_name, expected_sha256)
            return True
        except ControlledRecoveryError:
            return False

    def list_files(self) -> list[str]:
        if not self.root.is_dir():
            return []
        files: list[str] = []
        for current, _, names in os.walk(self.root):
            current_path = Path(current)
            for name in sorted(names):
                full = current_path / name
                if not full.is_file():
                    continue
                if name.endswith(TEMP_SUFFIX):
                    continue
                files.append(_real_path(full).relative_to(self.root).as_posix())
        return files

    def delete(self, relative_name: str) -> None:
        resolved = self.resolve_owned(relative_name)
        if resolved.is_file():
            os.unlink(resolved)