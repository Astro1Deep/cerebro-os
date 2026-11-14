"""Recoverable, allowlisted patching for an existing Cerebro beta install."""

from __future__ import annotations

from datetime import datetime, timezone
import shutil
from pathlib import Path

from cerebro_release.installer import REVIEWED_FILES


def _copy_atomically(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    staged = destination.with_name(f".{destination.name}.cerebro-new")
    shutil.copy2(source, staged)
    staged.replace(destination)


def _patch_file(source: Path, destination: Path, relative_path: Path, backup_dir: Path) -> bool:
    if destination.exists():
        backup = backup_dir / relative_path
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(destination, backup)
    _copy_atomically(source, destination)
    return True


def patch_payload(
    *, source_dir: Path,
    target_dir: Path,
    backup_root: Path | None = None,
    dry_run: bool = False,
) -> dict[str, object]:
    """Apply only reviewed code files, never secrets, runtime state or deletions."""

    source_dir = source_dir.resolve()
    target_dir = target_dir.expanduser().resolve()
    if source_dir == target_dir:
        raise ValueError("El paquete del parche y la instalación deben ser distintos")
    if not target_dir.is_dir():
        raise FileNotFoundError("No existe una instalación de Cerebro para actualizar")
    all_sources = [source_dir / name for name in REVIEWED_FILES]
    missing = [str(path.relative_to(source_dir)) for path in all_sources if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"El paquete de parche está incompleto: {', '.join(missing)}")

    chosen_root = backup_root or target_dir / "patch-backups"
    backup_dir = chosen_root / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    relative_files = list(REVIEWED_FILES)
    result: dict[str, object] = {
        "target": str(target_dir),
        "backup_dir": str(backup_dir),
        "updated": [str(path) for path in relative_files],
        "dry_run": dry_run,
    }
    if dry_run:
        return result

    backup_dir.mkdir(parents=True, exist_ok=False)
    for relative_path in relative_files:
        _patch_file(
            source_dir / relative_path,
            target_dir / relative_path,
            Path(relative_path),
            backup_dir,
        )
    return result
