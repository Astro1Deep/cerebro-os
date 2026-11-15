"""Deterministic allowlisted archive builder for the Cerebro beta."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Final
import zipfile

from cerebro_release.installer import FILES, REVIEWED_FILES


ENTRYPOINT_FILES: Final[tuple[str, ...]] = (
    "install.py",
    "patch.py",
    "install.sh",
    "patch.sh",
    "install.ps1",
    "patch.ps1",
)

PORTABLE_LAUNCHER = '''#!/usr/bin/env bash
set -euo pipefail
stack_dir=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$stack_dir"
python_bin="$stack_dir/venv/bin/python"
if [ ! -x "$python_bin" ]; then python_bin=python3; fi
exec "$python_bin" -B -m cerebro_cli "$@"
'''


def _source_paths(source_dir: Path, *, patch_only: bool) -> list[Path]:
    names = list(FILES)
    names.extend(name for name in ENTRYPOINT_FILES if not patch_only or not name.startswith("install"))
    unique_names = sorted(set((*names, *REVIEWED_FILES)))
    paths = [source_dir / name for name in unique_names]
    missing = [str(path.relative_to(source_dir)) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"No se puede empaquetar; faltan: {', '.join(missing)}")
    return sorted(paths, key=lambda path: path.relative_to(source_dir).as_posix())


def _write_zip(
    source_dir: Path,
    destination: Path,
    root_name: str,
    *,
    patch_only: bool,
    release_version: str,
) -> Path:
    temporary = destination.with_name(f".{destination.name}.new")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in _source_paths(source_dir, patch_only=patch_only):
            relative = path.relative_to(source_dir).as_posix()
            archive_name = f"{root_name}/{relative}"
            if relative == "cerebro":
                info = zipfile.ZipInfo(archive_name)
                info.create_system = 3
                info.external_attr = 0o100755 << 16
                archive.writestr(info, PORTABLE_LAUNCHER)
            elif relative == "VERSION":
                archive.writestr(zipfile.ZipInfo(archive_name), f"{release_version}\n")
            else:
                info = zipfile.ZipInfo(archive_name)
                info.create_system = 3
                info.external_attr = (0o100755 if path.suffix == ".sh" else 0o100644) << 16
                info.compress_type = zipfile.ZIP_DEFLATED
                archive.writestr(info, path.read_bytes(), compresslevel=9)
    temporary.replace(destination)
    return destination


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_archives(source_dir: Path, output_dir: Path, *, release_version: str) -> dict[str, Path]:
    """Build installation and recoverable-patch archives from an exact allowlist."""

    source_dir = source_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    root_name = f"Cerebro-{release_version}"
    installer = _write_zip(
        source_dir,
        output_dir / f"{root_name}-universal.zip",
        root_name,
        patch_only=False,
        release_version=release_version,
    )
    patch = _write_zip(
        source_dir,
        output_dir / f"{root_name}-patch.zip",
        root_name,
        patch_only=True,
        release_version=release_version,
    )
    checksums = output_dir / "SHA256SUMS.txt"
    checksums.write_text(
        f"{_sha256(installer)}  {installer.name}\n{_sha256(patch)}  {patch.name}\n",
        encoding="utf-8",
    )
    return {"installer": installer, "patch": patch, "checksums": checksums}



def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Build Cerebro OS distribution archives")
    parser.add_argument("--output", type=Path, default=Path("dist"))
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[1]
    version = (source / "VERSION").read_text(encoding="utf-8").strip()
    for path in build_archives(source, args.output, release_version=version).values():
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
